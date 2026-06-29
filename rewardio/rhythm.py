import librosa
import mir_eval
import numpy as np
import torch
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from IPython.display import Audio, display
import matplotlib.pyplot as plt

from scipy import signal
from .dsp import normalize
from beat_this.inference import Audio2Beats

# BEAT THIS! singleton (lazy-loaded)
_beat_this_model = None

def _get_beat_this_model():
    """Return a cached Audio2Beats instance (loads model once)."""
    global _beat_this_model
    if _beat_this_model is None:
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _beat_this_model = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    return _beat_this_model


# Helpers moved from evas_eval.py

METER_WEIGHTS = {
    4: np.array([5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1], dtype=int),
    3: np.array([5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1], dtype=int),
    2: np.array([5, 1, 2, 1, 4, 1, 2, 1], dtype=int),
}

def _pattern_from_beats_and_onsets(beat_times, onset_times, beat_positions=None, beats_per_bar=4, steps_per_beat=4):
    """
    Convert beat times and onset times into a quantized pattern grid.
    
    Args:
        beat_times: Array of beat times in seconds
        onset_times: Array of onset times in seconds
        beat_positions: Array of beat positions (1=downbeat, 2, 3, 4...).
                       If provided, uses these to correctly align bars.
        beats_per_bar: Number of beats per bar (default 4 for 4/4)
        steps_per_beat: Subdivisions per beat (default 4 for 16th notes)
    
    Returns:
        grid: Binary array of length (num_bars * beats_per_bar * steps_per_beat)
        num_bars: Number of complete bars
    """
    bt = np.asarray(beat_times, dtype=float)
    ot = np.asarray(onset_times, dtype=float)

    if len(bt) < 2:
        return None, None

    # If beat_positions provided, use them to find bar boundaries
    if beat_positions is not None:
        bp = np.asarray(beat_positions, dtype=int)
        
        # Find indices where beat_position == 1 (downbeats)
        downbeat_indices = np.where(bp == 1)[0]
        
        if len(downbeat_indices) < 2:
            # Fall back to old behavior if not enough downbeats
            downbeat_indices = np.arange(0, len(bt), beats_per_bar)
        
        num_bars = len(downbeat_indices) - 1  # Complete bars only
        if num_bars < 1:
            return None, None
        
        grid_length = num_bars * beats_per_bar * steps_per_beat
        grid = np.zeros(grid_length, dtype=int)
        
        # For each onset, find which bar it belongs to and its position
        for onset_time in ot:
            bar_idx = np.searchsorted(bt[downbeat_indices], onset_time, side="right") - 1
            
            if bar_idx < 0 or bar_idx >= num_bars:
                continue
            
            bar_start_time = bt[downbeat_indices[bar_idx]]
            bar_end_time = bt[downbeat_indices[bar_idx + 1]]
            bar_duration = bar_end_time - bar_start_time
            
            if bar_duration <= 0:
                continue
            
            frac_in_bar = (onset_time - bar_start_time) / bar_duration
            slot_in_bar = int(np.round(frac_in_bar * beats_per_bar * steps_per_beat))
            
            if slot_in_bar >= beats_per_bar * steps_per_beat:
                bar_idx += 1
                slot_in_bar = 0
                if bar_idx >= num_bars:
                    continue
            
            slot = bar_idx * beats_per_bar * steps_per_beat + slot_in_bar
            if 0 <= slot < grid_length:
                grid[slot] = 1
        
        return grid, num_bars
    
    # Fallback: no beat_positions, group every beats_per_bar beats
    num_beats_avail = len(bt) - 1
    num_bars = max(1, num_beats_avail // beats_per_bar)
    last_grid_beat = num_bars * beats_per_bar
    grid_length = num_bars * beats_per_bar * steps_per_beat

    grid = np.zeros(grid_length, dtype=int)

    beat_idx = np.searchsorted(bt, ot, side="right") - 1
    for onset_time, i in zip(ot, beat_idx):
        if i < 0 or i >= len(bt) - 1:
            continue
        if i >= last_grid_beat:
            continue

        beat_start = bt[i]
        beat_end = bt[i + 1]
        beat_dur = beat_end - beat_start
        if beat_dur <= 0:
            continue

        frac = (onset_time - beat_start) / beat_dur
        sub = int(np.round(frac * steps_per_beat))

        if sub == steps_per_beat:
            i += 1
            sub = 0
            if i >= last_grid_beat:
                continue

        slot = i * steps_per_beat + sub
        if 0 <= slot < grid_length:
            grid[slot] = 1

    return grid, num_bars

def _score_pattern(pattern, num_bars, beats_per_bar=4, steps_per_beat=4):
    """(Helper) Calculate Toussaint syncopation score from pattern."""
    if pattern is None or num_bars is None:
        return float("nan")

    w = METER_WEIGHTS.get(beats_per_bar, METER_WEIGHTS[4])
    bar_len = beats_per_bar * steps_per_beat
    pattern = np.asarray(pattern, dtype=int).ravel()

    need = num_bars * bar_len
    if len(pattern) < need:
        return float("nan")

    patt = pattern[:need].reshape(num_bars, bar_len)
    
    syncopation_per_bar = []
    for bar in patt:
        onsets = np.where(bar > 0)[0]
        if len(onsets) == 0:
            syncopation_per_bar.append(0.0)
            continue
        
        bar_sync_score = np.sum(5 - w[onsets]) / len(onsets)
        syncopation_per_bar.append(bar_sync_score)

    # Scale from raw 0-4 range to 0-100
    raw = float(np.mean(syncopation_per_bar))
    return round(raw * 25)


def onset_detection(y, sr, hop_length=512, delta=0.25):
    """
    Detect onsets using Path 2 logic (no HPSS, good for drums).
    
    Args:
        y: Audio time series (monophonic).
        sr: Sample rate.
        hop_length: Hop length.
        delta: Detection threshold (default 0.25).
        
    Returns:
        onset_times: Array of onset times in seconds.
    """
    preroll_sec = 0.010
    pad = int(preroll_sec * sr)
    # Pad start to catch early onsets
    y_pad = np.concatenate([np.zeros(pad, dtype=y.dtype), y])

    env = librosa.onset.onset_strength(y=y_pad, sr=sr, hop_length=hop_length)

    wait_frames = int(0.020 * sr / hop_length)

    onset_frames = librosa.onset.onset_detect(
        y=y_pad,
        onset_envelope=env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        pre_max=int(0.015 * sr / hop_length),
        post_max=int(0.015 * sr / hop_length),
        pre_avg=int(0.080 * sr / hop_length),
        post_avg=int(0.080 * sr / hop_length),
        delta=delta,
        wait=wait_frames,
        backtrack=False
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length) - preroll_sec
    onset_times = onset_times[onset_times >= 0]
    
    return onset_times


def syncopation_score(beat_times, separated_drums, sr=44100, beats_per_bar=4, onset_times=None, drums_sr=None, beat_positions=None):
    """
    Calculate syncopation score using Path 2 logic.
    
    Args:
        beat_times: Array of beat times (seconds) from the original song.
        separated_drums: Audio array of separated drums (can be stereo or mono).
        sr: Sample rate of the song audio.
        beats_per_bar: Number of beats per bar (default 4).
        onset_times: Optional pre-calculated onsets. If None, detects from drums.
        drums_sr: Sample rate of the separated drums. If None, assumes same as sr.
        beat_positions: Array of beat positions (1=downbeat, 2, 3, 4...).
                       Enables correct bar alignment.
        
    Returns:
        tuple: (Syncopation score 0-100, onset_times)
    """
    # If onsets are not provided, detect them
    if onset_times is None:
        drums_y = np.asarray(separated_drums)
        if drums_y.ndim == 2:
            drums_y = np.mean(drums_y, axis=0) # Mix to mono
        
        # Resample drums to match song SR if needed (Demucs often returns 48000 Hz)
        if drums_sr is not None and drums_sr != sr:
            drums_y = librosa.resample(drums_y, orig_sr=drums_sr, target_sr=sr)
        
        # Low-pass filter (from Path 2 logic)
        nyquist = 0.5 * sr
        cutoff_hz = 1500
        b_low, a_low = signal.butter(4, cutoff_hz / nyquist, btype="low")
        drums_lp = normalize(signal.filtfilt(b_low, a_low, drums_y))
        
        # Detect onsets
        onset_times = onset_detection(drums_lp, sr, delta=0.25)
    
    # Quantize onsets to the SONG's beat grid (with downbeat alignment if available)
    pattern, num_bars = _pattern_from_beats_and_onsets(
        beat_times, onset_times,
        beat_positions=beat_positions,
        beats_per_bar=beats_per_bar
    )
    
    score = _score_pattern(pattern, num_bars, beats_per_bar=beats_per_bar)
    return score, onset_times


# Beat detection  (BEAT THIS! default, madmom fallback)

def _derive_positions_and_meter(beat_times, downbeat_times):
    """
    Derive beat positions (1=downbeat, 2, 3, …) and meter from
    beat_times + downbeat_times arrays (as returned by BEAT THIS!).

    Returns:
        beat_positions: array of ints, same length as beat_times
        meter: int (2, 3, or 4)
    """
    bt = np.asarray(beat_times, dtype=float)
    db = np.asarray(downbeat_times, dtype=float)
    positions = np.ones(len(bt), dtype=int)  # default all to 1

    if len(db) < 2:
        # Not enough downbeats — assume 4/4
        for i in range(len(bt)):
            positions[i] = (i % 4) + 1
        return positions, 4

    # For each beat, find which downbeat interval it belongs to
    # and assign position 1..N within that interval
    db_idx = np.searchsorted(db, bt, side="right") - 1

    beats_per_bar_counts = []
    for d in range(len(db) - 1):
        mask = db_idx == d
        bar_beats = np.where(mask)[0]
        if len(bar_beats) > 0:
            beats_per_bar_counts.append(len(bar_beats))
            for j, bi in enumerate(bar_beats):
                positions[bi] = j + 1

    # Handle beats after last downbeat
    mask = db_idx >= len(db) - 1
    bar_beats = np.where(mask)[0]
    if len(bar_beats) > 0:
        for j, bi in enumerate(bar_beats):
            positions[bi] = j + 1

    # Infer meter from the most common beats-per-bar count
    if beats_per_bar_counts:
        from collections import Counter
        meter = Counter(beats_per_bar_counts).most_common(1)[0][0]
    else:
        meter = 4

    # Ensure meter is valid (2, 3, or 4)
    if meter not in [2, 3, 4]:
        meter = 4

    return positions, meter


def detect_beats(y, sr, hop_length=512, use_madmom=False):
    """
    Detect beats, downbeat positions, and meter.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (mono).
    sr : int
        Sample rate.
    hop_length : int
        Hop length for frame conversion.
    use_madmom : bool
        If True, use madmom RNN+DBN pipeline instead of BEAT THIS!.

    Returns
    -------
    beat_times : np.ndarray
        Beat times in seconds.
    beat_frames : np.ndarray
        Beat times converted to frame indices.
    beat_positions : np.ndarray
        Position of each beat within its bar (1=downbeat, 2, 3, 4…).
    meter : int
        Detected beats per bar (2, 3, or 4).
    """
    if use_madmom:
        return _detect_beats_madmom(y, sr, hop_length)
    else:
        return _detect_beats_beat_this(y, sr, hop_length)


def _detect_beats_beat_this(y, sr, hop_length=512):
    """Beat detection using BEAT THIS! (ISMIR 2024)."""

    model = _get_beat_this_model()

    # Audio2Beats expects a 1-D float32 tensor or numpy array
    y_float = y.astype(np.float32) if y.dtype != np.float32 else y

    # Run inference — returns (beats_seconds, downbeats_seconds)
    beat_times, downbeat_times = model(y_float, sr)

    beat_times = np.asarray(beat_times, dtype=float)
    downbeat_times = np.asarray(downbeat_times, dtype=float)

    if len(beat_times) == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int), 4

    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    beat_positions, meter = _derive_positions_and_meter(beat_times, downbeat_times)

    return beat_times, beat_frames, beat_positions, meter


def _detect_beats_madmom(y, sr, hop_length=512):
    """Beat + downbeat detection using madmom (legacy)."""

    # Create processor objects — allow detection of 4/4
    rnn_proc = RNNDownBeatProcessor()
    dbn_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)

    # Mirror-pad the start so madmom has musical context to detect the first beat.
    pad_seconds = 4.0
    pad_samples = int(pad_seconds * sr)
    pad_samples = min(pad_samples, len(y))
    mirror = y[:pad_samples][::-1].copy()
    y_padded = np.concatenate([mirror, y])

    # Get activations from padded audio
    act = rnn_proc(y_padded, sr=sr)

    # Result is Nx2 array: [[time, beat_position], ...]
    result = dbn_proc(act)

    if len(result) == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int), 4

    # Subtract padding offset and keep only beats >= 0
    beat_times = result[:, 0] - pad_seconds
    beat_positions = result[:, 1].astype(int)

    valid = beat_times >= -0.01
    beat_times = beat_times[valid]
    beat_positions = beat_positions[valid]
    beat_times = np.maximum(beat_times, 0.0)

    # Infer meter from max beat position value
    meter = int(np.max(beat_positions))
    if meter not in [2, 3, 4]:
        meter = 4

    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)

    return beat_times, beat_frames, beat_positions, meter


# Keep legacy wrapper so evas_eval.py / evas_debug_pipeline.py still work
def detect_downbeats_with_meter(y, sr, hop_length=512):
    """
    Legacy wrapper — calls detect_beats(use_madmom=True) and returns
    (beat_times, beat_positions, meter) for backward compatibility.
    """
    beat_times, _, beat_positions, meter = detect_beats(y, sr, hop_length, use_madmom=True)
    return beat_times, beat_positions, meter


def get_bpm(beat_times):
    if len(beat_times) < 2:
        return 0

    # Calculate the intervals (time in seconds) between each beat
    beat_intervals = np.diff(beat_times)

    # Get the median interval. We use median instead of mean (average)
    median_interval = np.median(beat_intervals)
    if np.isnan(median_interval) or median_interval == 0:
        return 0

    # Convert the interval (seconds per beat) to BPM (beats per minute)
    # (60 seconds per minute) / (X seconds per beat) = Y beats per minute
    bpm = int(np.round(60.0 / median_interval))

    return bpm

def plot_detected_beats(y, sr, beat_times):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y=y, sr=sr, linewidth=1.0, alpha=0.85)
    plt.vlines(beat_times, 
               ymin=y.min(), ymax=y.max(),
               color="r", linestyles="dashed", 
               alpha=0.8, label="detected beats")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 10)  
    plt.title("Waveform + detected beats")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def sonify_beats(y, sr, beat_times):
    click_track = mir_eval.sonify.clicks(beat_times, 
        								 fs=sr, 
        								 length=len(y))

    # Mix original audio with click track
    mixed_audio = (y * 0.5) + (click_track * 0.5)


    # Normalize to prevent clipping
    mixed_audio_norm = librosa.util.normalize(mixed_audio)

    # Display the audio player in your notebook
    print("Playing audio with detected beats (click track)")
    display(Audio(mixed_audio_norm, rate=sr))