import sys
import os
import csv
import torch
import librosa
from scipy import signal
import numpy as np

sys.path.append('../scripts')

from core import load_audio
from dsp import normalize
from separate import separate
from rhythm import detect_beats


# =========================
# Parsing helpers
# =========================
def parse_stimulus(file_name):
    base = os.path.basename(file_name)
    if not base.lower().endswith(".wav"):
        return None

    stem, _ = os.path.splitext(base)  # "S12-33A"
    if "-" not in stem:
        return None

    subject, code = stem.split("-", 1)  # "S12", "33A"
    subject = subject.upper()

    if len(code) != 3:
        return None

    try:
        x = int(code[0])
        y = int(code[1])
    except ValueError:
        return None

    ab = code[2].upper()
    if x not in (1, 2, 3) or y not in (1, 2, 3, 4, 5) or ab not in ("A", "B"):
        return None

    return subject, x, y, ab


def make_file_name(subject: str, x: int, y: int, ab: str) -> str:
    return f"{subject}-{x}{y}{ab}.wav"


# =========================
# Pattern + scoring helpers
# =========================
def _pattern_from_beats_and_onsets(beat_times, onset_times, beat_positions=None, beats_per_bar=4, steps_per_beat=4):
    """
    Convert beat times and onset times into a quantized pattern grid.
    
    Args:
        beat_times: Array of beat times in seconds
        onset_times: Array of onset times in seconds
        beat_positions: Array of beat positions (1=downbeat, 2, 3, 4...) from madmom.
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
        
        # For each onset, find which bar it belongs to and its position within the bar
        for onset_time in ot:
            # Find which bar this onset belongs to
            bar_idx = np.searchsorted(bt[downbeat_indices], onset_time, side="right") - 1
            
            if bar_idx < 0 or bar_idx >= num_bars:
                continue
            
            # Get the beat times for this bar
            bar_start_beat_idx = downbeat_indices[bar_idx]
            bar_end_beat_idx = downbeat_indices[bar_idx + 1]
            
            bar_start_time = bt[bar_start_beat_idx]
            bar_end_time = bt[bar_end_beat_idx]
            bar_duration = bar_end_time - bar_start_time
            
            if bar_duration <= 0:
                continue
            
            # Calculate position within the bar (0.0 to 1.0)
            frac_in_bar = (onset_time - bar_start_time) / bar_duration
            
            # Convert to slot number (0 to beats_per_bar * steps_per_beat - 1)
            slot_in_bar = int(np.round(frac_in_bar * beats_per_bar * steps_per_beat))
            
            # Handle edge case where onset rounds to next bar
            if slot_in_bar >= beats_per_bar * steps_per_beat:
                bar_idx += 1
                slot_in_bar = 0
                if bar_idx >= num_bars:
                    continue
            
            # Compute global slot
            slot = bar_idx * beats_per_bar * steps_per_beat + slot_in_bar
            if 0 <= slot < grid_length:
                grid[slot] = 1
        
        return grid, num_bars
    
    # Original behavior when beat_positions not provided
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


# Toussaint/L&J Hierarchical Weights by meter
# Higher weight = stronger metrical position
METER_WEIGHTS = {
    # 4/4 time: 16 pulses (4 beats x 4 subdivisions)
    4: np.array([5, 1, 2, 1,   # beat 1 (downbeat strongest)
                 3, 1, 2, 1,   # beat 2
                 4, 1, 2, 1,   # beat 3
                 3, 1, 2, 1],  # beat 4
                dtype=int),
    # 3/4 time: 12 pulses (3 beats x 4 subdivisions)
    3: np.array([5, 1, 2, 1,   # beat 1 (downbeat)
                 3, 1, 2, 1,   # beat 2
                 4, 1, 2, 1],  # beat 3
                dtype=int),
    # 2/4 time: 8 pulses (2 beats x 4 subdivisions)
    2: np.array([5, 1, 2, 1,   # beat 1 (downbeat)
                 4, 1, 2, 1],  # beat 2
                dtype=int),
}

### DEBUG
def _score_pattern(pattern, num_bars, beats_per_bar=4, steps_per_beat=4):
    if pattern is None or num_bars is None:
        return float("nan")

    # Get meter-specific weights, default to 4/4
    w = METER_WEIGHTS.get(beats_per_bar, METER_WEIGHTS[4])

    bar_len = beats_per_bar * steps_per_beat  # 16
    pattern = np.asarray(pattern, dtype=int).ravel()

    need = num_bars * bar_len
    if len(pattern) < need:
        return float("nan")

    patt = pattern[:need].reshape(num_bars, bar_len)
    
    # Calculate the sum of weights for each bar
    # Higher sum = Higher Metrical Clarity (Simpler)
    # Lower sum = Higher Syncopation (More Complex)
    bar_sums = (patt @ w).astype(float)
    
    # Toussaint Normalization: To get a "Syncopation Score" where Higher = More Syncopated,
    # we subtract the weight from the maximum possible weight (5) for each onset.
    # Score = Sum(Max_Weight - Actual_Weight_at_Onset) / Number_of_Onsets
    
    syncopation_per_bar = []
    for bar in patt:
        onsets = np.where(bar > 0)[0]
        if len(onsets) == 0:
            syncopation_per_bar.append(0.0)
            continue
        
        # Calculate how "far" each onset is from the strongest possible pulse (weight 5)
        bar_sync_score = np.sum(5 - w[onsets]) / len(onsets)
        syncopation_per_bar.append(bar_sync_score)

    return float(np.mean(syncopation_per_bar))


def _fmt_score(score):
    # return "" for NaN / inf, else scale to 0-100 and round to integer
    if not isinstance(score, (float, int)):
        return ""
    score = float(score)
    if not np.isfinite(score):
        return ""
    # Scale from 0-4 to 0-100 and round
    scaled = round(score * 25)
    return str(scaled)


# =========================
# New Public API (Path 2 Logic)
# =========================
def onset_detection(y, sr, hop_length=512, delta=0.25, wait_frames=None):
    """
    Detect onsets in an audio signal using the logic from Path 2 (drums).
    
    Args:
        y: Audio time series (monophonic)
        sr: Sample rate
        hop_length: Hop length for onset detection
        delta: Threshold for onset detection (default 0.25 for drums)
        wait_frames: Wait frames for onset detection (optional)
        
    Returns:
        onset_times: Array of onset times in seconds
    """
    preroll_sec = 0.010
    pad = int(preroll_sec * sr)
    # Pad start to catch early onsets
    y_pad = np.concatenate([np.zeros(pad, dtype=y.dtype), y])

    # Note: Path 2 does NOT use hpss (percussive separation) here, it assumes input is already processed drums
    env = librosa.onset.onset_strength(y=y_pad, sr=sr, hop_length=hop_length)

    if wait_frames is None:
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


def syncopation_score(file_path):
    """
    Calculate the syncopation score for a given audio file using Path 2 logic:
    1. Detect beats on the original song.
    2. Separate 'drums' stem.
    3. Detect onsets on the drum stem.
    4. Quantize onsets to the song's beat grid.
    5. Calculate Toussaint syncopation score.
    
    Returns:
        float: Toussaint syncopation score (raw 0-4 scale roughly), or nan if failed.
    """
    # 1. Load song and detect beats
    song_y, song_sr = load_audio(file_path)
    song_y = normalize(song_y)
    hop_length = 512
    
    # Use detect_beats (captures first beat reliably). Force 4/4 by grouping every 4 beats.
    song_beat_times, _ = detect_beats(song_y, song_sr, hop_length=hop_length)
    if len(song_beat_times) < 2:
        return float("nan")

    # 2. Separate drums
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = None
        
    try:
        drums_y, _, drums_sr = separate(file_path, "drums", DEVICE)
    except Exception as e:
        print(f"Separation failed for {file_path}: {e}")
        return float("nan")

    drums_y = np.asarray(drums_y)
    if drums_y.ndim == 2:
        drums_y = np.mean(drums_y, axis=0)

    # Resample drums to match song sample rate if needed
    if drums_sr != song_sr:
        drums_y = librosa.resample(drums_y, orig_sr=drums_sr, target_sr=song_sr)
        drums_sr = song_sr

    # 3. Process drums (low-pass)
    nyquist = 0.5 * drums_sr
    cutoff_hz = 1500
    b_low, a_low = signal.butter(4, cutoff_hz / nyquist, btype="low")
    drums_lp = normalize(signal.filtfilt(b_low, a_low, drums_y))
    
    # 4. Detect onsets
    onset_times = onset_detection(drums_lp, drums_sr, hop_length, delta=0.25)

    # 5. Quantize and Score
    # Always use 4/4 (beats_per_bar=4)
    pattern, num_bars = _pattern_from_beats_and_onsets(song_beat_times, onset_times, beats_per_bar=4)
    
    return _score_pattern(pattern, num_bars)


# =========================
# PATTERN extraction (Path 1 / Path 2)
# These return (pattern, num_bars, meter), not a score.
# =========================
def _extract_pattern_path1(file_path: str):
    y, sr = load_audio(file_path)
    y = normalize(y)

    hop_length = 512
    
    # Use detect_beats (captures first beat reliably). Force 4/4 by grouping every 4 beats.
    beat_times, _ = detect_beats(y, sr, hop_length=hop_length)
    if len(beat_times) < 2:
        return None, None

    preroll_sec = 0.010
    pad = int(preroll_sec * sr)
    y_pad = np.concatenate([np.zeros(pad, dtype=y.dtype), y])

    y_perc = librosa.effects.percussive(y_pad)
    env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)

    onset_frames = librosa.onset.onset_detect(
        y=y_perc,
        onset_envelope=env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        pre_max=int(0.015 * sr / hop_length),
        post_max=int(0.015 * sr / hop_length),
        pre_avg=int(0.080 * sr / hop_length),
        post_avg=int(0.080 * sr / hop_length),
        delta=0.10,
        wait=int(0.020 * sr / hop_length),
        backtrack=False
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length) - preroll_sec
    onset_times = onset_times[onset_times >= 0]

    # Always use 4/4 (beats_per_bar=4), no beat_positions needed
    return _pattern_from_beats_and_onsets(beat_times, onset_times, beats_per_bar=4)


def _extract_pattern_path2(file_path: str):
    song_y, song_sr = load_audio(file_path)
    song_y = normalize(song_y)

    hop_length = 512
    
    # Use detect_beats (captures first beat reliably). Force 4/4 by grouping every 4 beats.
    song_beat_times, _ = detect_beats(song_y, song_sr, hop_length=hop_length)
    if len(song_beat_times) < 2:
        return None, None

    # Use GPU only if CUDA is available
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = None  # separate() expects None for CPU
    drums_y, _, drums_sr = separate(file_path, "drums", DEVICE)

    drums_y = np.asarray(drums_y)
    if drums_y.ndim == 2:
        # Assume (channels, samples) format from separator
        drums_y = np.mean(drums_y, axis=0)

    # Resample drums to match song sample rate if needed
    if drums_sr != song_sr:
        drums_y = librosa.resample(drums_y, orig_sr=drums_sr, target_sr=song_sr)
        drums_sr = song_sr

    nyquist = 0.5 * drums_sr
    cutoff_hz = 1500
    b_low, a_low = signal.butter(4, cutoff_hz / nyquist, btype="low")
    drums_lp = normalize(signal.filtfilt(b_low, a_low, drums_y))

    # Use the new onset_detection function
    onset_times = onset_detection(drums_lp, drums_sr, hop_length, delta=0.25)

    # IMPORTANT: quantize drum onsets to the SONG beat grid, always 4/4
    return _pattern_from_beats_and_onsets(song_beat_times, onset_times, beats_per_bar=4)


# =========================
# Pattern cache
# =========================
PATTERN_CACHE = {}

def get_pattern(mode: str, file_path: str):
    """
    mode: "p1" -> drums-only style pattern (Path 1)
          "p2" -> song->separate->drums pattern (Path 2)
    Returns: (pattern, num_bars) - always 4/4 (16 steps per bar)
    """
    key = (mode, os.path.realpath(file_path))
    if key in PATTERN_CACHE:
        return PATTERN_CACHE[key]

    if mode == "p1":
        pat, nb = _extract_pattern_path1(file_path)
    elif mode == "p2":
        pat, nb = _extract_pattern_path2(file_path)
    else:
        raise ValueError("mode must be 'p1' or 'p2'")

    PATTERN_CACHE[key] = (pat, nb)
    return pat, nb


# =========================
# Scoring paths (all use 4/4 = 16 steps per bar)
# =========================
def path_1(file_path: str) -> float:
    pat, nb = get_pattern("p1", file_path)
    return _score_pattern(pat, nb)

def path_2(file_path: str) -> float:
    pat, nb = get_pattern("p2", file_path)
    return _score_pattern(pat, nb)

def path_or(file_path_x1: str, file_path_1y: str) -> float:
    """
    OR-combine patterns: X1 (use p2) and 1Y (use p1), then score once.
    """
    pat_a, nb_a = get_pattern("p2", file_path_x1)  # X1 -> Path2 style
    pat_b, nb_b = get_pattern("p1", file_path_1y)  # 1Y -> Path1 style

    if pat_a is None or pat_b is None or nb_a is None or nb_b is None:
        return float("nan")

    nb = int(min(nb_a, nb_b))
    L = nb * 16  # 4 beats * 4 subdivisions = 16 steps per bar
    if L <= 0:
        return float("nan")

    a = np.asarray(pat_a, dtype=int).ravel()
    b = np.asarray(pat_b, dtype=int).ravel()

    if len(a) < L or len(b) < L:
        return float("nan")

    combined = np.logical_or(a[:L] > 0, b[:L] > 0).astype(int)
    return _score_pattern(combined, nb)


# =========================
# Main driver
# =========================
def get_syncopation_scores(folder_path, out_csv_path):
    """
    Reads all .wav in folder_path, computes scores, and writes a pivot-style CSV.
    
    Output CSV format:
        - Rows: file codes (12A, 12B, 13A, 13B, ..., 35A, 35B)
        - Columns: subjects (S1, S2, ..., S36)
        - Cells: syncopation scores (0-100)
    
    Optimizations:
        - X=1 files (1Y patterns: 12, 13, 14, 15) are identical across all subjects
          and A/B versions, so we compute once using S1-A and reuse.
        - X!=1 files are computed per subject.
    """
    global PATTERN_CACHE
    PATTERN_CACHE = {}  # reset cache for a clean run

    all_files = sorted(os.listdir(folder_path))
    wav_files = [f for f in all_files if f.lower().endswith(".wav")]
    index = {f: os.path.join(folder_path, f) for f in wav_files}

    # Build column headers: all valid XY+version combinations (excluding 11)
    file_codes = []
    for x in range(1, 4):  # 1, 2, 3
        for y in range(1, 6):  # 1, 2, 3, 4, 5
            if x == 1 and y == 1:
                continue  # skip 11
            for version in ["A", "B"]:
                file_codes.append(f"{x}{y}{version}")
    
    # Dictionary to store scores: key=(subject, file_code), value=score
    scores_dict = {}
    
    # ==========================================
    # Step 1: Compute X=1 scores ONCE using S1-A
    # These are shared across all subjects and A/B versions
    # ==========================================
    shared_1y_scores = {}  # key: y (2,3,4,5), value: score
    
    for y in range(2, 6):  # 12, 13, 14, 15
        file_name = f"S1-1{y}A.wav"
        if file_name in index:
            score = path_1(index[file_name])
            shared_1y_scores[y] = score
    
    # Apply shared scores to all subjects and versions for X=1 files
    for s in range(1, 37):
        subject = f"S{s}"
        for y in range(2, 6):
            if y in shared_1y_scores:
                score = shared_1y_scores[y]
                for version in ["A", "B"]:
                    file_code = f"1{y}{version}"
                    scores_dict[(subject, file_code)] = _fmt_score(score)
    
    # ==========================================
    # Step 2: Compute X!=1 scores per subject
    # ==========================================
    for s in range(1, 37):
        subject = f"S{s}"
        subject_files = [f for f in wav_files if f.upper().startswith(subject.upper() + "-")]
        
        for file_name in subject_files:
            parsed = parse_stimulus(file_name)
            if parsed is None:
                continue

            subject_parsed, x, y, ab = parsed

            # Skip 11 and X=1 files (already handled above)
            if x == 1:
                continue

            file_path = index[file_name]
            file_code = f"{x}{y}{ab}"
            
            # Check if already computed (shouldn't happen, but safety)
            if (subject_parsed, file_code) in scores_dict:
                continue

            if y == 1:
                # Y=1 files (X1): compute with path_2
                score = path_2(file_path)
            else:
                # XY files: OR combination
                f_x1 = make_file_name(subject_parsed, x, 1, ab)  # X1A or X1B
                f_1y = make_file_name(subject_parsed, 1, y, ab)  # 1YA or 1YB

                if f_x1 not in index or f_1y not in index:
                    score = float("nan")
                else:
                    score = path_or(index[f_x1], index[f_1y])

            scores_dict[(subject_parsed, file_code)] = _fmt_score(score)

    # ==========================================
    # Step 3: Build and write CSV
    # ==========================================
    subjects = [f"S{s}" for s in range(1, 37)]
    rows = []
    for fc in file_codes:
        row = {"file": fc}
        for subject in subjects:
            row[subject] = scores_dict.get((subject, fc), "")
        rows.append(row)

    fieldnames = ["file"] + subjects
    os.makedirs(os.path.dirname(out_csv_path) if os.path.dirname(out_csv_path) else ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Calculation complete @ {out_csv_path}")


#### RUN with the low songs
#### RUN with song
#### CREATE A Click - 4 (beats) 0 score, and silence 
