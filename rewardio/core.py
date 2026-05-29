import librosa
import numpy as np
import os
import csv
from datetime import datetime


def load_audio(audio_file_path, sr=44100, mono=False):
    """
    Load an audio file, preserving stereo by default.

    Parameters
    ----------
    audio_file_path : str
        Path to the audio file.
    sr : int
        Target sample rate.
    mono : bool
        If True, mix down to mono. Default is False (preserve channels).

    Returns
    -------
    y : np.ndarray
        Audio signal. Shape (n_samples,) if mono, (n_channels, n_samples) if stereo.
    sr : int
        Sample rate.
    """
    y, sr = librosa.load(audio_file_path, sr=sr, mono=mono)
    return y, sr


def write_to_csv(rows, output_path=None):
    """
    Write a list of row-dicts to a timestamped CSV inside an
    Analysis_{DD-MM-YYYY} folder.

    Parameters
    ----------
    rows : list[dict]
        Each dict is one row. Keys become column headers.
    output_path : str or None
        Parent directory for the Analysis folder.
        Defaults to the directory this script lives in.

    Returns
    -------
    csv_path : str
        Absolute path to the written CSV file.
    """
    if output_path is None:
        output_path = os.path.dirname(os.path.abspath(__file__))

    now = datetime.now()
    folder_name = now.strftime("Analysis_%d-%m-%Y")
    folder_path = os.path.join(output_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    file_name = now.strftime("rewardio_%H-%M.csv")
    csv_path = os.path.join(folder_path, file_name)

    # Collect all unique keys across rows to handle sparse data
    all_keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved → {csv_path}")


# ──────────────────────────────────────────────
#  Stimulus helpers  (help / print)
# ──────────────────────────────────────────────

def stimulus_help(stimulus):
    """Print all available attributes and methods."""
    print("\n=== Stimulus Help ===")
    print("\nAttributes:")
    print("  .audio_file_path           - Full path to source file")
    print("  .audio_file_name           - Filename")
    print("  .y                         - Audio signal")
    print("  .sr                        - Sample rate")
    print("  .n_channels                - Number of channels")
    print("  .duration                  - Duration in seconds")
    print("  .loudness_lufs             - Integrated loudness [−70 to 0] LUFS")
    print("  .loudness_rms_db           - RMS level [−∞ to 0] dBFS")
    print("  .rms                       - Mean RMS energy [0–1]")
    print("  .BPM                       - Detected tempo [~30–300] BPM")
    print("  .beat_times                - numpy array of detected beats (seconds)")
    print("  .onset_times               - numpy array of detected onsets (seconds)")
    print("  .toussaint_sync_score      - Toussaint syncopation score [0–100]")
    print("  .separated_drums           - Only drums audio")
    print("  .genre                     - Top predicted genre (Discogs400)")
    print("  .genre_top5                - Top 5 genre predictions w/ confidence [0–1]")
    print("  .voice_instrumental        - 'voice' or 'instrumental'")
    print("  .mood                      - Mood dict [0–1] (happy/sad/aggressive/relaxed)")
    print("  .pitch                     - Median pitch (Hz, CREPE, voiced frames)")
    print("  .pitch_mean/std            - Mean/std pitch (Hz, voiced frames)")
    print("  .pitch_conf_mean/std       - CREPE pitch confidence [0–1]")
    print("  .beat_ioi_mean/std         - Inter-beat interval (seconds)")
    print("  .onset_ioi_mean/std        - Inter-onset interval (seconds)")
    print("  .fluctuation               - Rhythmic periodicity [0–∞] (Pampalk, 2002)")
    print("  .irregularity              - Spectral irregularity [0–1] (Jensen, 1999)")
    print("  .spectral_centroid_mean/std - Brightness (Hz)")
    print("  .spectral_bandwidth_mean/std- Frequency spread (Hz)")
    print("  .spectral_rolloff_mean/std  - Energy concentration (Hz)")
    print("  .spectral_flatness_mean/std - Tonal vs noisy [0=tone, 1=noise]")
    print("  .zcr_mean/std              - Zero-crossing rate [0–1]")

    print("\nMethods:")
    print("  .play(x, y)               : Interactive player (play/pause + toggle beats/onsets)")
    print("  .separate()                : Isolate drums using Demucs")
    print("  .detect_onsets()           : Onset detection")
    print("  .syncopation_score()       : Calculate Toussaint score")
    print("  .classify()                : Run ALL classifiers + pitch at once")
    print("  .save(path)                : Save current attributes to CSV")
    print("  .save_timeseries(path)     : Save all time series as .npz")
    print("  .process_and_save(path)    : Compute all metrics, then save CSV")
    print("  .help()                    : Show attributes and methods")
    print("  .print()                   : Concise summary of key attributes")
    print("  .print_all()               : List ALL attribute values")

    print("\nUtilities:")
    print("  clear()                    : Clear terminal screen")
    print("=" * 30 + "\n")


def stimulus_print(stimulus):
    """Print a concise summary of the most useful attributes."""
    print("\n=== Stimulus Summary ===")
    print(f"  File     : {stimulus.audio_file_name}")
    print(f"  Duration : {stimulus.duration:.2f}s")
    print(f"  SR       : {stimulus.sr} Hz  ({stimulus.n_channels}ch)")

    # Loudness (always available)
    print(f"  Int.LUFS : {stimulus.loudness_lufs}  [−70 to 0] (ITU-R BS.1770-4)")
    print(f"  RMS (dB) : {stimulus.loudness_rms_db} dBFS  [−∞ to 0]")
    print(f"  RMS      : {stimulus.rms}  [0–1]")

    # BPM (lazy — don't trigger)
    if getattr(stimulus, '_bpm', None) is not None:
        print(f"  BPM      : {stimulus.bpm}  [~30–300]")
    else:
        print("  BPM      : (...)")

    # Meter
    if getattr(stimulus, '_meter', None) is not None:
        print(f"  Meter    : {stimulus._meter}/4")

    # Syncopation
    if stimulus.toussaint_syncopation_score is not None:
        print(f"  Sync.    : {stimulus.toussaint_syncopation_score:.1f}  [0–100]")
    if stimulus.toussaint_syncopation_score_meter is not None:
        print(f"  Sync.(m) : {stimulus.toussaint_syncopation_score_meter:.1f}  [0–100]")

    # Onsets
    if stimulus.onset_times is not None:
        print(f"  Onsets   : {len(stimulus.onset_times)} detected")

    # Genre (lazy — don't trigger)
    if getattr(stimulus, '_genre_predictions', None) is not None:
        top = stimulus._genre_predictions[0]
        print(f"  Genre    : {top[0]}  ({top[1]*100:.1f}%)  [0–100%]")
    if getattr(stimulus, '_voice_instrumental', None) is not None:
        print(f"  Vocal    : {stimulus._voice_instrumental[0][0]}")
    if getattr(stimulus, '_mood', None) is not None:
        top_mood = max(stimulus._mood, key=stimulus._mood.get)
        print(f"  Mood     : {top_mood} ({stimulus._mood[top_mood]*100:.0f}%)  [0–100%]")
    if getattr(stimulus, '_pitch_freq', None) is not None:
        import numpy as _np
        voiced = stimulus._pitch_freq[stimulus._pitch_conf > 0.5]
        if len(voiced) > 0:
            print(f"  Pitch    : {float(_np.median(voiced)):.1f} Hz  [~50–2000]")

    # Spectral features (always available — computed at init)
    print(f"  Fluct.   : {stimulus.fluctuation:.2f}  [0–∞]")
    print(f"  Irreg.   : {stimulus.irregularity:.4f}  [0–1]")
    if getattr(stimulus, '_spectral_features', None) is not None:
        print(f"  Centroid : {stimulus.spectral_centroid_mean:.1f} Hz (±{stimulus.spectral_centroid_std:.1f})  [0–{stimulus.sr//2}]")
        print(f"  Bwidth   : {stimulus.spectral_bandwidth_mean:.1f} Hz (±{stimulus.spectral_bandwidth_std:.1f})  [0–{stimulus.sr//2}]")
        print(f"  Rolloff  : {stimulus.spectral_rolloff_mean:.1f} Hz (±{stimulus.spectral_rolloff_std:.1f})  [0–{stimulus.sr//2}]")
        print(f"  Flatness : {stimulus.spectral_flatness_mean:.4f} (±{stimulus.spectral_flatness_std:.4f})  [0–1]")
        print(f"  ZCR      : {stimulus.zcr_mean:.4f} (±{stimulus.zcr_std:.4f})  [0–1]")
    print("=" * 30 + "\n")


def stimulus_print_all(stimulus):
    """Print the current value of every attribute on this Stimulus."""
    print("\n=== Stimulus — All Attributes ===")
    print(f"  File: {stimulus.audio_file_name}\n")
    
    # Explicitly print properties (lazy loaded)
    # Check internal attributes to avoid triggering calculation
    if getattr(stimulus, '_bpm', None) is not None:
        print(f"  bpm                       : {stimulus.bpm}")
    else:
        print("  bpm                       : (Not calculated)")
    
    if getattr(stimulus, '_beat_times', None) is not None:
        val = stimulus.beat_times
        summary = f"ndarray  shape={val.shape}  dtype={val.dtype}"
        print(f"  beat_times                : {summary}")
    else:
        print("  beat_times                : (Not calculated)")

    # Loudness (always available — computed on init)
    print(f"  loudness_lufs             : {stimulus.loudness_lufs} LUFS")
    print(f"  loudness_rms_db           : {stimulus.loudness_rms_db} dBFS")
    print(f"  rms                       : {stimulus.rms}")
    
    # Print other attributes in __dict__, skipping private ones
    for key, value in stimulus.__dict__.items():
        if key.startswith("_"):
            continue
            
        if isinstance(value, np.ndarray):
            summary = f"ndarray  shape={value.shape}  dtype={value.dtype}"
            print(f"  {key:25s} : {summary}")
        elif isinstance(value, (list, tuple)) and len(value) > 10:
            print(f"  {key:25s} : {type(value).__name__} (len={len(value)})")
        else:
            print(f"  {key:25s} : {value}")
    print("=" * 30 + "\n")


def session_help(session):
    """Print available attributes and methods for Session class."""
    print("\n=== Session Help ===")
    print("\nAttributes:")
    print("  .folder_path               - Full path to source folder")
    print("  .items                     - List of Stimulus objects")

    print("\nMethods:")
    print("  session(i)                 : Select and focus on the i-th Stimulus")
    print("  session('name')            : Select by partial filename match")
    print("  session[i]                 : Get the i-th Stimulus (1-based)")
    print("  session['name']            : Get Stimulus by partial filename match")
    print("  .boxplot()                 : Plot boxplots for BPM, LUFS, syncopation")
    print("  .save(path)                : Save current attributes to CSV")
    print("  .save_timeseries(path)     : Save all time series as .npz")
    print("  .process_and_save(path)    : Compute all metrics, then save CSV")
    print("  .help()                    : Show this message")
    print("  .print()                   : Summary with item listing")

    print("\nUtilities:")
    print("  clear()                    : Clear terminal screen")
    print("=" * 30 + "\n")


def session_print(session):
    """Print summary of the Session."""
    print("\n=== Session Summary ===")
    print(f"  Path   : {session.folder_path}")
    print(f"  Songs  : {len(session)}\n")
    # Item listing
    print("\n  Songs:")
    if len(session) == 0:
        print("    (No songs loaded)")
    else:
        for i, s in enumerate(session.items):
            print(f"    [{i+1}] {s.audio_file_name}")
            if i >= 19:  # Limit output
                print(f"    ... and {len(session) - 20} more.")
                break
    print("=" * 30 + "\n")


# ──────────────────────────────────────────────
#  Patient helpers  (help / print)
# ──────────────────────────────────────────────

def patient_help(patient):
    """Print available attributes and methods for the Patient class."""
    print("\n=== Patient Help ===")
    print("\nAttributes:")
    print("  .folder_path               - Full path to the patient folder")
    print("  .sessions                  - List of Session objects")
    print("  .n_sessions                - Number of sessions (sub-folders)")

    print("\nMethods:")
    print("  patient(i)                 : Select and focus on the i-th session")
    print("  patient('name')            : Select by partial folder name match")
    print("  patient[i]                 : Get the i-th session (1-based)")
    print("  patient['name']            : Get session by partial folder name match")
    print("  .save(path)                : Save current attributes of all sessions to CSV")
    print("  .process_and_save(path)    : Compute all metrics for every session, then save")
    print("  .help()                    : Show this message")
    print("  .print()                   : Summary with session listing")

    print("\nUtilities:")
    print("  clear()                    : Clear terminal screen")
    print("=" * 30 + "\n")


def patient_print(patient):
    """Print summary of the Patient collection."""
    print("\n=== Patient Summary ===")
    print(f"  Path     : {patient.folder_path}")
    print(f"  Sessions : {patient.n_sessions}\n")

    # Session listing
    print("  Sessions:")
    if patient.n_sessions == 0:
        print("    (No sessions loaded)")
    else:
        for i, s in enumerate(patient.sessions):
            folder_name = os.path.basename(s.folder_path)
            print(f"    [{i+1}] {folder_name}  ({len(s)} songs)")
            if i >= 19:
                print(f"    ... and {patient.n_sessions - 20} more.")
                break
    print("=" * 30 + "\n")


def clear():
    """Clear the terminal screen completely (including scrollback)."""
    # Try system command first
    ret = os.system('cls' if os.name == 'nt' else 'clear')
    
    # If using ANSI terminal (Mac/Linux), also send escape codes to 
    # clear scrollback (3J), clear screen (2J), and move cursor (H).
    if os.name != 'nt':
        print("\033[H\033[2J\033[3J", end="")