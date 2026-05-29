"""
Debug script: traces every step of the syncopation pipeline for a given file.
Run:  python debug_pipeline.py <path_to_wav>

Shows:
  1. Madmom beat detection (times + positions)
  2. Onset detection (times)
  3. Bar boundaries (downbeat times)
  4. Pattern grid per bar (which slots are filled)
  5. Scoring per bar (weights, syncopation)
  6. Final score
"""
import sys, os
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_audio
from dsp import normalize
from rhythm import detect_beats

METER_WEIGHTS = np.array([5,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1], dtype=int)
SLOT_LABELS = [
    "Beat1",    "e",    "&",    "a",
    "Beat2",    "e",    "&",    "a",
    "Beat3",    "e",    "&",    "a",
    "Beat4",    "e",    "&",    "a",
]

def debug_file(file_path):
    print("=" * 70)
    print(f"FILE: {os.path.basename(file_path)}")
    print("=" * 70)

    # Step 1: Load audio
    y, sr = load_audio(file_path)
    y = normalize(y)
    hop_length = 512
    print(f"\n[1] AUDIO: {len(y)} samples, sr={sr}, duration={len(y)/sr:.2f}s")

    # Step 2: Beat detection (using detect_beats - captures first beat)
    beat_times, _ = detect_beats(y, sr, hop_length=hop_length)
    print(f"\n[2] MADMOM BEATS: {len(beat_times)} beats (4/4 forced)")
    print(f"    First 20 beats:")
    for i in range(min(20, len(beat_times))):
        bar_num = i // 4 + 1
        beat_in_bar = i % 4 + 1
        marker = " <<<< DOWNBEAT" if beat_in_bar == 1 else ""
        print(f"      beat {i:3d}: t={beat_times[i]:7.3f}s  bar={bar_num} beat={beat_in_bar}{marker}")
    if len(beat_times) > 20:
        print(f"      ... ({len(beat_times) - 20} more)")

    # Step 3: Find downbeats (every 4th beat for 4/4)
    beats_per_bar = 4
    downbeat_indices = np.arange(0, len(beat_times), beats_per_bar)
    print(f"\n[3] DOWNBEATS (every 4 beats): {len(downbeat_indices)} found")
    for i, di in enumerate(downbeat_indices):
        if i < 15:
            print(f"      Bar {i+1} starts at beat index {di}, time={beat_times[di]:.3f}s")
    num_bars = max(1, (len(beat_times) - 1) // beats_per_bar)
    print(f"    Complete bars: {num_bars}")

    # Step 4: Onset detection
    preroll_sec = 0.010
    pad = int(preroll_sec * sr)
    y_pad = np.concatenate([np.zeros(pad, dtype=y.dtype), y])
    y_perc = librosa.effects.percussive(y_pad)
    env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)

    onset_frames = librosa.onset.onset_detect(
        y=y_perc, onset_envelope=env, sr=sr, hop_length=hop_length,
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
    print(f"\n[4] ONSETS: {len(onset_times)} detected")
    for i, ot in enumerate(onset_times):
        if i < 30:
            print(f"      onset {i:3d}: t={ot:.3f}s")
    if len(onset_times) > 30:
        print(f"      ... ({len(onset_times) - 30} more)")

    # Step 5: Quantize onsets to grid
    print(f"\n[5] QUANTIZING ONSETS TO GRID")
    bt = np.asarray(beat_times, dtype=float)
    grid_length = num_bars * 16
    grid = np.zeros(grid_length, dtype=int)

    for onset_time in onset_times:
        bar_idx = np.searchsorted(bt[downbeat_indices], onset_time, side="right") - 1
        if bar_idx < 0 or bar_idx >= num_bars:
            print(f"    onset {onset_time:.3f}s -> OUTSIDE grid (bar_idx={bar_idx})")
            continue

        bar_start = bt[downbeat_indices[bar_idx]]
        bar_end = bt[downbeat_indices[bar_idx + 1]]
        bar_dur = bar_end - bar_start
        if bar_dur <= 0:
            continue

        frac = (onset_time - bar_start) / bar_dur
        slot_in_bar = int(np.round(frac * 16))

        if slot_in_bar >= 16:
            bar_idx += 1
            slot_in_bar = 0
            if bar_idx >= num_bars:
                continue

        slot = bar_idx * 16 + slot_in_bar
        if 0 <= slot < grid_length:
            grid[slot] = 1
            print(f"    onset {onset_time:.3f}s -> Bar {bar_idx+1}, slot {slot_in_bar:2d} ({SLOT_LABELS[slot_in_bar]}), weight={METER_WEIGHTS[slot_in_bar]}")

    # Step 6: Show pattern and score per bar
    print(f"\n[6] PATTERN & SCORING PER BAR")
    print(f"    Weights: {list(METER_WEIGHTS)}")
    print(f"    Labels:  {SLOT_LABELS}")
    print()

    patt = grid[:num_bars * 16].reshape(num_bars, 16)
    syncopation_per_bar = []

    for bar_idx, bar in enumerate(patt):
        onsets = np.where(bar > 0)[0]
        print(f"    Bar {bar_idx+1}: {list(bar)}")

        if len(onsets) == 0:
            print(f"           No onsets -> score = 0.0")
            syncopation_per_bar.append(0.0)
            continue

        onset_labels = [f"{SLOT_LABELS[s]}(w={METER_WEIGHTS[s]})" for s in onsets]
        sync_values = [5 - METER_WEIGHTS[s] for s in onsets]
        total = sum(sync_values)
        count = len(onsets)
        bar_score = total / count

        print(f"           Onsets at: {onset_labels}")
        print(f"           Sync values (5-w): {sync_values}")
        print(f"           Score: sum={total} / count={count} = {bar_score:.2f}")
        syncopation_per_bar.append(bar_score)

    # Step 7: Final score
    final_raw = float(np.mean(syncopation_per_bar))
    final_scaled = round(final_raw * 25)
    print(f"\n[7] FINAL SCORE")
    print(f"    Per-bar scores: {[f'{s:.2f}' for s in syncopation_per_bar]}")
    print(f"    Mean (raw):     {final_raw:.4f}")
    print(f"    Scaled (0-100): {final_scaled}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pipeline.py <path_to_wav>")
        print("Example: python debug_pipeline.py /path/to/S1-15A.wav")
        sys.exit(1)
    
    debug_file(sys.argv[1])
