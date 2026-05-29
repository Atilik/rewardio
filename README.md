# rewardio: Music Analysis Toolbox

A simple, interactive command-line Python tool for music information retrieval (MIR). `rewardio` provides an intuitive interactive shell to load, analyze, and visualize audio files—either individually, by session (folder of songs), or by patient (folder of sessions).

## Features

- **Rhythm & Syncopation**: Beat and downbeat tracking using BEAT THIS! (or madmom), tempo estimation (BPM), and Toussaint syncopation scoring.
- **Stem Separation**: Drum separation using Demucs.
- **Timbre & Dynamics**: LUFS loudness, RMS energy, spectral irregularity, and fluctuation strength.
- **Harmony & Pitch**: Key and scale extraction (Essentia), and frame-level pitch tracking (CREPE).
- **Genre & Mood**: Pre-trained deep learning classifiers for genre, danceability, mood, and voice/instrumental detection (Essentia).
- **Interactive Player**: A built-in GUI player (`play()`) to view waveforms, spectrograms (linear/log/mel), and pitch contours, while sonifying detected beats and onsets over the audio.
- **Batch Processing**: Automatic process and save everything as CSV (`process_and_save()`).

---

## 🚀 Getting Started

Open the terminal from your macos.

Launch the tool by passing the path to an audio file, a session folder, or a patient folder.

```bash
cd scripts/
# Load a single song:
python rewardio.py /path/to/song.wav

# Load a single session (folder of audio files):
python rewardio.py /path/to/session_folder/

# Load a patient (folder containing session folders):
python rewardio.py /path/to/patient_folder/
```

This drops you into an interactive Python shell pre-loaded with your data.

---

## 📖 The Hierarchy

`rewardio` structures your data into three levels depending on the folder you pass:

1. **Patient**: A folder containing multiple *Session* folders.
2. **Stimuli** (Session): A folder containing multiple *Stimulus* audio files.
3. **Stimulus**: A single audio file (e.g., a `.wav` or `.mp3`).

When you load a folder, `rewardio` automatically gives you variables (`patient`, `stimuli`, `stimulus`) to interact with your data immediately.

---

## 💻 Using the Interactive Shell

Type the following commands directly into the terminal once `rewardio` is launched:

### Navigating Data
- `patient(1)` — Focus on the 1st session. Updates the `stimuli` variable.
- `patient("baseline")` — Focus on a session containing "baseline" in its folder name.
- `stimuli(3)` — Focus on the 3rd song in the current session. Updates the `stimulus` variable.
- `stimuli("beatles")` — Focus on a song containing "beatles" in its filename.

### Viewing Info
- `stimulus.help()` — List all available attributes and methods for the current song.
- `stimuli.help()` — List all available methods for the session.
- `patient.help()` — List all available methods for the patient.
- `stimulus.print()` — Print a summary of the current song (loudness, BPM, syncopation, key, etc.).
- `stimuli.print()` — Print summary metrics averaged across the whole session.
- `patient.print()` — Print the current attributes and sessions loaded.

### Interactive Player & Viz
- **`stimulus.play()`**
  Launch the interactive unified player. You can switch between Waveform, Mel, Log, Linear, and Pitch views. Click **Beats** or **Onsets** to visualize and sonify rhythm markers directly over the audio playback.
- `stimulus.plot()` — Quick static waveform/spectrogram.
- `stimuli.plot_radar(metrics=['bpm', 'lufs', 'syncopation_score'])` — Radar chart of metrics across the session.
- `stimuli.plot_boxplots()` — Boxplots showing metric distributions.

### Getting Metrics
Access properties on-the-fly. If a metric hasn't been computed yet, `rewardio` computes it instantly.
```python
>>> stimulus.bpm
120.5
>>> stimulus.key
'C#'
>>> stimulus.scale
'minor'
>>> stimulus.toussaint_syncopation_score
0.42
```

### Exporting Data
- `stimulus.process_and_save("output_folder")`
- Computes ALL available metrics (beats, syncopation, loudness, genre, mood, key, etc.) for the specific song and saves them to a CSV file.
- `stimuli.process_and_save("output_folder")`
  Computes ALL available metrics (beats, syncopation, loudness, genre, mood, key, etc.) for every song in the session and saves them to a CSV file.
- `patient.process_and_save("output_folder")`
  Does the same, but loops through every session folder, adding a `session` column to the final CSV.

---

## 🛠 Advanced Features

### Separation & Syncopation
Syncopation requires isolating the drums. `rewardio` will prompt you to run Demucs separation the first time you ask for a syncopation score.
```python
>>> stimulus.syncopation_score()
Checking separation...
⚠️  Do you want to run Demucs drum separation on this track? (y/n)
```

### Pitch Tracking
The pitch view mode in `stimulus.play()` runs CREPE pitch detection.
```python
>>> stimulus.pitch_time
>>> stimulus.pitch_freq
```
