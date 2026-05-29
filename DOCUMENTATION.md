# Rewardio — Feature Documentation

**Rewardio** is a Music Information Retrieval (MIR) toolbox for analyzing audio session. It extracts musical features from audio files and exports them as CSV for downstream analysis.

---

## Quick Start

```python
from rewardio import Stimulus, Session

# Single file
s = Stimulus("song.wav")
s.process_and_save()              # compute all → CSV
s.process_and_save(timeseries=True)  # also save .npz time series

# Folder of files
session = Session("path/to/folder/")
session.process_and_save()
```

---

## Feature Reference

### Audio Properties

| Attribute | Type | Range | Description |
|-----------|------|-------|-------------|
| `.duration` | float | seconds | Duration of the audio file |
| `.sr` | int | Hz | Sample rate |
| `.n_channels` | int | 1 or 2 | Number of audio channels (mono/stereo) |

---

### Loudness

| Attribute | Range | Unit | Description |
|-----------|-------|------|-------------|
| `.loudness_lufs` | −70 to 0 | LUFS | **Integrated loudness** per ITU-R BS.1770-4. Measures perceived loudness over the entire file. −14 LUFS is a common streaming target. More negative = quieter. |
| `.loudness_rms_db` | −∞ to 0 | dBFS | **RMS level in decibels** relative to full scale. 0 dBFS = maximum digital level. Typical music sits around −10 to −20 dBFS. |
| `.rms` | 0 to 1 | linear | **Mean RMS energy** (librosa). Linear amplitude measure. Higher = louder. |

> **What do the numbers mean?**
> - LUFS of **−14** ≈ Spotify loudness normalization target
> - LUFS of **−23** ≈ broadcast standard (EBU R128)
> - LUFS of **−6** ≈ very loud, heavily compressed master

---

### Rhythm & Tempo

| Attribute | Range | Unit | Description |
|-----------|-------|------|-------------|
| `.bpm` | ~30–300 | BPM | **Beats per minute** detected via librosa's beat tracker. |
| `.beat_times` | — | seconds | Numpy array of detected beat timestamps. |
| `.onset_times` | — | seconds | Numpy array of detected onset (note attack) timestamps. Requires `.detect_onsets()`. |
| `.beat_ioi_mean` | seconds | s | **Mean inter-beat interval.** The average time between consecutive beats. Related to BPM: `IOI ≈ 60/BPM`. |
| `.beat_ioi_std` | seconds | s | **Std deviation of inter-beat intervals.** Measures **tempo stability** — low = steady tempo, high = tempo fluctuations or rubato. |
| `.onset_ioi_mean` | seconds | s | **Mean inter-onset interval.** Average time between note attacks. |
| `.onset_ioi_std` | seconds | s | **Std deviation of inter-onset intervals.** Measures **rhythmic regularity** — low = metronomic, high = varied rhythmic patterns. |

> **What do the numbers mean?**
> - `beat_ioi_mean` of **0.5s** = 120 BPM
> - `beat_ioi_std` of **0.01s** = very stable tempo
> - `beat_ioi_std` of **0.1s** = significant tempo variation

---

### Syncopation

| Attribute | Range | Unit | Description |
|-----------|-------|------|-------------|
| `.toussaint_syncopation_score` | 0–100 | score | **Toussaint syncopation score** using standard beat grid (4/4 assumed). Measures how much rhythmic emphasis falls off the beat. 0 = no syncopation (all onsets on beats), 100 = maximum syncopation. |
| `.toussaint_syncopation_score_meter` | 0–100 | score | **Meter-aware syncopation score.** Same algorithm but uses detected downbeats and meter for accurate bar alignment. |
| `.meter` | 2, 3, 4… | beats/bar | Detected time signature numerator (e.g., 4 for 4/4). |

> **What do the numbers mean?**
> - Score of **0–20**: Minimal syncopation (e.g., straight rock beat)
> - Score of **20–50**: Moderate syncopation (e.g., pop, R&B)
> - Score of **50+**: Heavy syncopation (e.g., jazz, funk, Afro-Cuban)

---

### Pitch (CREPE)

Pitch detection uses **CREPE** (Kim et al., 2018), a deep learning model for monophonic pitch estimation.

| Attribute | Range | Unit | Description |
|-----------|-------|------|-------------|
| `.pitch` | ~50–2000 | Hz | **Median pitch** across voiced frames (confidence > 0.5). |
| `.pitch_mean` | ~50–2000 | Hz | **Mean pitch** across voiced frames. |
| `.pitch_std` | Hz | Hz | **Std deviation of pitch.** Measures pitch range — low = monotone, high = wide melodic range. |
| `.pitch_conf_mean` | 0–1 | — | **Mean CREPE confidence.** How confident the model is that each frame is pitched. High = clear pitched content, low = noisy/unpitched. |
| `.pitch_conf_std` | 0–1 | — | **Std deviation of pitch confidence.** High = mixture of pitched and unpitched sections. |

> **What do the numbers mean?**
> - Pitch of **440 Hz** = A4 (concert pitch)
> - Pitch of **130 Hz** = C3 (male vocal range)
> - Pitch of **1000 Hz** = B5 (high soprano range)
> - `pitch_std` of **50 Hz** = narrow range; **200+ Hz** = wide melodic movement

---

### Key & Scale

| Attribute | Range | Description |
|-----------|-------|-------------|
| `.key` | C, C#, D, … B | Detected musical key (Essentia). |
| `.scale` | major / minor | Detected scale. |
| `.key_strength` | 0–1 | Confidence in the key detection. Higher = more tonal clarity. |

---

### Genre, Voice, & Mood (Essentia)

Classification uses **Essentia** neural network models with Discogs-EffNet embeddings.

| Attribute | Range | Description |
|-----------|-------|-------------|
| `.genre` | string | Top predicted genre from Discogs taxonomy (~400 genres). |
| `.genre_top5` | list | Top 5 genre predictions with confidence scores [0–1]. |
| `.voice_instrumental` | "voice" / "instrumental" | Whether the track contains vocals. |
| `.mood` | dict, values 0–1 | Mood probabilities: `happy`, `sad`, `aggressive`, `relaxed`. Values sum to ~1. |

> **What do the numbers mean?**
> - Genre confidence of **0.8** = high confidence (80%)
> - Genre confidence of **0.3** = ambiguous genre
> - Mood `happy: 0.7, relaxed: 0.2` = predominantly happy

---

### Fluctuation (Rhythmic Periodicity)

*Reference: Pampalk, E., Rauber, A., & Merkl, D. (2002). Content-based organization and visualization of music archives.*

| Attribute | Range | Description |
|-----------|-------|-------------|
| `.fluctuation` | 0 to ∞ | **Rhythmic periodicity strength.** Measures how much the signal fluctuates at perceptually salient modulation rates (~4 Hz, the "groove" frequency). |

**How it works:**
1. Compute a mel spectrogram (40 bands, 23ms frames)
2. Take the FFT of each band → modulation spectrum
3. Weight by the Fastl psychoacoustic model: `w(f) = 1/(f/4 + 4/f)` (peak at 4 Hz)
4. Sum across bands

> **What do the numbers mean?**
> - **0–50**: Low rhythmic periodicity (ambient, spoken word)
> - **50–200**: Moderate (ballads, slow tracks)
> - **200–500**: Strong groove (pop, rock)
> - **500+**: Very strong rhythmic drive (EDM, dance)

---

### Spectral Irregularity

*Reference: Jensen, K. (1999). Timbre models of musical sounds.*

| Attribute | Range | Description |
|-----------|-------|-------------|
| `.irregularity` | 0 to ~1 | **Spectral irregularity** — measures how jagged/smooth the frequency spectrum is. |

**How it works:**

For each frame, compute: `irregularity = Σ(aₖ − aₖ₊₁)² / Σ(aₖ²)`

where `aₖ` is the amplitude of the k-th frequency bin.

> **What do the numbers mean?**
> - **~0.0**: Very smooth spectrum (pure tones, sine waves)
> - **~0.1–0.3**: Moderate irregularity (most music)
> - **~0.5+**: Very jagged spectrum (noise, distortion, complex timbres)

---

### Spectral Features (librosa)

These features describe the **shape and distribution of spectral energy** across frequencies. All are computed per-frame and summarized as mean ± std.

| Attribute | Range | Unit | Description |
|-----------|-------|------|-------------|
| `.spectral_centroid_mean/std` | 0 to sr/2 | Hz | **"Brightness"** — the weighted mean of frequencies. Higher = brighter/tinnier sound, lower = darker/warmer. |
| `.spectral_bandwidth_mean/std` | 0 to sr/2 | Hz | **Frequency spread** around the centroid. Narrow = pure/tonal, wide = noisy/rich timbre. |
| `.spectral_rolloff_mean/std` | 0 to sr/2 | Hz | **Energy concentration** — frequency below which 85% of spectral energy sits. High = treble-heavy, low = bass-heavy. |
| `.spectral_flatness_mean/std` | 0 to 1 | — | **Tonality measure.** 0 = pure tone (all energy in one frequency), 1 = white noise (energy spread equally). |
| `.zcr_mean/std` | 0 to 1 | — | **Zero-crossing rate** — how often the signal crosses zero amplitude per frame. High = noisy/percussive, low = smooth/tonal. |

> **What do the numbers mean?**
> - Centroid of **1000 Hz** = warm/dark sound; **4000+ Hz** = bright/harsh
> - Flatness of **0.01** = very tonal (e.g., flute); **0.3+** = noisy (e.g., hi-hat)
> - ZCR of **0.02** = low-frequency tonal; **0.2+** = percussive/noisy

---

## Time Series Export

All time-series data can be saved as a NumPy `.npz` archive:

```python
s.save_timeseries()
# or
s.process_and_save(timeseries=True)
```

**Loading saved time series:**
```python
import numpy as np
data = np.load("song_timeseries.npz")
print(data.files)  # list all arrays
pitch = data["pitch_freq"]
centroid = data["spectral_centroid"]
```

**Arrays saved:**
| Key | Description |
|-----|-------------|
| `pitch_time` | CREPE time axis (seconds) |
| `pitch_freq` | CREPE pitch estimates (Hz) |
| `pitch_conf` | CREPE confidence [0–1] |
| `beat_times` | Beat timestamps (seconds) |
| `onset_times` | Onset timestamps (seconds) |
| `spectral_centroid` | Per-frame spectral centroid (Hz) |
| `spectral_bandwidth` | Per-frame spectral bandwidth (Hz) |
| `spectral_rolloff` | Per-frame spectral rolloff (Hz) |
| `spectral_flatness` | Per-frame spectral flatness [0–1] |
| `zcr` | Per-frame zero-crossing rate [0–1] |

---

## CSV Output

Running `.process_and_save()` generates a CSV with one row per audio file. All numeric values are stored at **full precision** (no rounding).

**Columns:**
`filename`, `duration`, `sr`, `n_channels`, `loudness_lufs`, `loudness_rms_db`, `rms`, `bpm`, `n_beats`, `n_onsets`, `syncopation_score`, `syncopation_score_meter`, `meter`, `genre`, `genre_confidence`, `voice_instrumental`, `mood_happy`, `mood_sad`, `mood_aggressive`, `mood_relaxed`, `pitch_median_hz`, `pitch_mean_hz`, `pitch_std_hz`, `pitch_conf_mean`, `pitch_conf_std`, `beat_ioi_mean`, `beat_ioi_std`, `onset_ioi_mean`, `onset_ioi_std`, `key`, `scale`, `key_strength`, `fluctuation`, `spectral_irregularity`, `spectral_centroid_mean`, `spectral_centroid_std`, `spectral_bandwidth_mean`, `spectral_bandwidth_std`, `spectral_rolloff_mean`, `spectral_rolloff_std`, `spectral_flatness_mean`, `spectral_flatness_std`, `zcr_mean`, `zcr_std`

---

## Dependencies

| Library | Purpose |
|---------|---------|
| **librosa** | Audio loading, beat/onset detection, spectral features |
| **CREPE** | Deep learning pitch detection (Kim et al., 2018) |
| **Essentia** | Genre/mood/voice classification, key detection |
| **Demucs** | Source separation (drum isolation for syncopation) |
| **pyloudnorm** | Integrated loudness (ITU-R BS.1770-4) |
| **scipy** | Signal processing (filters) |
| **numpy** | Numerical computation |

---

## References
### Going to be updated
- Jensen, Timbre Models of Musical Sounds, Rapport 99/7, University of Copenhagen, 1999.
- Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). CREPE: A convolutional representation for pitch estimation. ICASSP.
- Pampalk, E., Rauber, A., Merkl, D. "Content-based Organization and Visualization of Music Archives", ACM Multimedia 2002, pp. 570-579.
- ITU-R BS.1770-4 (2015). Algorithms to measure audio programme loudness and true-peak audio level.
