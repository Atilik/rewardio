import librosa
import numpy as np
import pyloudnorm as pyln
from scipy import signal


def get_loudness(y, sr):
    """
    Compute integrated loudness (LUFS) and RMS (dBFS).

    Parameters
    ----------
    y : np.ndarray
        Audio signal.  Mono (n_samples,) or stereo (n_channels, n_samples).
    sr : int
        Sample rate.

    Returns
    -------
    lufs : float
        Integrated loudness in LUFS (ITU-R BS.1770-4).
    rms_db : float
        RMS level in dBFS.
    """
    # pyloudnorm expects shape (n_samples,) for mono or (n_samples, n_channels)
    if y.ndim == 2:
        y_ln = y.T  # (channels, samples) -> (samples, channels)
    else:
        y_ln = y

    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(y_ln)

    # RMS in dBFS
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    rms_db = 20.0 * np.log10(rms + 1e-12)  # guard against log(0)

    # Full precision — display code formats for readability; CSV stores raw
    return float(lufs), float(rms_db)


def get_rms(y, sr):
    """
    Compute mean RMS energy using librosa.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.  Mono (n_samples,) or stereo (n_channels, n_samples).
    sr : int
        Sample rate (unused, kept for API consistency).

    Returns
    -------
    rms_mean : float
        Mean RMS energy (linear).
    """
    # librosa.feature.rms expects mono
    if y.ndim == 2:
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y

    rms = librosa.feature.rms(y=y_mono)[0]  # shape (1, n_frames) -> (n_frames,)
    rms_mean = float(np.mean(rms))

    return rms_mean


def normalize(y):
    # https://librosa.org/doc/main/generated/librosa.util.normalize.html
    y_normalized = librosa.util.normalize(y)
    
    return y_normalized

def filter(y, filter_type, cutoff_freq, sr=44100, hop_length=512):
    if filter_type == "low_pass":
        return filter_low_pass(y, cutoff_freq, sr, hop_length)
    elif filter_type == "high_pass":
        return filter_high_pass(y, cutoff_freq, sr, hop_length)
    elif filter_type == "bell":
        return filter_bell(y, cutoff_freq, sr, hop_length)
    else:
        return "Usage: filter(y, filter_type, cutoff_freq, sr, hop_length"

def filter_low_pass(y, cutoff_freq, sr=44100, hop_length=512):
    # Nyquist
    nyquist = 0.5 * sr
    
    # Butterworth
    low_cutoff_norm = cutoff_freq / nyquist
    b_low, a_low = signal.butter(4, low_cutoff_norm, btype="low")
    
    # APPLY FILTERS (zero-phase!)
    # filtfilt = no phase shift → onsets stay in the right place
    y_low_pass = signal.filtfilt(b_low, a_low, y)
    
    # Normalize
    y_low_pass = librosa.util.normalize(y_low_pass)
    
    return y_low_pass

def filter_high_pass(y, cutoff_freq, sr=44100, hop_length=512):
    nyquist = 0.5 * sr
    high_cutoff_norm = cutoff_freq / nyquist
    b_high, a_high = signal.butter(4, high_cutoff_norm, btype="high")
    
    y_high_pass = signal.filtfilt(b_high, a_high, y)
    y_high_pass = librosa.util.normalize(y_high_pass)
    
    return y_high_pass

def filter_bell(y, cutoff_freq, sr=44100, hop_length=512):
    raise NotImplementedError("filter_bell is not yet implemented.")


# ──────────────────────────────────────────────
#  Fluctuation strength  (Pampalk et al., 2002)
# ──────────────────────────────────────────────

def compute_fluctuation(y, sr, n_bands=40, frame_sec=0.023, max_mod_freq=10.0):
    """
    Compute rhythmic periodicity (fluctuation strength) along auditory bands.

    Based on MIRtoolbox's mirfluctuation (Pampalk et al., 2002).

    Pipeline:
      1. Compute power mel spectrogram  (n_bands mel bands, 23ms frames)
      2. Convert to dB
      3. For each band, compute FFT → modulation spectrum
      4. Weight by Fastl fluctuation strength model:  w(f) = 1/(f/4 + 4/f)
      5. Sum across bands → fluctuation summary

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    n_bands : int
        Number of mel bands (default 40, like MIRtoolbox 'Mel' mode).
    frame_sec : float
        Frame length in seconds (default 0.023 = 23ms).
    max_mod_freq : float
        Maximum modulation frequency in Hz (default 10).

    Returns
    -------
    fluctuation_summary : float
        Total fluctuation strength (scalar summary).
    mod_freqs : np.ndarray
        Modulation frequency axis in Hz.
    fluctuation_per_band : np.ndarray (n_bands, n_mod_bins)
        Fluctuation spectrum per auditory band.
    """
    if y.ndim == 2:
        y = librosa.to_mono(y)

    # 1. Power mel spectrogram
    hop_length = int(frame_sec * sr)
    n_fft = int(frame_sec * sr * 2)  # ~46ms window, 23ms hop
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bands,
        hop_length=hop_length, n_fft=n_fft, power=2.0
    )

    # 2. Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 3. For each band: FFT to get modulation spectrum
    n_frames = S_dB.shape[1]
    mod_spec = np.abs(np.fft.rfft(S_dB, axis=1))

    # Modulation frequency axis
    frame_rate = sr / hop_length  # frames per second
    mod_freqs = np.fft.rfftfreq(n_frames, d=1.0 / frame_rate)

    # Trim to max_mod_freq
    valid = mod_freqs <= max_mod_freq
    mod_freqs = mod_freqs[valid]
    mod_spec = mod_spec[:, valid]

    # 4. Fastl fluctuation strength weighting: w(f) = 1/(f/4 + 4/f)
    #    Peak at 4 Hz (typical perceptual tempo)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / (mod_freqs / 4.0 + 4.0 / mod_freqs)
    weights[0] = 0.0  # DC has no rhythmic meaning
    weights = np.nan_to_num(weights)

    # Apply weighting
    fluctuation_per_band = mod_spec * weights[np.newaxis, :]

    # 5. Summary: sum across bands, take mean across modulation bins
    fluctuation_summary = float(np.mean(np.sum(fluctuation_per_band, axis=0)))

    return fluctuation_summary, mod_freqs, fluctuation_per_band


# ──────────────────────────────────────────────
#  Spectral Irregularity  (Jensen, 1999)
# ──────────────────────────────────────────────

def spectral_irregularity(y, sr, n_fft=2048, hop_length=512):
    """
    Compute spectral irregularity (Jensen, 1999).

    Measures the variability of successive spectral peaks.

    Formula (Jensen):
        irregularity = sum((a_k - a_{k+1})^2) / sum(a_k^2)

    This is computed per-frame and averaged.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.

    Returns
    -------
    irregularity : float
        Mean spectral irregularity (0 = smooth spectrum, higher = more jagged).
    """
    if y.ndim == 2:
        y = librosa.to_mono(y)

    # Magnitude spectrum per frame
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Jensen irregularity per frame
    irreg_per_frame = []
    for frame_idx in range(S.shape[1]):
        a = S[:, frame_idx]
        denom = np.sum(a ** 2)
        if denom < 1e-12:
            irreg_per_frame.append(0.0)
            continue
        numer = np.sum(np.diff(a) ** 2)
        irreg_per_frame.append(numer / denom)

    return float(np.mean(irreg_per_frame))


# ──────────────────────────────────────────────
#  Spectral feature bundle
# ──────────────────────────────────────────────

def compute_spectral_features(y, sr, n_fft=2048, hop_length=512):
    """
    Compute standard spectral descriptors as per-frame arrays.

    Returns dict with keys:
        spectral_centroid, spectral_bandwidth, spectral_rolloff,
        spectral_flatness, zcr
    Each value is a 1-D numpy array (one value per frame).
    """
    if y.ndim == 2:
        y = librosa.to_mono(y)

    return {
        "spectral_centroid":   librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0],
        "spectral_bandwidth":  librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0],
        "spectral_rolloff":    librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0],
        "spectral_flatness":   librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0],
        "zcr":                 librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0],
    }
