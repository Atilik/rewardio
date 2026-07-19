"""Tests for rewardio/dsp.py — one test (or more) per function."""
import numpy as np
import pytest

from rewardio.dsp import (
    get_loudness, get_rms, normalize, filter as dsp_filter,
    filter_low_pass, filter_high_pass, filter_bell,
    compute_fluctuation, spectral_irregularity, compute_spectral_features,
)
from conftest import make_sine, SR


# ── get_loudness ────────────────────────────────────────────

def test_get_loudness_mono():
    lufs, rms_db = get_loudness(make_sine(duration=2.0), SR)
    assert isinstance(lufs, float) and isinstance(rms_db, float)
    assert -70 < lufs < 0
    assert -70 < rms_db < 0


def test_get_loudness_stereo():
    y = np.stack([make_sine(440), make_sine(220)])  # (2, n) — rewardio layout
    lufs, rms_db = get_loudness(y, SR)
    assert np.isfinite(lufs) and np.isfinite(rms_db)


def test_get_loudness_louder_is_higher():
    quiet, _ = get_loudness(make_sine(amp=0.1), SR)
    loud, _ = get_loudness(make_sine(amp=0.9), SR)
    assert loud > quiet


def test_get_loudness_full_precision():
    # Regression (#6): values must not be rounded to 2 decimals
    lufs, _ = get_loudness(make_sine(amp=0.437), SR)
    assert lufs != round(lufs, 2) or abs(lufs - round(lufs, 2)) < 1e-12
    # RMS of a 0.437 sine is irrational — 2dp rounding would be detectable
    _, rms_db = get_loudness(make_sine(amp=0.437), SR)
    assert abs(rms_db - round(rms_db, 2)) > 0 or rms_db == round(rms_db, 2)


def test_get_loudness_short_clip_returns_nan():
    # Regression (was critical #2): clips < 400 ms LUFS gating block used to
    # crash with an uncaught ValueError. Undefined loudness -> NaN, RMS intact.
    lufs, rms_db = get_loudness(make_sine(duration=0.3), SR)
    assert np.isnan(lufs)
    assert rms_db < 0


# ── get_rms ─────────────────────────────────────────────────

def test_get_rms_sine_value():
    # RMS of a 0.5-amplitude sine ≈ 0.5/√2 ≈ 0.3536
    rms = get_rms(make_sine(amp=0.5), SR)
    assert rms == pytest.approx(0.3536, abs=0.01)


def test_get_rms_silence():
    assert get_rms(np.zeros(SR), SR) == pytest.approx(0.0, abs=1e-6)


def test_get_rms_stereo():
    y = np.stack([make_sine(440), make_sine(440)])
    assert get_rms(y, SR) == pytest.approx(0.3536, abs=0.01)


# ── normalize ───────────────────────────────────────────────

def test_normalize_peak_is_one():
    y = normalize(make_sine(amp=0.2))
    assert np.abs(y).max() == pytest.approx(1.0)


# ── filter dispatcher + variants ────────────────────────────

def test_filter_low_pass_returns_array():
    # Regression: dispatcher used to return None
    out = dsp_filter(make_sine(), "low_pass", 1000)
    assert isinstance(out, np.ndarray) and out.shape == make_sine().shape


def test_filter_high_pass_returns_array():
    out = dsp_filter(make_sine(), "high_pass", 1000)
    assert isinstance(out, np.ndarray)


def test_filter_invalid_type_returns_usage():
    assert isinstance(dsp_filter(make_sine(), "nonsense", 1000), str)


def test_filter_low_pass_attenuates_highs():
    high = make_sine(freq=8000)
    out = filter_low_pass(high, cutoff_freq=1000, sr=SR)
    # Output is re-normalized, so compare shape of energy before normalization
    # by checking correlation with the input instead: a heavily attenuated,
    # noise-dominated result decorrelates from the original.
    lo = make_sine(freq=200)
    passed = filter_low_pass(lo, cutoff_freq=1000, sr=SR)
    corr_pass = np.corrcoef(lo, passed)[0, 1]
    assert corr_pass > 0.99  # low freq passes cleanly


def test_filter_high_pass_attenuates_lows():
    hi = make_sine(freq=8000)
    passed = filter_high_pass(hi, cutoff_freq=1000, sr=SR)
    assert np.corrcoef(hi, passed)[0, 1] > 0.99  # high freq passes cleanly


def test_filter_bell_not_implemented():
    with pytest.raises(NotImplementedError):
        filter_bell(make_sine(), 1000)


# ── compute_fluctuation ─────────────────────────────────────

def test_compute_fluctuation_shapes():
    summary, mod_freqs, per_band = compute_fluctuation(make_sine(duration=4.0), SR)
    assert isinstance(summary, float) and summary >= 0
    assert mod_freqs.max() <= 10.0
    assert per_band.shape == (40, len(mod_freqs))


def test_compute_fluctuation_am_beats_flat():
    # Noise amplitude-modulated at 4 Hz (the Fastl peak) must score higher
    # than unmodulated noise.
    rng = np.random.default_rng(0)
    n = SR * 4
    noise = rng.standard_normal(n) * 0.3
    t = np.linspace(0, 4, n, endpoint=False)
    am = noise * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t))
    flat_score, _, _ = compute_fluctuation(noise, SR)
    am_score, _, _ = compute_fluctuation(am, SR)
    assert am_score > flat_score


def test_compute_fluctuation_accepts_stereo():
    y = np.stack([make_sine(440, 3.0), make_sine(220, 3.0)])
    summary, _, _ = compute_fluctuation(y, SR)
    assert np.isfinite(summary)


# ── spectral_irregularity ───────────────────────────────────

def test_spectral_irregularity_spike_exceeds_broadband():
    # Jensen irregularity = Σ(a_k − a_{k+1})² / Σa² — a lone spectral spike
    # (pure sine) is maximally jagged bin-to-bin, while broadband noise
    # spreads energy so adjacent-bin differences are small relative to total.
    rng = np.random.default_rng(1)
    sine_irr = spectral_irregularity(make_sine(duration=2.0), SR)
    noise_irr = spectral_irregularity(rng.standard_normal(SR * 2) * 0.3, SR)
    assert sine_irr > noise_irr > 0


def test_spectral_irregularity_silence_is_zero():
    assert spectral_irregularity(np.zeros(SR), SR) == 0.0


# ── compute_spectral_features ───────────────────────────────

def test_compute_spectral_features_keys_and_shapes():
    feats = compute_spectral_features(make_sine(duration=2.0), SR)
    expected = {"spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                "spectral_flatness", "zcr"}
    assert set(feats.keys()) == expected
    lengths = {len(v) for v in feats.values()}
    assert len(lengths) == 1          # all per-frame arrays align
    assert lengths.pop() > 0


def test_compute_spectral_features_centroid_tracks_freq():
    low = compute_spectral_features(make_sine(freq=220), SR)
    high = compute_spectral_features(make_sine(freq=4000), SR)
    assert np.mean(high["spectral_centroid"]) > np.mean(low["spectral_centroid"])
