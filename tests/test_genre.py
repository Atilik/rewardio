"""Tests for rewardio/genre.py.

The TF-model classifiers need .pb files under rewardio/models/ (currently
absent) — those tests auto-skip and will activate once models are installed.
detect_key and the 16 kHz loader are algorithmic (no model files) and run on
tiny dummy audio.
"""
import numpy as np
import pytest
import soundfile as sf

from rewardio.genre import (
    detect_key, _load_audio_16k, _EFFNET_PATH, _CREPE_PATH,
)
import os

MODELS_PRESENT = os.path.isfile(_EFFNET_PATH)
CREPE_PRESENT = os.path.isfile(_CREPE_PATH)
SR = 44100


@pytest.fixture(scope="module")
def c_major_wav(tmp_path_factory):
    """3 s C-major chord (C4+E4+G4+C5) — unambiguous key content."""
    t = np.linspace(0, 3.0, SR * 3, endpoint=False)
    y = sum(0.2 * np.sin(2 * np.pi * f * t)
            for f in (261.63, 329.63, 392.00, 523.25))
    p = str(tmp_path_factory.mktemp("key") / "c_major.wav")
    sf.write(p, y, SR)
    return p


# ── detect_key (algorithmic — always runs) ──────────────────

def test_detect_key_c_major(c_major_wav):
    key, scale, strength = detect_key(c_major_wav)
    assert key == "C"
    assert scale == "major"
    assert 0 <= strength <= 1


def test_detect_key_returns_types(sine_wav):
    key, scale, strength = detect_key(sine_wav)
    assert isinstance(key, str) and isinstance(scale, str)
    assert isinstance(strength, float)


# ── _load_audio_16k ─────────────────────────────────────────

def test_load_audio_16k_resamples(sine_wav):
    audio = _load_audio_16k(sine_wav)          # 2 s file
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(2 * 16000, rel=0.01)


# ── TF-model classifiers (auto-activate when models installed) ──
# EffNet needs one full mel patch (~2.1 s at 16 kHz), so classifier tests
# use a 4 s fixture — the shared 2 s sine_wav is deliberately too short.

@pytest.fixture(scope="module")
def long_wav(tmp_path_factory):
    """4 s tone mixture — long enough for a Discogs-EffNet patch."""
    t = np.linspace(0, 4.0, SR * 4, endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    p = str(tmp_path_factory.mktemp("clf") / "long_tone.wav")
    sf.write(p, y, SR)
    return p


@pytest.mark.skipif(not MODELS_PRESENT, reason="rewardio/models/*.pb not installed")
def test_classify_too_short_audio_raises_clearly(sine_wav):
    # Regression: < ~2.1 s used to surface as a cryptic Essentia TypeError
    from rewardio.genre import classify_genre
    with pytest.raises(ValueError, match="too short"):
        classify_genre(sine_wav)               # 2 s — below one EffNet patch


@pytest.mark.skipif(not MODELS_PRESENT, reason="rewardio/models/*.pb not installed")
def test_classify_genre(long_wav):
    from rewardio.genre import classify_genre
    preds = classify_genre(long_wav, top_n=3)
    assert len(preds) == 3
    assert all(0 <= c <= 1 for _, c in preds)


@pytest.mark.skipif(not MODELS_PRESENT, reason="rewardio/models/*.pb not installed")
def test_classify_voice_instrumental(long_wav):
    from rewardio.genre import classify_voice_instrumental
    preds = classify_voice_instrumental(long_wav)
    labels = {label for label, _ in preds}
    assert labels <= {"voice", "instrumental"}


@pytest.mark.skipif(not MODELS_PRESENT, reason="rewardio/models/*.pb not installed")
def test_classify_mood(long_wav):
    from rewardio.genre import classify_mood
    moods = classify_mood(long_wav)
    assert set(moods.keys()) == {"happy", "sad", "aggressive", "relaxed"}
    assert all(0 <= v <= 1 for v in moods.values())


@pytest.mark.skipif(not MODELS_PRESENT, reason="rewardio/models/*.pb not installed")
def test_classify_all_shares_embeddings(long_wav):
    from rewardio.genre import classify_all
    results = classify_all(long_wav)
    assert set(results.keys()) == {"genre", "voice_instrumental", "mood"}


@pytest.mark.skipif(not CREPE_PRESENT, reason="crepe-medium-1.pb not installed")
def test_detect_pitch_crepe(sine_wav):
    from rewardio.genre import detect_pitch_crepe
    time, freq, conf = detect_pitch_crepe(sine_wav)     # 440 Hz sine
    voiced = freq[conf > 0.5]
    assert len(voiced) > 0
    assert np.median(voiced) == pytest.approx(440, rel=0.03)
