"""
Shared fixtures for the rewardio test suite.

All audio is synthetic (sine waves / click tracks) so tests are fast,
deterministic, and need no external data or ML models.

Heavy paths (Demucs, BEAT THIS!, Essentia classifiers) are gated behind
REWARDIO_RUN_SLOW=1 and model-cache checks — see test_separate.py / test_genre.py.
"""
import os
import sys
import shutil

import numpy as np
import pytest
import soundfile as sf

# Make `import rewardio.*` work no matter where pytest is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 44100

RUN_SLOW = bool(os.environ.get("REWARDIO_RUN_SLOW"))


# ── Signal builders ─────────────────────────────────────────

def make_sine(freq=440.0, duration=2.0, sr=SR, amp=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def make_clicks(times, duration, sr=SR, click_len=200):
    """Impulse train (Hann bursts) at the given times, peak-normalized."""
    y = np.zeros(int(duration * sr))
    for t in times:
        i = int(t * sr)
        if i + click_len < len(y):
            y[i:i + click_len] += np.hanning(click_len)
    peak = np.abs(y).max()
    return 0.9 * y / peak if peak > 0 else y


# ── Audio-file fixtures (session-scoped: read-only) ─────────

@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("audio")


@pytest.fixture(scope="session")
def sine_wav(audio_dir):
    """2 s mono 440 Hz sine at 44.1 kHz."""
    p = str(audio_dir / "sine_440.wav")
    sf.write(p, make_sine(), SR)
    return p


@pytest.fixture(scope="session")
def stereo_wav(audio_dir):
    """2 s stereo file (440 Hz left, 220 Hz right)."""
    y = np.stack([make_sine(440), make_sine(220)], axis=1)  # (n, 2) for soundfile
    p = str(audio_dir / "stereo.wav")
    sf.write(p, y, SR)
    return p


@pytest.fixture(scope="session")
def click_wav(audio_dir):
    """10 s click track at 120 BPM (clicks every 0.5 s from t=0.5)."""
    times = np.arange(0.5, 9.5, 0.5)
    p = str(audio_dir / "clicks_120bpm.wav")
    sf.write(p, make_clicks(times, 10.0), SR)
    return p


# ── Folder fixtures (function-scoped: tests may mutate objects) ──

@pytest.fixture
def session_folder(tmp_path, sine_wav, stereo_wav):
    """Folder with 2 audio files + 1 non-audio file."""
    shutil.copy(sine_wav, tmp_path / "01_alpha.wav")
    shutil.copy(stereo_wav, tmp_path / "02_beta.wav")
    (tmp_path / "notes.txt").write_text("not audio")
    return str(tmp_path)


@pytest.fixture
def participant_folder(tmp_path, sine_wav):
    """Folder with 2 session sub-folders, 1 song each."""
    for sess in ("sess_a", "sess_b"):
        d = tmp_path / sess
        d.mkdir()
        shutil.copy(sine_wav, d / f"{sess}_song.wav")
    return str(tmp_path)


# ── Rhythm-data injection (bypasses BEAT THIS! / Demucs) ────

@pytest.fixture
def inject_rhythm():
    """
    Inject synthetic beat data + fake separated drums into a Stimulus so
    syncopation paths run without any ML model.

    Drums are clicks exactly ON the beats -> deterministic low-syncopation
    pattern (onsets land on slot 0 of every beat).
    """
    def _inject(stimulus, sr=SR, dur=10.0, step=0.5):
        bt = np.arange(0.5, dur - 0.5, step)
        stimulus._beat_times = bt
        stimulus._beat_positions = np.array([(i % 4) + 1 for i in range(len(bt))])
        stimulus._meter = 4
        stimulus._bpm = 60.0 / step
        stimulus.separated_drums = make_clicks(bt, dur, sr=sr)
        stimulus.separated_drums_sr = sr
        return bt
    return _inject
