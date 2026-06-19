import os
import pytest
import numpy as np
import soundfile as sf

from rewardio.rewardio import Stimulus

TEST_AUDIO = "tests/dummy_test_audio.wav"

@pytest.fixture(scope="session", autouse=True)
def create_dummy_audio():
    # Generate 1 second of a 440 Hz sine wave at 44.1 kHz
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    os.makedirs("tests", exist_ok=True)
    sf.write(TEST_AUDIO, y, sr)
    
    yield
    
    # Cleanup after tests
    if os.path.exists(TEST_AUDIO):
        os.remove(TEST_AUDIO)

def test_stimulus_basic():
    s = Stimulus(TEST_AUDIO)
    assert s.audio_file_name == "dummy_test_audio.wav"
    # duration might be slightly off depending on numpy rounding, but 1.0 should be exact for 44100 samples
    assert s.duration == 1.0
    assert s.sr == 44100
    assert s.n_channels == 1
    assert s.y is not None

def test_stimulus_dsp():
    s = Stimulus(TEST_AUDIO)
    assert isinstance(s.loudness_lufs, float)
    assert isinstance(s.rms, float)
    assert s.rms > 0
    
    # Check spectral features
    assert isinstance(s.spectral_centroid_mean, float)
    assert s.spectral_centroid_mean > 0

    # Fluctuation and irregularity
    assert isinstance(s.fluctuation, float)
    assert isinstance(s.irregularity, float)
