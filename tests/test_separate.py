"""Tests for rewardio/separate.py.

Default tests only exercise the fast validation paths (no model, no audio
decode). The real Demucs separation runs only with REWARDIO_RUN_SLOW=1 AND
a cached model checkpoint.
"""
import glob
import os

import numpy as np
import pytest

from rewardio.separate import separate
from conftest import RUN_SLOW


def _demucs_checkpoint_cached():
    hub = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    return bool(glob.glob(os.path.join(hub, "*.th")))


# ── validation (fast, no model) ─────────────────────────────

def test_separate_rejects_unknown_target(sine_wav):
    # Regression (#1 first-pass): substring matching allowed typos through;
    # validation now happens before any audio/model loading.
    with pytest.raises(ValueError, match="Unknown target source"):
        separate(sine_wav, target_source="drum")


def test_separate_rejects_none_target(sine_wav):
    with pytest.raises(ValueError):
        separate(sine_wav, target_source=None)


def test_separate_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "audio.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="Unknown target source|Unsupported audio format"):
        separate(str(bad), target_source="drums")


# ── full separation (slow, model required) ──────────────────

@pytest.mark.skipif(
    not (RUN_SLOW and _demucs_checkpoint_cached()),
    reason="set REWARDIO_RUN_SLOW=1 (and have htdemucs cached) to run Demucs",
)
def test_separate_drums_end_to_end(click_wav):
    target, accompaniment, sr = separate(click_wav, target_source="drums")
    assert isinstance(target, np.ndarray) and target.ndim == 1
    assert isinstance(accompaniment, np.ndarray) and accompaniment.ndim == 1
    assert len(target) == len(accompaniment)
    assert sr > 0
