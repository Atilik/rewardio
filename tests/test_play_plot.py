"""Tests for rewardio/play.py and rewardio/plot.py.

The GUI windows (Tk + audio device) cannot be driven headlessly — those are
manual-test territory. Here we cover the pure helpers and the public API
surface so refactors that break imports/signatures fail fast.
"""
import inspect

import pytest


# ── pure helpers ────────────────────────────────────────────

def test_format_time():
    from rewardio.play import _format_time
    assert _format_time(0) == "0:00"
    assert _format_time(5) == "0:05"
    assert _format_time(65) == "1:05"
    assert _format_time(59.9) == "0:59"      # truncates, not rounds
    assert _format_time(600) == "10:00"
    assert _format_time(3599) == "59:59"


# ── API surface (regression against accidental renames) ─────

def test_play_module_api():
    import rewardio.play as play
    for name in ("play_audio", "play_interactive",
                 "sonify_beats", "sonify_beats_and_onsets"):
        assert callable(getattr(play, name)), f"play.{name} missing"


def test_plot_module_api():
    import rewardio.plot as plot
    for name in ("plot_beats", "plot_waveform", "plot_beats_and_onsets",
                 "plot_interactive", "plot_session_boxplots",
                 "plot_spectrogram", "plot_pitch"):
        assert callable(getattr(plot, name)), f"plot.{name} missing"


def test_play_interactive_signature():
    from rewardio.play import play_interactive
    params = inspect.signature(play_interactive).parameters
    assert list(params) == ["stimulus", "xlim", "ylim"]


def test_plot_spectrogram_scales_documented():
    from rewardio.plot import plot_spectrogram
    params = inspect.signature(plot_spectrogram).parameters
    assert params["scale"].default == "mel"


def test_stimulus_gui_methods_exist(sine_wav):
    from rewardio.rewardio import Stimulus
    s = Stimulus(sine_wav)
    for name in ("play", "plot", "plot_beats", "plot_waveform", "plot_onsets",
                 "sonify_beats", "sonify_onsets"):
        assert callable(getattr(s, name))
