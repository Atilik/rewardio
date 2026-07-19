"""Tests for rewardio/core.py — loading, CSV export, help/print helpers."""
import csv
import os

import numpy as np
import pytest

from rewardio.core import (
    load_audio, write_to_csv, clear,
    stimulus_help, stimulus_print, stimulus_print_all,
    session_help, session_print,
    participant_help, participant_print,
)
from rewardio.rewardio import Stimulus, Session, Participant


# ── load_audio ──────────────────────────────────────────────

def test_load_audio_mono_file(sine_wav):
    y, sr = load_audio(sine_wav)
    assert sr == 44100
    assert y.ndim == 1
    assert len(y) == 2 * 44100


def test_load_audio_preserves_stereo(stereo_wav):
    y, sr = load_audio(stereo_wav)
    assert y.ndim == 2 and y.shape[0] == 2


def test_load_audio_mono_flag_downmixes(stereo_wav):
    y, sr = load_audio(stereo_wav, mono=True)
    assert y.ndim == 1


def test_load_audio_resamples(sine_wav):
    y, sr = load_audio(sine_wav, sr=22050)
    assert sr == 22050
    assert len(y) == 2 * 22050


# ── write_to_csv ────────────────────────────────────────────

def test_write_to_csv_returns_path_and_writes(tmp_path):
    # Regression: used to return None
    path = write_to_csv([{"a": 1, "b": 2.5}], output_path=str(tmp_path))
    assert path is not None and os.path.isfile(path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["a"] == "1" and rows[0]["b"] == "2.5"


def test_write_to_csv_filename_has_seconds(tmp_path):
    # Regression: minute-resolution names silently overwrote each other
    path = write_to_csv([{"x": 1}], output_path=str(tmp_path))
    base = os.path.basename(path)          # rewardio_HH-MM-SS.csv
    assert base.count("-") == 2
    assert base.startswith("rewardio_") and base.endswith(".csv")


def test_write_to_csv_sparse_rows_union_header(tmp_path):
    path = write_to_csv([{"a": 1, "b": 2}, {"a": 3, "c": 4}], output_path=str(tmp_path))
    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["a", "b", "c"]
        rows = list(reader)
    assert rows[0]["c"] == "" and rows[1]["b"] == ""


def test_write_to_csv_creates_analysis_folder(tmp_path):
    path = write_to_csv([{"x": 1}], output_path=str(tmp_path))
    assert os.path.basename(os.path.dirname(path)).startswith("Analysis_")


# ── help / print helpers (smoke: run + key content, no crash) ──

def test_stimulus_help_prints_attributes(sine_wav, capsys):
    stimulus_help(Stimulus(sine_wav))
    out = capsys.readouterr().out
    assert ".loudness_lufs" in out and ".play(" in out


def test_stimulus_print_summary(sine_wav, capsys):
    stimulus_print(Stimulus(sine_wav))
    out = capsys.readouterr().out
    assert "Int.LUFS" in out and "sine_440.wav" in out
    assert "BPM      : (...)" in out       # lazy — must not trigger detection


def test_stimulus_print_all_lists_state(sine_wav, capsys):
    stimulus_print_all(Stimulus(sine_wav))
    out = capsys.readouterr().out
    assert "bpm" in out and "(Not calculated)" in out


def test_session_help_and_print(session_folder, capsys):
    sess = Session(session_folder)
    session_help(sess)
    session_print(sess)
    out = capsys.readouterr().out
    assert "session(i)" in out
    assert ".average_fluctuation" in out and ".partial_process_save" in out
    assert "01_alpha.wav" in out and "02_beta.wav" in out


def test_participant_help_and_print(participant_folder, capsys):
    p = Participant(participant_folder)
    participant_help(p)
    participant_print(p)
    out = capsys.readouterr().out
    assert "participant(i)" in out
    assert ".average_irregularity" in out and ".partial_process_save" in out
    assert "sess_a" in out and "sess_b" in out


def test_clear_runs_without_error():
    clear()
