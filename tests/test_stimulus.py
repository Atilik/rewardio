"""Tests for the Stimulus class (rewardio/rewardio.py).

ML-dependent attributes (beats, separation, classification) are exercised
through data injection so no model runs.
"""
import csv
import os

import numpy as np
import pytest

from rewardio.rewardio import Stimulus


# ── construction / validation ───────────────────────────────

def test_init_missing_file_raises():
    # Regression (#4): used to construct fine and explode later
    with pytest.raises(FileNotFoundError):
        Stimulus("/definitely/not/here.wav")


def test_init_unreadable_file_raises(tmp_path):
    bad = tmp_path / "garbage.wav"
    bad.write_text("not audio")
    with pytest.raises(Exception):
        Stimulus(str(bad))


def test_init_is_lazy(sine_wav):
    s = Stimulus(sine_wav)
    assert s._y is None                    # no audio loaded yet
    _ = s.y
    assert s._y is not None


# ── audio properties ────────────────────────────────────────

def test_audio_properties_mono(sine_wav):
    s = Stimulus(sine_wav)
    assert s.n_channels == 1
    assert s.duration == pytest.approx(2.0, abs=1e-3)
    assert s.sr == 44100
    assert s.y.ndim == 1


def test_audio_properties_stereo(stereo_wav):
    s = Stimulus(stereo_wav)
    assert s.n_channels == 2
    assert s.y.shape[0] == 2
    assert s.duration == pytest.approx(2.0, abs=1e-3)


def test_to_mono(stereo_wav):
    s = Stimulus(stereo_wav)
    assert s._to_mono().ndim == 1
    assert len(s._to_mono()) == s.y.shape[1]


# ── loudness (computed, cached) ─────────────────────────────

def test_loudness_properties(sine_wav):
    s = Stimulus(sine_wav)
    assert -70 < s.loudness_lufs < 0
    assert -70 < s.loudness_rms_db < 0
    assert 0 < s.rms < 1
    assert s._loudness_lufs is not None    # cached after first access


# ── rhythm-derived properties via injection ─────────────────

def test_beat_properties_from_injection(sine_wav, inject_rhythm):
    s = Stimulus(sine_wav)
    bt = inject_rhythm(s)
    assert np.array_equal(s.beat_times, bt)     # no detection triggered
    assert s.meter == 4
    assert s.bpm == pytest.approx(120.0)
    assert s.BPM == s.bpm                       # alias
    assert len(s.beat_positions) == len(bt)


def test_beat_ioi_stats(sine_wav, inject_rhythm):
    s = Stimulus(sine_wav)
    bt = inject_rhythm(s)
    assert s.beat_ioi_mean == pytest.approx(0.5, abs=1e-9)
    assert s.beat_ioi_std == pytest.approx(0.0, abs=1e-9)


def test_beat_ioi_empty_is_zero(sine_wav):
    s = Stimulus(sine_wav)
    s._beat_times = np.array([1.0])
    assert s.beat_ioi_mean == 0.0 and s.beat_ioi_std == 0.0


def test_onset_ioi_stats(sine_wav):
    s = Stimulus(sine_wav)
    assert s.onset_ioi_mean == 0.0              # None -> 0.0
    s.onset_times = np.array([0.0, 0.25, 0.5, 0.75])
    assert s.onset_ioi_mean == pytest.approx(0.25)
    assert s.onset_ioi_std == pytest.approx(0.0, abs=1e-9)


def test_onset_detection_with_explicit_y(sine_wav):
    from conftest import make_clicks
    s = Stimulus(sine_wav)
    times = np.arange(0.2, 1.8, 0.2)
    y = make_clicks(times, 2.0)
    detected = s.onset_detection(y=y)
    assert abs(len(detected) - len(times)) <= 1
    assert s.onset_times is detected


def test_syncopation_score_default_and_meter(sine_wav, inject_rhythm):
    s = Stimulus(sine_wav)
    inject_rhythm(s)
    score = s.syncopation_score()
    assert score == s.toussaint_syncopation_score
    assert 0 <= score <= 100
    assert s.onset_times is not None            # side effect: onsets stored
    score_m = s.syncopation_score(meter=True)
    assert score_m == s.toussaint_syncopation_score_meter
    assert 0 <= score_m <= 100


def test_detect_onsets_computes_score(sine_wav, inject_rhythm, capsys):
    s = Stimulus(sine_wav)
    inject_rhythm(s)
    s.detect_onsets()
    out = capsys.readouterr().out
    assert "onsets" in out.lower()
    assert s.onset_times is not None
    assert s.toussaint_syncopation_score is not None


def test_separate_declined_by_user(sine_wav, monkeypatch, capsys):
    s = Stimulus(sine_wav)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    result = s.separate()                       # confirm=True, user says no
    assert result is None
    assert s.separated_drums is None
    assert "Skipped" in capsys.readouterr().out


# ── classification properties via injection ─────────────────

def test_genre_properties_from_injection(sine_wav):
    s = Stimulus(sine_wav)
    s._genre_predictions = [("Rock---Classic Rock", 0.8), ("Pop", 0.1)]
    assert s.genre == "Rock---Classic Rock"
    assert s.genre_top5[0][1] == 0.8


def test_mood_and_voice_from_injection(sine_wav):
    s = Stimulus(sine_wav)
    s._mood = {"happy": 0.7, "sad": 0.1, "aggressive": 0.05, "relaxed": 0.15}
    s._voice_instrumental = [("instrumental", 0.9), ("voice", 0.1)]
    assert s.mood["happy"] == 0.7
    assert s.voice_instrumental == "instrumental"


def test_pitch_properties_voiced_filtering(sine_wav):
    s = Stimulus(sine_wav)
    s._pitch_time = np.array([0.0, 0.1, 0.2, 0.3])
    s._pitch_freq = np.array([100.0, 200.0, 300.0, 400.0])
    s._pitch_conf = np.array([0.9, 0.3, 0.8, 0.6])   # 200 Hz is unvoiced
    assert s.pitch == pytest.approx(300.0)            # median of [100, 300, 400]
    assert s.pitch_mean == pytest.approx(np.mean([100, 300, 400]))
    assert s.pitch_std > 0
    assert s.pitch_conf_mean == pytest.approx(np.mean([0.9, 0.3, 0.8, 0.6]))


def test_pitch_all_unvoiced_is_zero(sine_wav):
    s = Stimulus(sine_wav)
    s._pitch_time = np.array([0.0, 0.1])
    s._pitch_freq = np.array([100.0, 200.0])
    s._pitch_conf = np.array([0.1, 0.2])
    assert s.pitch == 0.0 and s.pitch_mean == 0.0 and s.pitch_std == 0.0


# ── spectral feature properties ─────────────────────────────

def test_spectral_properties(sine_wav):
    s = Stimulus(sine_wav)
    assert s.spectral_centroid_mean > 0
    assert s.spectral_bandwidth_mean > 0
    assert s.spectral_rolloff_mean > 0
    assert 0 <= s.spectral_flatness_mean <= 1
    assert 0 <= s.zcr_mean <= 1
    assert s._spectral_features is not None     # cached


def test_fluctuation_and_irregularity(sine_wav):
    s = Stimulus(sine_wav)
    assert isinstance(s.fluctuation, float)
    assert isinstance(s.irregularity, float)
    assert s.fluctuation >= 0 and s.irregularity >= 0


# ── CSV / npz export ────────────────────────────────────────

def test_collect_attrs_schema(sine_wav, inject_rhythm):
    s = Stimulus(sine_wav)
    inject_rhythm(s)
    s.onset_times = np.array([0.5, 1.0, 1.5])
    s.toussaint_syncopation_score = 12
    s.toussaint_syncopation_score_meter = 15
    s._genre_predictions = [("Rock", 0.8)]
    s._voice_instrumental = [("voice", 0.6)]
    s._mood = {"happy": 0.5, "sad": 0.2, "aggressive": 0.1, "relaxed": 0.2}
    s._pitch_freq = np.array([200.0, 300.0])
    s._pitch_conf = np.array([0.9, 0.8])
    s._key, s._scale, s._key_strength = "C", "major", 0.85
    s._compute_spectral_if_needed()

    row = s._collect_attrs()
    expected = {
        "filename", "duration", "sr", "n_channels",
        "loudness_lufs", "loudness_rms_db", "rms",
        "bpm", "n_beats", "n_onsets",
        "syncopation_score", "syncopation_score_meter", "meter",
        "genre", "genre_confidence", "voice_instrumental",
        "mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed",
        "pitch_median_hz", "pitch_mean_hz", "pitch_std_hz",
        "pitch_conf_mean", "pitch_conf_std",
        "beat_ioi_mean", "beat_ioi_std", "onset_ioi_mean", "onset_ioi_std",
        "key", "scale", "key_strength",
        "fluctuation", "spectral_irregularity",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_flatness_mean", "spectral_flatness_std",
        "zcr_mean", "zcr_std",
    }
    assert expected.issubset(set(row.keys()))


def test_save_writes_csv(sine_wav, tmp_path):
    s = Stimulus(sine_wav)
    s.save(output_path=str(tmp_path))
    analysis = [d for d in os.listdir(tmp_path) if d.startswith("Analysis_")]
    assert len(analysis) == 1
    csvs = os.listdir(tmp_path / analysis[0])
    assert len(csvs) == 1
    with open(tmp_path / analysis[0] / csvs[0]) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["filename"] == "sine_440.wav"


def test_save_timeseries_with_data(sine_wav, tmp_path):
    s = Stimulus(sine_wav)
    s._pitch_time = np.array([0.0, 0.1])
    s._pitch_freq = np.array([220.0, 221.0])
    s._pitch_conf = np.array([0.9, 0.9])
    s._beat_times = np.array([0.5, 1.0])
    s.onset_times = np.array([0.5])
    s._compute_spectral_if_needed()
    s.save_timeseries(output_path=str(tmp_path))
    analysis = [d for d in os.listdir(tmp_path) if d.startswith("Analysis_")][0]
    npz = [f for f in os.listdir(tmp_path / analysis) if f.endswith(".npz")]
    assert len(npz) == 1
    data = np.load(tmp_path / analysis / npz[0])
    for key in ("pitch_time", "pitch_freq", "pitch_conf", "beat_times",
                "onset_times", "spectral_centroid", "zcr"):
        assert key in data.files


def test_save_timeseries_without_data(sine_wav, tmp_path, capsys):
    s = Stimulus(sine_wav)
    s.save_timeseries(output_path=str(tmp_path))
    assert "No time series" in capsys.readouterr().out


# ── partial_process_save ────────────────────────────────────

def _read_saved_csv(out_dir):
    analysis = [d for d in os.listdir(out_dir) if d.startswith("Analysis_")][0]
    csv_file = os.listdir(os.path.join(out_dir, analysis))[0]
    with open(os.path.join(out_dir, analysis, csv_file)) as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_partial_process_save_rhythm_only(sine_wav, tmp_path, inject_rhythm):
    # rhythm = beats/BPM only — must NOT run the syncopation pipeline
    s = Stimulus(sine_wav)
    inject_rhythm(s)                            # beats+drums present, no ML
    s.partial_process_save(output_path=str(tmp_path), rhythm=True)
    header, rows = _read_saved_csv(str(tmp_path))
    assert "bpm" in header and "n_beats" in header
    assert "syncopation_score" not in header
    assert "genre" not in header and "pitch_median_hz" not in header


def test_partial_process_save_syncopation_only(sine_wav, tmp_path, inject_rhythm):
    s = Stimulus(sine_wav)
    inject_rhythm(s)                            # drums injected -> no Demucs
    s.partial_process_save(output_path=str(tmp_path), syncopation=True)
    header, rows = _read_saved_csv(str(tmp_path))
    assert "syncopation_score" in header and "syncopation_score_meter" in header
    assert rows[0]["syncopation_score"] != ""
    assert "genre" not in header


def test_partial_process_save_genre_only(sine_wav, tmp_path, monkeypatch):
    import rewardio.rewardio as rw
    monkeypatch.setattr(rw, "_classify_all", lambda p: {
        "genre": [("Rock---Classic Rock", 0.9)],
        "voice_instrumental": [("voice", 0.8), ("instrumental", 0.2)],
        "mood": {"happy": 0.6, "sad": 0.1, "aggressive": 0.1, "relaxed": 0.2},
    })
    s = Stimulus(sine_wav)
    s.partial_process_save(output_path=str(tmp_path), genre=True)
    header, rows = _read_saved_csv(str(tmp_path))
    assert rows[0]["genre"] == "Rock---Classic Rock"
    assert rows[0]["voice_instrumental"] == "voice"
    assert "mood_happy" in header
    assert "bpm" not in header and "key" not in header


def test_partial_process_save_pitch_only(sine_wav, tmp_path, monkeypatch):
    import rewardio.rewardio as rw
    monkeypatch.setattr(rw, "_detect_pitch_crepe", lambda p: (
        np.array([0.0, 0.1, 0.2]),
        np.array([440.0, 441.0, 439.0]),
        np.array([0.9, 0.95, 0.9]),
    ))
    s = Stimulus(sine_wav)
    s.partial_process_save(output_path=str(tmp_path), pitch=True)
    header, rows = _read_saved_csv(str(tmp_path))
    assert float(rows[0]["pitch_median_hz"]) == pytest.approx(440.0)
    assert "genre" not in header and "bpm" not in header


def test_partial_process_save_key_real(sine_wav, tmp_path):
    # KeyExtractor is algorithmic (no model files) — run it for real
    s = Stimulus(sine_wav)
    s.partial_process_save(output_path=str(tmp_path), key=True)
    header, rows = _read_saved_csv(str(tmp_path))
    assert rows[0]["key"] != ""
    assert rows[0]["scale"] in ("major", "minor")
    assert "genre" not in header


def test_partial_process_save_spectral_only(sine_wav, tmp_path):
    s = Stimulus(sine_wav)
    s.partial_process_save(output_path=str(tmp_path), spectral=True)
    header, rows = _read_saved_csv(str(tmp_path))
    assert "spectral_centroid_mean" in header and "zcr_std" in header
    assert "bpm" not in header


def test_partial_process_save_no_groups_hint(sine_wav, tmp_path, capsys):
    s = Stimulus(sine_wav)
    s.partial_process_save(output_path=str(tmp_path))
    assert "No feature group selected" in capsys.readouterr().out
    header, _ = _read_saved_csv(str(tmp_path))
    assert "loudness_lufs" in header            # basics always saved


def test_short_file_loudness_and_save(tmp_path):
    # Regression (was critical #2): a < 400 ms file must not crash the
    # save/export path — LUFS is NaN, everything else still computes.
    import soundfile as sf
    from conftest import make_sine
    p = str(tmp_path / "blip.wav")
    sf.write(p, make_sine(duration=0.3), 44100)
    s = Stimulus(p)
    assert np.isnan(s.loudness_lufs)
    assert s.loudness_rms_db < 0
    assert s.rms > 0
    s.save(output_path=str(tmp_path))           # must not raise
    analysis = [d for d in os.listdir(tmp_path) if d.startswith("Analysis_")]
    assert len(analysis) == 1


# ── repr ────────────────────────────────────────────────────

def test_repr_formatted(sine_wav):
    s = Stimulus(sine_wav)
    r = repr(s)
    assert 'Stimulus("sine_440.wav")' in r
    assert "LUFS" in r and "(...)" in r          # BPM not yet computed
    # Loudness shown with 2 decimals (regression for display formatting)
    import re
    assert re.search(r"Loudness : -?\d+\.\d{2} LUFS", r)
