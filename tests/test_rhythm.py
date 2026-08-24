"""Tests for rewardio/rhythm.py.

Pure-logic functions are tested directly. Model-based beat detection
(BEAT THIS! / madmom) is gated behind REWARDIO_RUN_SLOW=1.
"""
import math
import os

import numpy as np
import pytest

from rewardio.rhythm import (
    get_bpm, onset_detection, syncopation_score,
    _pattern_from_beats_and_onsets, _score_pattern,
    _derive_positions_and_meter, METER_WEIGHTS,
)
from conftest import make_clicks, SR, RUN_SLOW


# ── get_bpm ─────────────────────────────────────────────────

def test_get_bpm_120():
    assert get_bpm(np.arange(0, 10, 0.5)) == pytest.approx(120.0)


def test_get_bpm_returns_float_with_sub_bpm_precision():
    # Regression (#6): used to quantize to int
    bpm = get_bpm(np.arange(0, 20, 0.483))
    assert isinstance(bpm, float)
    assert bpm == pytest.approx(124.22, abs=0.01)


def test_get_bpm_too_few_beats():
    assert get_bpm(np.array([1.0])) == 0
    assert get_bpm(np.array([])) == 0


def test_get_bpm_robust_to_outlier():
    # Median-based: one missed beat (double interval) must not shift BPM
    bt = np.concatenate([np.arange(0, 5, 0.5), [6.0], np.arange(6.5, 10, 0.5)])
    assert get_bpm(bt) == pytest.approx(120.0)


# ── METER_WEIGHTS sanity ────────────────────────────────────

def test_meter_weights_lengths():
    for meter, w in METER_WEIGHTS.items():
        assert len(w) == meter * 4     # 4 subdivisions per beat
        assert w[0] == 5               # downbeat carries max weight


# ── _pattern_from_beats_and_onsets ──────────────────────────

def test_pattern_onsets_on_beats():
    bt = np.arange(0.5, 8.5, 0.5)          # 16 beats
    grid, num_bars = _pattern_from_beats_and_onsets(bt, bt)
    assert num_bars == (len(bt) - 1) // 4
    slots = np.where(grid == 1)[0]
    assert len(slots) > 0
    assert np.all(slots % 4 == 0)          # every onset on a beat slot


def test_pattern_offbeat_onsets():
    bt = np.arange(0.5, 8.5, 0.5)
    offbeats = bt[:-1] + 0.25              # exact 8th-note offbeats
    grid, _ = _pattern_from_beats_and_onsets(bt, offbeats)
    slots = np.where(grid == 1)[0]
    assert np.all(slots % 4 == 2)          # halfway between beat slots


def test_pattern_too_few_beats_returns_none():
    grid, num_bars = _pattern_from_beats_and_onsets(np.array([1.0]), np.array([1.0]))
    assert grid is None and num_bars is None


def test_pattern_with_beat_positions_bar_count():
    bt = np.arange(0.5, 8.5, 0.5)
    pos = np.array([(i % 4) + 1 for i in range(len(bt))])   # downbeats at 0,4,8,12
    grid, num_bars = _pattern_from_beats_and_onsets(bt, bt, beat_positions=pos)
    assert num_bars == 3                   # 4 downbeats -> 3 complete bars
    assert len(grid) == 3 * 16


def test_pattern_ignores_onsets_outside_grid():
    bt = np.arange(0.5, 8.5, 0.5)
    onsets = np.array([100.0, -5.0])       # far outside
    grid, _ = _pattern_from_beats_and_onsets(bt, onsets)
    assert grid.sum() == 0


# ── _score_pattern ──────────────────────────────────────────

def test_score_pattern_downbeats_only_scores_zero():
    # Onsets only on downbeats (weight 5) -> zero syncopation
    pattern = np.zeros(2 * 16, dtype=int)
    pattern[0] = pattern[16] = 1
    assert _score_pattern(pattern, 2) == 0


def test_score_pattern_weak_slots_score_high():
    # Onsets only on the weakest 16th slots (weight 1) -> max score 100
    pattern = np.zeros(1 * 16, dtype=int)
    pattern[[1, 3, 5, 7]] = 1
    assert _score_pattern(pattern, 1) == 100


def test_score_pattern_empty_bars_score_zero():
    assert _score_pattern(np.zeros(16, dtype=int), 1) == 0.0


def test_score_pattern_none_returns_nan():
    assert math.isnan(_score_pattern(None, None))


def test_score_pattern_short_pattern_returns_nan():
    assert math.isnan(_score_pattern(np.zeros(8, dtype=int), 2))


# ── _derive_positions_and_meter ─────────────────────────────

def test_derive_positions_aligned_4_4():
    bt = np.arange(0.0, 8.0, 0.5)
    db = bt[::4]                            # every 4th beat is a downbeat
    pos, meter = _derive_positions_and_meter(bt, db)
    assert meter == 4
    assert list(pos[:8]) == [1, 2, 3, 4, 1, 2, 3, 4]


def test_derive_positions_3_4():
    bt = np.arange(0.0, 6.0, 0.5)
    db = bt[::3]
    pos, meter = _derive_positions_and_meter(bt, db)
    assert meter == 3
    assert list(pos[:6]) == [1, 2, 3, 1, 2, 3]


def test_derive_positions_too_few_downbeats_assumes_4_4():
    bt = np.arange(0.0, 4.0, 0.5)
    pos, meter = _derive_positions_and_meter(bt, np.array([0.0]))
    assert meter == 4
    assert list(pos[:4]) == [1, 2, 3, 4]


def test_derive_positions_with_pickup_beats():
    # Regression (was critical #1): 2 pickup beats before the first downbeat
    # at t=1.0 must be back-counted (3, 4), not treated as spurious downbeats.
    bt = np.arange(0.0, 8.0, 0.5)
    db = np.array([1.0, 3.0, 5.0, 7.0])
    pos, meter = _derive_positions_and_meter(bt, db)
    assert meter == 4
    assert int((pos == 1).sum()) == len(db)     # only true downbeats are 1
    assert list(pos[:2]) == [3, 4]              # back-counted pickup


def test_derive_positions_full_bar_of_pickups_wraps():
    # With >= one full bar before the first detected downbeat, the earliest
    # pickup IS a legitimate downbeat of the previous bar (wrap-around).
    bt = np.arange(0.0, 8.0, 0.5)
    db = np.array([2.0, 4.0, 6.0])              # 4 pickup beats (one full bar)
    pos, meter = _derive_positions_and_meter(bt, db)
    assert meter == 4
    assert list(pos[:4]) == [1, 2, 3, 4]        # previous-bar downbeat + count-in


def test_syncopation_meter_score_unaffected_by_pickup():
    # End-to-end regression: derived positions must yield the same score as
    # ground-truth positions for the identical rhythm with a pickup.
    bt = np.arange(0.0, 10.0, 0.5)
    db = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    pos_derived, meter = _derive_positions_and_meter(bt, db)
    i0 = 2                                       # index of first downbeat (t=1.0)
    pos_true = np.array([((i - i0) % 4) + 1 for i in range(len(bt))])
    assert list(pos_derived) == list(pos_true)
    drums = make_clicks(bt, 10.0)
    s_derived, _ = syncopation_score(bt, drums, sr=SR, beats_per_bar=meter,
                                     beat_positions=pos_derived)
    s_true, _ = syncopation_score(bt, drums, sr=SR, beats_per_bar=meter,
                                  beat_positions=pos_true)
    assert s_derived == s_true


# ── onset_detection ─────────────────────────────────────────

def test_onset_detection_finds_clicks():
    times = np.arange(0.5, 9.5, 0.5)
    y = make_clicks(times, 10.0)
    detected = onset_detection(y, SR)
    assert abs(len(detected) - len(times)) <= 1
    # Every true click matched within 30 ms
    for t in times:
        assert np.min(np.abs(detected - t)) < 0.03


def test_onset_detection_silence_finds_nothing():
    assert len(onset_detection(np.zeros(SR * 3), SR)) == 0


# ── syncopation_score (full pipeline, no ML) ────────────────

def test_syncopation_score_on_beat_clicks():
    bt = np.arange(0.5, 9.5, 0.5)
    drums = make_clicks(bt, 10.0)
    score, onsets = syncopation_score(bt, drums, sr=SR)
    # All onsets on beats -> slots 0/4/8/12, weights [5,3,4,3] -> exactly 31
    assert score == 31
    assert len(onsets) > 0


def test_syncopation_score_uses_provided_onsets():
    bt = np.arange(0.5, 9.5, 0.5)
    score, onsets = syncopation_score(bt, None, sr=SR, onset_times=bt)
    assert score == 31
    assert onsets is bt                    # passthrough, no detection


def test_syncopation_score_resamples_drums():
    # Drums at 22050 Hz must be resampled to the song SR without error
    bt = np.arange(0.5, 9.5, 0.5)
    drums_22k = make_clicks(bt, 10.0, sr=22050)
    score, _ = syncopation_score(bt, drums_22k, sr=SR, drums_sr=22050)
    assert 0 <= score <= 100


def test_syncopation_score_with_beat_positions():
    bt = np.arange(0.5, 9.5, 0.5)
    pos = np.array([(i % 4) + 1 for i in range(len(bt))])
    drums = make_clicks(bt, 10.0)
    score, _ = syncopation_score(bt, drums, sr=SR, beat_positions=pos)
    assert 0 <= score <= 100


# ── detect_beats (model-based — slow) ───────────────────────

@pytest.mark.skipif(not RUN_SLOW, reason="set REWARDIO_RUN_SLOW=1 to run model-based beat detection")
def test_detect_beats_beat_this_on_clicks():
    # Default beat-detection path (BEAT THIS!) — downloads its checkpoint on
    # first ever run (~80 MB), cached afterwards.
    from rewardio.rhythm import detect_beats
    times = np.arange(0.5, 9.5, 0.5)
    y = make_clicks(times, 10.0)
    beat_times, beat_frames, positions, meter = detect_beats(y, SR)
    assert len(beat_times) > 10
    assert meter in (2, 3, 4)
    assert len(positions) == len(beat_times)
    bpm = get_bpm(beat_times)
    assert 110 <= bpm <= 130 or 55 <= bpm <= 65   # 120 or half-tempo octave


@pytest.mark.skipif(not RUN_SLOW, reason="set REWARDIO_RUN_SLOW=1 to run model-based beat detection")
def test_detect_beats_madmom_on_clicks():
    from rewardio.rhythm import detect_beats
    times = np.arange(0.5, 9.5, 0.5)
    y = make_clicks(times, 10.0)
    beat_times, beat_frames, positions, meter = detect_beats(y, SR, use_madmom=True)
    assert len(beat_times) > 0
    assert meter in (2, 3, 4)
    assert len(positions) == len(beat_times)
    bpm = get_bpm(beat_times)
    assert 110 <= bpm <= 130 or 55 <= bpm <= 65   # 120 or half-tempo octave
