import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
import numpy as np
import librosa
from datetime import datetime

from .plot import plot_beats as _plot_beats, plot_waveform as _plot_waveform, plot_beats_and_onsets as _plot_beats_and_onsets, plot_interactive as _plot_interactive, plot_session_boxplots as _plot_session_boxplots, plot_spectrogram as _plot_spectrogram
from .rhythm import detect_beats, get_bpm, onset_detection, syncopation_score
from .core import load_audio, write_to_csv, stimulus_help as _stimulus_help, stimulus_print as _stimulus_print, stimulus_print_all as _stimulus_print_all, session_help as _session_help, session_print as _session_print, participant_help as _participant_help, participant_print as _participant_print, clear
from .dsp import normalize, get_loudness, get_rms, compute_fluctuation as _compute_fluctuation, spectral_irregularity as _spectral_irregularity, compute_spectral_features as _compute_spectral_features
from .play import play_audio, play_interactive as _play_interactive, sonify_beats as _sonify_beats, sonify_beats_and_onsets as _sonify_beats_and_onsets
from .separate import separate
from .genre import (classify_genre as _classify_genre,
                   classify_voice_instrumental as _classify_voice_instrumental,
                   classify_mood as _classify_mood,
                   classify_all as _classify_all,
                   detect_pitch_crepe as _detect_pitch_crepe,
                   detect_key as _detect_key)

from scipy import signal as sp_signal
from .dsp import normalize as _normalize



#  Stimulus class

class Stimulus:
    """
    Wraps a single audio file and provides analysis attributes and methods.

    Usage
    -----
        s = Stimulus("path/to/audio.wav")
        print(s)          # shows filename, BPM, duration, channels
        s.play()          # listen in a notebook
        s.plot_beats()    # waveform + beat markers
        s.separate()      # run Demucs source separation
    """

    def __init__(self, audio_file_path, sr=44100):
        # Audio (lazy load)
        self.audio_file_path = audio_file_path
        self.audio_file_name = os.path.basename(audio_file_path)
        # Fail fast on missing/unreadable files so Session can skip them at
        # load time instead of crashing mid-batch on the first lazy access.
        if not os.path.isfile(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        librosa.get_samplerate(audio_file_path)  # cheap header probe; raises if unreadable
        self.sr = sr
        self._y = None
        self._n_channels = None
        self._duration = None

        # Beat detection (lazy)
        self._beat_times = None
        self._beat_frames = None
        self._beat_positions = None
        self._meter = None
        self._bpm = None

        # Source separation (lazy)
        self.separated_drums = None
        self.separated_drums_sr = None
        self.separated_accompaniment = None

        # Loudness (lazy load)
        self._loudness_lufs = None
        self._loudness_rms_db = None
        self._rms = None

        # Analysis results (lazy)
        self.onset_times = None
        self.toussaint_syncopation_score = None
        self.toussaint_syncopation_score_meter = None

        # Essentia classification (lazy)
        self._genre = None
        self._mood = None
        self._voice_instrumental = None
        self._pitch_time = None
        self._pitch_freq = None
        self._pitch_conf = None
        self._key = None
        self._scale = None
        self._key_strength = None

        # Extra Lazy Features
        self._genre_predictions = None
        self._fluctuation = None
        self._irregularity = None
        self._spectral_features = None  # dict of per-frame arrays

    # Audio Properties
    def _load_audio_if_needed(self):
        if self._y is None:
            sr = getattr(self, 'sr', 44100)
            self._y, self.sr = load_audio(self.audio_file_path, sr=sr)
            if self._y.ndim == 1:
                self._n_channels = 1
                self._duration = len(self._y) / self.sr
            else:
                self._n_channels = self._y.shape[0]
                self._duration = self._y.shape[1] / self.sr

    @property
    def y(self):
        self._load_audio_if_needed()
        return self._y

    @property
    def n_channels(self):
        self._load_audio_if_needed()
        return self._n_channels

    @property
    def duration(self):
        self._load_audio_if_needed()
        return self._duration


    @property
    def beat_times(self):
        if self._beat_times is None:
            y_mono = normalize(self._to_mono())
            self._beat_times, self._beat_frames, self._beat_positions, self._meter = detect_beats(y_mono, self.sr)
            self._bpm = get_bpm(self._beat_times)
        return self._beat_times

    # Spectral Properties
    @property
    def fluctuation(self):
        if self._fluctuation is None:
            y_mono = self._to_mono()
            self._fluctuation, _, _ = _compute_fluctuation(y_mono, self.sr)
        return self._fluctuation

    @property
    def irregularity(self):
        if self._irregularity is None:
            y_mono = self._to_mono()
            self._irregularity = _spectral_irregularity(y_mono, self.sr)
        return self._irregularity
    
    @property
    def beat_positions(self):
        """Beat positions (1=downbeat, 2, 3, 4…) from beat detection."""
        if self._beat_positions is None:
            _ = self.beat_times  # triggers detect_beats
        return self._beat_positions
    
    @property
    def meter(self):
        """Detected meter (2, 3, or 4) from beat detection."""
        if self._meter is None:
            _ = self.beat_times  # triggers detect_beats
        return self._meter

    @property
    def beat_frames(self):
        if self._beat_frames is None:
             _ = self.beat_times
        return self._beat_frames

    @property
    def bpm(self):
        if self._bpm is None:
             self._bpm = get_bpm(self.beat_times)
        return self._bpm

    @property
    def BPM(self):
        """Alias for .bpm"""
        return self.bpm

    # Loudness Properties
    def _calc_loudness_if_needed(self):
        if self._loudness_lufs is None:
            self._loudness_lufs, self._loudness_rms_db = get_loudness(self.y, self.sr)
            self._rms = get_rms(self.y, self.sr)

    @property
    def loudness_lufs(self):
        self._calc_loudness_if_needed()
        return self._loudness_lufs

    @property
    def loudness_rms_db(self):
        self._calc_loudness_if_needed()
        return self._loudness_rms_db

    @property
    def rms(self):
        self._calc_loudness_if_needed()
        return self._rms

    # Private helpers
    def _to_mono(self):
        """Return a mono mix of the audio (for analysis only)."""
        if self.y.ndim == 1:
            return self.y
        return librosa.to_mono(self.y)

    # Public methods

    def separate(self, target="drums", gpu=None, confirm=True):
        """
        Run Demucs source separation.

        Parameters
        ----------
        target : str
            Target source to isolate (default: 'drums').
        gpu : torch.device or None
            GPU device. If None, uses CPU.
        confirm : bool
            If True, ask user before proceeding (default: True).
        """
        if confirm and self.separated_drums is None:
            ans = input(
                "!!! This requires Demucs source separation (~5-10s per song depending on GPU).\n"
                "Proceed? [Y/n] "
            ).strip().lower()
            if ans in ('n', 'no'):
                print("Skipped.")
                return

        target_audio, accompaniment, drums_sr = separate(
            self.audio_file_path, target_source=target, gpu=gpu
        )
        self.separated_drums = target_audio
        self.separated_drums_sr = drums_sr
        self.separated_accompaniment = accompaniment
        
        # Auto-calculate syncopation score since we have drums
        # (This will trigger onset detection if not already done)
        self.syncopation_score()

    def onset_detection(self, y=None, hop_length=512, delta=0.25):
        """
        Detect onsets using Path 2 logic (no HPSS, good for drums).
        Delegates to rhythm.onset_detection.
        
        Args:
            y: Audio time series (monophonic). If None, uses self.separated_drums.
            hop_length: Hop length.
            delta: Detection threshold (default 0.25).
            
        Returns:
            onset_times: Array of onset times in seconds.
        """
        if y is None:
            if self.separated_drums is None:
                self.separate()

            drums_y = self.separated_drums
            if drums_y.ndim == 2:
                drums_y = np.mean(drums_y, axis=0)  # Mix to mono

            # Resample to match song SR if Demucs returned a different rate (e.g. 48000)
            if self.separated_drums_sr is not None and self.separated_drums_sr != self.sr:
                drums_y = librosa.resample(drums_y, orig_sr=self.separated_drums_sr, target_sr=self.sr)

            # Low-pass filter (same as syncopation_score pipeline)
            
            nyquist = 0.5 * self.sr
            cutoff_hz = 1500
            b_low, a_low = sp_signal.butter(4, cutoff_hz / nyquist, btype="low")
            drums_y = _normalize(sp_signal.filtfilt(b_low, a_low, drums_y))

            y = drums_y

        self.onset_times = onset_detection(y, self.sr, hop_length, delta)
        return self.onset_times

    def detect_onsets(self, y=None, hop_length=512, delta=0.25):
        """
        Alias for onset_detection.
        Detect onsets using Path 2 logic.
        """
        onsets = self.onset_detection(y=y, hop_length=hop_length, delta=delta)
        # Auto-calculate syncopation score since we have onsets
        self.syncopation_score()
        print(f"Detected {len(onsets)} onsets.")

    def syncopation_score(self, meter=False):
        """
        Calculate syncopation score using Path 2 logic (on separated drums).
        
        Args:
            meter: If True, uses detected meter and beat positions for
                   proper bar alignment.
                   If False (default), assumes 4/4 with no bar alignment.
        """
        if self.separated_drums is None:
            self.separate()

        if meter:
            # Meter-aware: use beat_positions for proper bar alignment
            self.toussaint_syncopation_score_meter, detected_onsets = syncopation_score(
                self.beat_times,
                self.separated_drums, 
                self.sr, 
                beats_per_bar=self.meter,
                onset_times=self.onset_times,
                drums_sr=self.separated_drums_sr,
                beat_positions=self.beat_positions
            )
        else:
            # Default: plain beat detection, assumes 4/4
            self.toussaint_syncopation_score, detected_onsets = syncopation_score(
                self.beat_times, 
                self.separated_drums, 
                self.sr, 
                beats_per_bar=4,
                onset_times=self.onset_times,
                drums_sr=self.separated_drums_sr
            )
        
        self.onset_times = detected_onsets
        return self.toussaint_syncopation_score_meter if meter else self.toussaint_syncopation_score

    @property
    def genre(self):
        """Top predicted genre (e.g. 'Electronic---House')."""
        if self._genre_predictions is None:
            self.classify_genre()
        return self._genre_predictions[0][0] if self._genre_predictions else None

    @property
    def genre_top5(self):
        """Top 5 genre predictions as [(label, confidence), ...]."""
        if self._genre_predictions is None:
            self.classify_genre()
        return self._genre_predictions

    def classify(self):
        """
        Run ALL Essentia classifiers at once (genre,
        voice/instrumental, mood) on shared embeddings.
        Also runs CREPE pitch detection and key/scale detection.
        """
        print(f"Classifying {self.audio_file_name}...\n")
        results = _classify_all(self.audio_file_path)

        # Store results
        self._genre_predictions = results["genre"]
        self._voice_instrumental = results["voice_instrumental"]
        self._mood = results["mood"]

        # Pitch (separate model)
        print("  Running CREPE pitch detection...")
        self._pitch_time, self._pitch_freq, self._pitch_conf = _detect_pitch_crepe(
            self.audio_file_path
        )

        # Key/scale detection
        print("  Detecting key/scale...")
        self._key, self._scale, self._key_strength = _detect_key(
            self.audio_file_path
        )

        # ── Print summary ──
        print(f"\n  {'Attribute':<20s} {'Result':>s}")
        print(f"  {'─' * 20} {'─' * 35}")

        # Genre
        top_genre = self._genre_predictions[0]
        print(f"  {'Genre':<20s} {top_genre[0]}  ({top_genre[1]*100:.1f}%)")

        # Voice/Instrumental
        top_vi = self._voice_instrumental[0]
        print(f"  {'Vocal':<20s} {top_vi[0]}")

        # Key
        print(f"  {'Key':<20s} {self._key} {self._scale}  ({self._key_strength*100:.0f}%)")

        # Mood — just top mood
        top_mood = max(self._mood, key=self._mood.get)
        print(f"  {'Mood':<20s} {top_mood} ({self._mood[top_mood]*100:.0f}%)")

        # Pitch
        voiced = self._pitch_freq[self._pitch_conf > 0.5]
        if len(voiced) > 0:
            print(f"  {'Pitch':<20s} {float(np.median(voiced)):.1f} Hz")
        else:
            print(f"  {'Pitch':<20s} (no voiced frames)")
        print()

    # Voice / Instrumental
    @property
    def voice_instrumental(self):
        """'voice' or 'instrumental'."""
        if self._voice_instrumental is None:
            self.classify()
        return self._voice_instrumental[0][0] if self._voice_instrumental else None

    # Mood
    @property
    def mood(self):
        """Mood dict: {'happy': 0.8, 'sad': 0.1, 'aggressive': 0.2, 'relaxed': 0.6}."""
        if self._mood is None:
            self.classify()
        return self._mood

    # Key / Scale
    @property
    def key(self):
        """Musical key, e.g. 'C', 'F#', 'Bb'."""
        if self._key is None:
            self.classify()
        return self._key

    @property
    def scale(self):
        """'major' or 'minor'."""
        if self._scale is None:
            self.classify()
        return self._scale

    # Pitch (CREPE)
    def _voiced_pitch(self):
        """Return voiced pitch frequencies (confidence > 0.5)."""
        if self._pitch_freq is None:
            self.classify()
        return self._pitch_freq[self._pitch_conf > 0.5]

    @property
    def pitch(self):
        """Median pitch frequency in Hz (CREPE, voiced frames only)."""
        voiced = self._voiced_pitch()
        return float(np.median(voiced)) if len(voiced) > 0 else 0.0

    @property
    def pitch_mean(self):
        """Mean pitch frequency in Hz (CREPE, voiced frames only)."""
        voiced = self._voiced_pitch()
        return float(np.mean(voiced)) if len(voiced) > 0 else 0.0

    @property
    def pitch_std(self):
        """Std deviation of pitch frequency in Hz (CREPE, voiced frames only)."""
        voiced = self._voiced_pitch()
        return float(np.std(voiced)) if len(voiced) > 0 else 0.0

    @property
    def pitch_conf_mean(self):
        """Mean CREPE pitch confidence [0, 1]."""
        if self._pitch_conf is None:
            self.classify()
        return float(np.mean(self._pitch_conf))

    @property
    def pitch_conf_std(self):
        """Std deviation of CREPE pitch confidence."""
        if self._pitch_conf is None:
            self.classify()
        return float(np.std(self._pitch_conf))

    # Beat IOI
    @property
    def beat_ioi_mean(self):
        """Mean inter-beat interval in seconds."""
        bt = self.beat_times
        if len(bt) < 2:
            return 0.0
        return float(np.mean(np.diff(bt)))

    @property
    def beat_ioi_std(self):
        """Std deviation of inter-beat intervals in seconds."""
        bt = self.beat_times
        if len(bt) < 2:
            return 0.0
        return float(np.std(np.diff(bt)))

    # Onset IOI
    @property
    def onset_ioi_mean(self):
        """Mean inter-onset interval in seconds."""
        if self.onset_times is None or len(self.onset_times) < 2:
            return 0.0
        return float(np.mean(np.diff(self.onset_times)))

    @property
    def onset_ioi_std(self):
        """Std deviation of inter-onset intervals in seconds."""
        if self.onset_times is None or len(self.onset_times) < 2:
            return 0.0
        return float(np.std(np.diff(self.onset_times)))

    # Spectral Features
    def _compute_spectral_if_needed(self):
        """Compute spectral features (once, cached)."""
        if self._spectral_features is None:
            y_mono = self._to_mono()
            self._spectral_features = _compute_spectral_features(y_mono, self.sr)
        return self._spectral_features

    @property
    def spectral_centroid_mean(self):
        """Mean spectral centroid in Hz."""
        return float(np.mean(self._compute_spectral_if_needed()["spectral_centroid"]))

    @property
    def spectral_centroid_std(self):
        """Std deviation of spectral centroid."""
        return float(np.std(self._compute_spectral_if_needed()["spectral_centroid"]))

    @property
    def spectral_bandwidth_mean(self):
        """Mean spectral bandwidth in Hz."""
        return float(np.mean(self._compute_spectral_if_needed()["spectral_bandwidth"]))

    @property
    def spectral_bandwidth_std(self):
        """Std deviation of spectral bandwidth."""
        return float(np.std(self._compute_spectral_if_needed()["spectral_bandwidth"]))

    @property
    def spectral_rolloff_mean(self):
        """Mean spectral rolloff in Hz."""
        return float(np.mean(self._compute_spectral_if_needed()["spectral_rolloff"]))

    @property
    def spectral_rolloff_std(self):
        """Std deviation of spectral rolloff."""
        return float(np.std(self._compute_spectral_if_needed()["spectral_rolloff"]))

    @property
    def spectral_flatness_mean(self):
        """Mean spectral flatness [0=tonal, 1=noisy]."""
        return float(np.mean(self._compute_spectral_if_needed()["spectral_flatness"]))

    @property
    def spectral_flatness_std(self):
        """Std deviation of spectral flatness."""
        return float(np.std(self._compute_spectral_if_needed()["spectral_flatness"]))

    @property
    def zcr_mean(self):
        """Mean zero-crossing rate."""
        return float(np.mean(self._compute_spectral_if_needed()["zcr"]))

    @property
    def zcr_std(self):
        """Std deviation of zero-crossing rate."""
        return float(np.std(self._compute_spectral_if_needed()["zcr"]))

    # Fluctuation
    # Computed eagerly at init — access directly: stimulus.fluctuation

    def _collect_attrs(self):
        """Gather currently-computed attributes into a dict for CSV export."""
        row = {
            "filename": self.audio_file_name,
            "duration": self.duration,
            "sr": self.sr,
            "n_channels": self.n_channels,
            "loudness_lufs": self.loudness_lufs,
            "loudness_rms_db": self.loudness_rms_db,
            "rms": self.rms,
        }
        if self._bpm is not None:
            row["bpm"] = self._bpm
        if self._beat_times is not None:
            row["n_beats"] = len(self._beat_times)
        if self.onset_times is not None:
            row["n_onsets"] = len(self.onset_times)
        if self.toussaint_syncopation_score is not None:
            row["syncopation_score"] = self.toussaint_syncopation_score
        if self.toussaint_syncopation_score_meter is not None:
            row["syncopation_score_meter"] = self.toussaint_syncopation_score_meter
        if self._meter is not None:
            row["meter"] = self._meter
        if self._genre_predictions is not None:
            row["genre"] = self._genre_predictions[0][0]
            row["genre_confidence"] = float(self._genre_predictions[0][1])
        if self._voice_instrumental is not None:
            row["voice_instrumental"] = self._voice_instrumental[0][0]
        if self._mood is not None:
            for m, v in self._mood.items():
                row[f"mood_{m}"] = float(v)
        if self._pitch_freq is not None:
            voiced = self._pitch_freq[self._pitch_conf > 0.5]
            if len(voiced) > 0:
                row["pitch_median_hz"] = float(np.median(voiced))
                row["pitch_mean_hz"] = float(np.mean(voiced))
                row["pitch_std_hz"] = float(np.std(voiced))
            row["pitch_conf_mean"] = float(np.mean(self._pitch_conf))
            row["pitch_conf_std"] = float(np.std(self._pitch_conf))
        if self._beat_times is not None and len(self._beat_times) >= 2:
            ioi = np.diff(self._beat_times)
            row["beat_ioi_mean"] = float(np.mean(ioi))
            row["beat_ioi_std"] = float(np.std(ioi))
        if self.onset_times is not None and len(self.onset_times) >= 2:
            oioi = np.diff(self.onset_times)
            row["onset_ioi_mean"] = float(np.mean(oioi))
            row["onset_ioi_std"] = float(np.std(oioi))
        if self._key is not None:
            row["key"] = self._key
            row["scale"] = self._scale
            row["key_strength"] = float(self._key_strength)
        row["fluctuation"] = self.fluctuation
        row["spectral_irregularity"] = self.irregularity
        if self._spectral_features is not None:
            sf = self._spectral_features
            row["spectral_centroid_mean"] = float(np.mean(sf["spectral_centroid"]))
            row["spectral_centroid_std"] = float(np.std(sf["spectral_centroid"]))
            row["spectral_bandwidth_mean"] = float(np.mean(sf["spectral_bandwidth"]))
            row["spectral_bandwidth_std"] = float(np.std(sf["spectral_bandwidth"]))
            row["spectral_rolloff_mean"] = float(np.mean(sf["spectral_rolloff"]))
            row["spectral_rolloff_std"] = float(np.std(sf["spectral_rolloff"]))
            row["spectral_flatness_mean"] = float(np.mean(sf["spectral_flatness"]))
            row["spectral_flatness_std"] = float(np.std(sf["spectral_flatness"]))
            row["zcr_mean"] = float(np.mean(sf["zcr"]))
            row["zcr_std"] = float(np.std(sf["zcr"]))
        return row

    def save(self, output_path=None):
        """
        Save currently-calculated attributes to a CSV file.

        Parameters
        ----------
        output_path : str or None
            Folder to write the Analysis directory in. Defaults to scripts dir.
        """
        row = self._collect_attrs()
        write_to_csv([row], output_path=output_path)

    def save_timeseries(self, output_path=None):
        """
        Save all computed time series as a .npz archive.

        The file is saved alongside CSVs in the Analysis folder.
        Load with: data = np.load("file.npz"); data["pitch_freq"]

        Parameters
        ----------
        output_path : str or None
            Parent directory for the Analysis folder.
        """
        if output_path is None:
            output_path = os.path.dirname(os.path.abspath(__file__))

        now = datetime.now()
        folder_name = now.strftime("Analysis_%d-%m-%Y")
        folder_path = os.path.join(output_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        base = os.path.splitext(self.audio_file_name)[0]
        npz_path = os.path.join(folder_path, f"{base}_timeseries.npz")

        arrays = {}
        if self._pitch_time is not None:
            arrays["pitch_time"] = self._pitch_time
            arrays["pitch_freq"] = self._pitch_freq
            arrays["pitch_conf"] = self._pitch_conf
        if self._beat_times is not None:
            arrays["beat_times"] = self._beat_times
        if self.onset_times is not None:
            arrays["onset_times"] = self.onset_times
        if self._spectral_features is not None:
            for k, v in self._spectral_features.items():
                arrays[k] = v

        if arrays:
            np.savez_compressed(npz_path, **arrays)
            print(f"Time series saved → {npz_path}")
        else:
            print("No time series data to save.")

    def process_and_save(self, output_path=None, timeseries=False):
        """
        Compute ALL metrics (BPM, separation, syncopation) then save to CSV.

        Parameters
        ----------
        output_path : str or None
            Folder to write the Analysis directory in. Defaults to scripts dir.
        timeseries : bool
            If True, also save all time series as a .npz file.
        """
        print(f"Processing {self.audio_file_name}...")
        # Trigger beat detection + BPM
        _ = self.bpm
        # Trigger separation (no confirmation — batch mode)
        if self.separated_drums is None:
            self.separate(confirm=False)
        # Trigger both syncopation scores
        self.syncopation_score()             # default (plain beats, 4/4)
        self.syncopation_score(meter=True)   # meter-aware (downbeat detection)
        self.classify()
        # Trigger spectral features
        self._compute_spectral_if_needed()
        print("Processing complete.")
        self.save(output_path=output_path)
        if timeseries:
            self.save_timeseries(output_path=output_path)

    def play(self, x=None, y=None):
        """
        Open the unified interactive player.
        Includes waveform with cursor, play/pause/stop controls,
        and toggle buttons for beats & onsets (visual + sonification).

        Args:
            x: xlim (time range in seconds), e.g. (5, 15).
            y: ylim (amplitude range).
        """
        _play_interactive(self, xlim=x, ylim=y)

    # ── Legacy playback wrappers ──────────
    def sonify_beats(self, x=None, y=None):
        """Sonify beats (legacy - use .play() instead)."""
        _sonify_beats(self.y, self.sr, self.beat_times,
                      title=f"Beats - {self.audio_file_name}",
                      xlim=x, ylim=y)

    def sonify_onsets(self, x=None, y=None):
        """Sonify beats + onsets (legacy - use .play() instead)."""
        if self.onset_times is None:
            self.detect_onsets()

        _sonify_beats_and_onsets(self.y, self.sr, self.beat_times, self.onset_times,
                                 title=f"Beats + Onsets - {self.audio_file_name}",
                                 xlim=x, ylim=y)
 
    def plot(self, x=None, y=None, scale='mel'):
        """
        Open a spectrogram window.

        Args:
            x: xlim (time range in seconds), e.g. (5, 15).
            y: ylim (frequency range in Hz).
            scale: 'mel' (default), 'linear', or 'log'.
        """
        _plot_spectrogram(self, xlim=x, ylim=y, scale=scale)

    # ── Legacy plot wrappers ──────────────
    def plot_beats(self, x=None, y=None):
        """Plot waveform with detected beat markers."""
        _plot_beats(self, xlim=x, ylim=y)

    def plot_waveform(self, x=None, y=None):
        """Plot the raw waveform."""
        _plot_waveform(self, xlim=x, ylim=y)

    def plot_onsets(self, x=None, y=None):
        """Plot waveform with beat markers AND detected onsets."""
        if self.onset_times is None:
            self.detect_onsets()
        _plot_beats_and_onsets(self, xlim=x, ylim=y)

    def help(self):
        """Print all available attributes and methods."""
        _stimulus_help(self)

    def print(self):
        """Print a concise summary of key attributes."""
        _stimulus_print(self)

    def print_all(self):
        """Print the current value of every attribute on this Stimulus."""
        _stimulus_print_all(self)

    def __repr__(self):
        ch_str = "mono" if self.n_channels == 1 else f"{self.n_channels}ch stereo"
        bpm_str = f"{self._bpm}" if self._bpm is not None else "(...)"
        beats_str = f"{len(self._beat_times)} detected" if self._beat_times is not None else "(...)"
        return (
            f"Stimulus(\"{self.audio_file_name}\")\n"
            f"  BPM      : {bpm_str}\n"
            f"  Loudness : {self.loudness_lufs:.2f} LUFS  ({self.loudness_rms_db:.2f} dBFS RMS)\n"
            f"  RMS      : {self.rms:.6f}\n"
            f"  Duration : {self.duration:.2f}s\n"
            f"  SR       : {self.sr} Hz\n"
            f"  Channels : {ch_str}\n"
            f"  Beats    : {beats_str}"
        )


# ──────────────────────────────────────────────
#  Session class (Folder)
# ──────────────────────────────────────────────

class Session:
    """
    Wraps a folder of audio files, creating a Stimulus for each.
    """
    def __init__(self, folder_path, sr=44100):
        self.folder_path = folder_path
        self.items = []
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")
            
        # Iterate over files in the directory
        for fname in sorted(os.listdir(folder_path)):
             ext = os.path.splitext(fname)[1].lower()
             if ext in ('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a'):
                 fpath = os.path.join(folder_path, fname)
                 try:
                     s = Stimulus(fpath, sr=sr)
                     self.items.append(s)
                 except Exception as e:
                     reason = str(e) or type(e).__name__
                     print(f"  [Skipped] {fname}: unreadable audio ({reason})")
        


    def _process_all_items(self, do_separate=None):
        """
        Trigger BPM, separation, and syncopation for all items.
        """
        print(f"Processing {len(self.items)} items...")

        # Check how many items need drum separation
        need_sep = [s for s in self.items if s.separated_drums is None]
        if do_separate is None:
            do_separate = True
            if need_sep:
                ans = input(
                    f"⚠️  {len(need_sep)} song(s) need Demucs drum separation for syncopation scoring.\n"
                    f"    This takes up to 1 minute per song depending on your GPU.\n"
                    f"    Proceed? [Y/n] "
                ).strip().lower()
                if ans in ('n', 'no'):
                    do_separate = False
                    print("Skipping drum separation — syncopation scores will not be calculated.")

        for s in self.items:
            # Trigger BPM
            _ = s.bpm
            
            # Syncopation (only if separation is allowed)
            if do_separate:
                try:
                    if s.toussaint_syncopation_score is None:
                        if s.separated_drums is None:
                            s.separate(confirm=False)
                        s.syncopation_score()
                    # Meter-aware score too, so session/participant CSVs get the
                    # syncopation_score_meter column (parity with Stimulus.process_and_save)
                    if s.toussaint_syncopation_score_meter is None and s.separated_drums is not None:
                        s.syncopation_score(meter=True)
                except Exception as e:
                    print(f"  [Skipped syncopation] {s.audio_file_name}: {e}")

    def save(self, output_path=None):
        """
        Save currently-calculated attributes of every item to a CSV file.

        Parameters
        ----------
        output_path : str or None
            Folder to write the Analysis directory in. Defaults to scripts dir.
        """
        rows = [s._collect_attrs() for s in self.items]
        write_to_csv(rows, output_path=output_path)

    def process_and_save(self, index=None, output_path=None, timeseries=False):
        """
        Compute ALL metrics, then save to CSV.

        Parameters
        ----------
        index : int or None
            If given (1-based), process only that stimulus.
            If None, process all items.
        output_path : str or None
            Folder to write the Analysis directory in. Defaults to scripts dir.
        timeseries : bool
            If True, also save time series as .npz files.
        """
        if index is not None:
            item = self[index]  # 1-based via __getitem__
            print(f"Processing stimulus {index}: {item.audio_file_name}...")
            # Trigger beat detection → BPM
            _ = item.beat_times
            # Trigger separation + syncopation (skip per-song confirmation)
            if item.separated_drums is None:
                item.separate(confirm=False)
            item.syncopation_score()
            item.syncopation_score(meter=True)
            item.classify()
            item._compute_spectral_if_needed()
            print("Processing complete.")
            rows = [item._collect_attrs()]
            write_to_csv(rows, output_path=output_path)
            if timeseries:
                item.save_timeseries(output_path=output_path)
        else:
            print(f"Processing {len(self.items)} items...")
            self._process_all_items()
            for s in self.items:
                s.classify()
                s._compute_spectral_if_needed()
            print("Processing complete.")
            self.save(output_path=output_path)
            if timeseries:
                for s in self.items:
                    s.save_timeseries(output_path=output_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if not isinstance(index, (int, str)):
            raise TypeError(f"Index must be an integer or string, got {type(index).__name__}")
        if isinstance(index, int):
            if len(self.items) == 0:
                raise ValueError("This session has no stimuli to select.")
            if index < 1 or index > len(self.items):
                print(
                    f"⚠️  Index {index} is out of range. "
                    f"Indexing starts from 1 to {len(self.items)}. "
                    f"Falling back to session(1)."
                )
                index = 1
            return self.items[index - 1]  # 1-based indexing
        # String lookup (case-insensitive partial match)
        matches = [s for s in self.items if index.lower() in s.audio_file_name.lower()]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            names = [s.audio_file_name for s in matches[:5]]
            msg = f"Ambiguous key '{index}'. Matches: {names}"
            if len(matches) > 5: msg += "..."
            raise ValueError(msg)
        else:
            raise KeyError(f"No stimulus found matching '{index}'")

    def __call__(self, index=None):
        """
        Select a specific Stimulus by index (1-based) or name.
        Automatically updates the 'stimulus' variable in the interactive shell.

        Usage
        -----
            session(1)          # focus on first item
            session("song")     # focus by partial name match
            session()           # list all items
        """
        if index is None:
            self.print()
            return self
        if isinstance(index, int):
            item = self[index]   # __getitem__ handles 1-based
        else:
            item = self[index]
        # Auto-update the interactive shell's 'stimulus' variable
        if hasattr(self, '_shell_ns'):
            self._shell_ns['stimulus'] = item
        item.print()

    def __repr__(self):
        return f"Session(path='{os.path.basename(self.folder_path)}', count={len(self.items)})"

    def help(self):
        """Print available attributes and methods for Session class."""
        _session_help(self)

    def print(self):
        """Print summary of the Session collection."""
        _session_print(self)

    def boxplot(self):
        """
        Plot boxplots for BPM, LUFS, and Syncopation Score distribution.
        Shows min, max, median, quartiles, and mean.
        Example:
            session.boxplot()
        """
        _plot_session_boxplots(self)


# ──────────────────────────────────────────────
#  Participant class (Folder of Session folders)
# ──────────────────────────────────────────────

class Participant:
    """
    Wraps a folder of sub-folders, creating a Session for each sub-folder.

    Hierarchy:  Participant → Session (sessions) → Stimulus (songs)
    """
    def __init__(self, folder_path, sr=44100):
        self.folder_path = folder_path
        self.sessions = []

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        print(f"Loading participant from: {folder_path}...")

        for name in sorted(os.listdir(folder_path)):
            sub_path = os.path.join(folder_path, name)
            if os.path.isdir(sub_path):
                try:
                    sess = Session(sub_path, sr=sr)
                    self.sessions.append(sess)
                except Exception as e:
                    print(f"  [Skipped] {name}: {e}")

        self.n_sessions = len(self.sessions)
        print(f"Loaded {self.n_sessions} sessions.")

    def save(self, output_path=None):
        """
        Save current attributes of every song in every session to a single CSV.
        """
        rows = []
        for i, session in enumerate(self.sessions):
            session_name = os.path.basename(session.folder_path)
            for s in session.items:
                row = s._collect_attrs()
                row["session"] = session_name
                rows.append(row)
            if i < self.n_sessions - 1:
                rows.append({})  # Blank row between sessions
        return write_to_csv(rows, output_path=output_path)

    def process_and_save(self, index=None, output_path=None, timeseries=False):
        """
        Compute ALL metrics, then save to CSV.

        Parameters
        ----------
        index : int or None
            If given (1-based), process only that session.
            If None, process all sessions.
        output_path : str or None
            Folder to write the Analysis directory in.
        timeseries : bool
            If True, also save time series as .npz files.
        """
        if index is not None:
            session = self[index]  # 1-based via __getitem__
            session_name = os.path.basename(session.folder_path)
            print(f"Processing session {index}: {session_name}...")
            session._process_all_items()
            for s in session.items:
                s.classify()
                s._compute_spectral_if_needed()
            print("Processing complete.")
            rows = []
            for s in session.items:
                row = s._collect_attrs()
                row["session"] = session_name
                rows.append(row)
            if timeseries:
                for s in session.items:
                    s.save_timeseries(output_path=output_path)
            return write_to_csv(rows, output_path=output_path)
        else:
            print(f"Processing {self.n_sessions} sessions...")
            
            # Global prompt for demucs separation across all sessions
            all_need_sep = []
            for session in self.sessions:
                all_need_sep.extend([s for s in session.items if s.separated_drums is None])
                
            do_separate = True
            if all_need_sep:
                ans = input(
                    f"\n⚠️  {len(all_need_sep)} song(s) across all sessions need Demucs drum separation for syncopation scoring.\n"
                    f"    This takes ~5-10s per song depending on your GPU.\n"
                    f"    Proceed? [Y/n] "
                ).strip().lower()
                if ans in ('n', 'no'):
                    do_separate = False
                    print("Skipping drum separation — syncopation scores will not be calculated.\n")

            for i, session in enumerate(self.sessions):
                session_name = os.path.basename(session.folder_path)
                print(f"\n── Session {i+1}/{self.n_sessions}: {session_name} ──")
                session._process_all_items(do_separate=do_separate)
                for s in session.items:
                    s.classify()
                    s._compute_spectral_if_needed()
            print("\nProcessing complete.")
            result = self.save(output_path=output_path)
            if timeseries:
                for session in self.sessions:
                    for s in session.items:
                        s.save_timeseries(output_path=output_path)
            return result

    def __len__(self):
        return self.n_sessions

    def __getitem__(self, index):
        if not isinstance(index, (int, str)):
            raise TypeError(f"Index must be an integer or string, got {type(index).__name__}")
        if isinstance(index, int):
            if self.n_sessions == 0:
                raise ValueError("This participant has no sessions to select.")
            if index < 1 or index > self.n_sessions:
                print(
                    f"!!! Index {index} is out of range. "
                    f"Indexing starts from 1 to {self.n_sessions}. "
                    f"Falling back to participant(1)."
                )
                index = 1
            return self.sessions[index - 1]  # 1-based
        # String lookup (case-insensitive partial match on folder name)
        matches = [
            s for s in self.sessions
            if index.lower() in os.path.basename(s.folder_path).lower()
        ]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            names = [os.path.basename(s.folder_path) for s in matches[:5]]
            msg = f"Ambiguous key '{index}'. Matches: {names}"
            if len(matches) > 5: msg += "..."
            raise ValueError(msg)
        else:
            raise KeyError(f"No session found matching '{index}'")

    def __call__(self, index=None):
        """
        Select a specific session by index (1-based) or name.
        Auto-updates 'session' in the interactive shell.

        Usage
        -----
            participant(1)        # focus on first session
            participant("session") # focus by partial folder name
            participant()         # list all sessions
        """
        if index is None:
            self.print()
            return self
        item = self[index]
        if hasattr(self, '_shell_ns'):
            self._shell_ns['session'] = item
            item._shell_ns = self._shell_ns
            folder_name = os.path.basename(item.folder_path)
            # Auto-select first stimulus in the session
            if len(item) > 0:
                first = item[1]
                self._shell_ns['stimulus'] = first
        
        item.print()

    def __repr__(self):
        return (
            f"Participant(path='{os.path.basename(self.folder_path)}', "
            f"sessions={self.n_sessions})"
        )

    def help(self):
        """Print available attributes and methods."""
        _participant_help(self)

    def print(self):
        """Print summary of the Participant collection."""
        _participant_print(self)


# _folder_has_subdirs — helper for rewardio()

def _folder_has_subdirs(path):
    """Check if a directory contains at least one sub-directory."""
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            return True
    return False


# rewardio() — smart entry point

def rewardio(path):
    """
    Detect what *path* points to and return the appropriate object.

        - audio file                 ->  Stimulus
        - folder of audio files      ->  Session
        - folder of folders          ->  Participant
    """
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a'):
            return Stimulus(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    elif os.path.isdir(path):
        if _folder_has_subdirs(path):
            audio_exts = ('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a')
            loose = [f for f in os.listdir(path)
                     if os.path.splitext(f)[1].lower() in audio_exts]
            if loose:
                print(
                    f"⚠️  Ignoring {len(loose)} audio file(s) at the top level of "
                    f"'{path}' — it contains sub-folders, so it is loaded as a "
                    f"participant (audio must live inside session folders)."
                )
            return Participant(path)
        else:
            return Session(path)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


# CLI

if __name__ == "__main__":
    import sys, code
    if len(sys.argv) < 2:
        print("Usage: python rewardio.py <audio_file_or_folder>")
        sys.exit(1)

    obj = rewardio(sys.argv[1])

    # Set up convenient variables for the interactive shell
    ns = {
        'rewardio': rewardio,
        'Stimulus': Stimulus,
        'Session': Session,
        'Participant': Participant,
        'clear': clear,
        'meter': True,
    }

    if isinstance(obj, Participant):
        ns['participant'] = obj
        ns['session'] = obj[1] if obj.n_sessions > 0 else obj
        ns['stimulus'] = ns['session'][1] if isinstance(ns['session'], Session) and len(ns['session']) > 0 else None
        obj._shell_ns = ns
        if isinstance(ns['session'], Session):
            ns['session']._shell_ns = ns
    elif isinstance(obj, Session):
        ns['session'] = obj
        ns['stimulus'] = obj[1] if len(obj) > 0 else None
        obj._shell_ns = ns
    else:
        ns['stimulus'] = obj
        ns['session'] = obj

    clear()
    print(obj)

    if isinstance(obj, Participant):
        print(
            "\nrewardio is an interactive tool built for music analysis.\n"
            "Please type participant.help(), session.help(), or stimulus.help() to get started.\n\n"
            "  participant          -> all sessions for this participant\n"
            "  session          -> currently focused session\n"
            "  stimulus         -> currently focused item\n"
            "  participant(1)       -> focus on a specific session\n"
            '  participant("name")  -> focus session by folder name\n'
            "  session(1)       -> focus on a specific item\n"
            '  session("name")  -> focus by partial filename match\n'
        )
    else:
        print(
            "\nrewardio is an interactive tool built for music analysis.\n"
            "Please type session.help() or stimulus.help() to get started.\n\n"
            "  session          -> the full collection\n"
            "  stimulus         -> currently focused item\n"
            "  session(1)       -> focus on a specific item by index\n"
            '  session("name")  -> focus by partial filename match\n'
        )
    code.interact(local=ns, banner="")
