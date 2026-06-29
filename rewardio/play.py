import os
import time as _time
import tkinter as tk
import numpy as np
import mir_eval
import sounddevice as sd
import subprocess

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents Tk conflicts
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Shared tkinter root (singleton)

_tk_root = None

def _get_tk_root():
    """Return a persistent hidden Tk root (created once)."""
    global _tk_root
    if _tk_root is None or not _tk_root.winfo_exists():
        _tk_root = tk.Tk()
        _tk_root.withdraw()
    return _tk_root


def _show_figure(fig, title="Plot"):
    """
    Display a matplotlib Figure in a tkinter Toplevel window.
    Blocks until the window is closed.
    """
    root = _get_tk_root()
    win = tk.Toplevel(root)
    win.title(title)
    win.configure(bg="#1e1e2e")
    win.resizable(True, True)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    win.lift()
    win.attributes("-topmost", True)
    win.after(100, lambda: win.attributes("-topmost", False))

    win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))
    win.wait_window()


def _format_time(seconds):
    """Format seconds as M:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def play_audio(y, sr, title="Audio Player", beat_times=None, onset_times=None, start_time=0.0, ylim=None):
    """
    Open a GUI player with matplotlib waveform, moving cursor, and controls.
    Uses FigureCanvasTkAgg + sounddevice for playback.
    """
    # Prepare audio for sounddevice: (n_samples,) or (n_samples, n_channels)
    if y.ndim == 2:
        y_mono = np.mean(y, axis=0)
        audio = y.T.astype(np.float32)
    else:
        y_mono = y.copy()
        audio = y.astype(np.float32)

    total_samples = audio.shape[0]
    duration = total_samples / sr
    times = np.linspace(start_time, start_time + duration, len(y_mono))

    # Playback state
    state = {
        "playing": False,
        "paused": False,
        "start_sample": 0,
        "play_start_wall": 0,
        "closed": False,
        "after_id": None,
    }

    def _current_sample():
        if state["playing"] and not state["paused"]:
            elapsed = _time.time() - state["play_start_wall"]
            pos = state["start_sample"] + int(elapsed * sr)
            return min(pos, total_samples)
        return state["start_sample"]

    # Colors
    BG = "#1e1e2e"
    FG = "#cdd6f4"

    # Tkinter window
    root = _get_tk_root()
    win = tk.Toplevel(root)
    win.title(title)
    win.configure(bg=BG)
    win.resizable(False, False)

    # Bring to front on macOS
    win.lift()
    win.attributes("-topmost", True)
    win.after(100, lambda: win.attributes("-topmost", False))
    try:
        subprocess.Popen([
            "osascript", "-e",
            'tell application "System Events" to set frontmost of '
            'the first process whose unix id is '
            f'{os.getpid()} to true'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # Matplotlib waveform
    fig = Figure(figsize=(12, 3.5), dpi=100, facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#181825")
    ax.plot(times, y_mono, color="#89b4fa", linewidth=0.4, alpha=0.85)
    ax.set_xlim(start_time, start_time + duration)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if beat_times is not None:
        ax.vlines(beat_times, ymin=y_mono.min(), ymax=y_mono.max(),
                  color="#f38ba8", linestyles="dashed", linewidth=0.8,
                  alpha=0.6, label="beats")
                  
    if onset_times is not None:
        ax.vlines(onset_times, ymin=y_mono.min(), ymax=y_mono.max(),
                  color="#a6e3a1", linestyles="solid", linewidth=1.2,
                  alpha=0.7, label="onsets")

    if beat_times is not None or onset_times is not None:
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#181825", edgecolor="#45475a",
                  labelcolor=FG)
                  
    ax.set_xlabel("Time (s)", color=FG, fontsize=9)
    ax.set_ylabel("Amplitude", color=FG, fontsize=9)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#45475a")
    ax.set_title(title, color=FG, fontsize=11, pad=8)
    fig.tight_layout(pad=1.5)

    # Cursor line (animated via blitting)
    cursor_line = ax.axvline(x=start_time, color="#f38ba8", linewidth=1.5, alpha=0.9)

    canvas_widget = FigureCanvasTkAgg(fig, master=win)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(padx=8, pady=(8, 0))

    # Save static background for blitting
    bg_snapshot = canvas_widget.copy_from_bbox(ax.bbox)

    # Click on waveform to seek
    def _on_click(event):
        if event.inaxes == ax:
            click_time = event.xdata
            if click_time is not None:
                # Convert time to sample index relative to the start of the audio buffer
                rel_time = click_time - start_time
                sample = int(rel_time * sr)
                sample = max(0, min(sample, total_samples - 1))
                if state["playing"] and not state["paused"]:
                    _play_from(sample)
                else:
                    state["start_sample"] = sample

    fig.canvas.mpl_connect("button_press_event", _on_click)

    # Time label
    time_var = tk.StringVar(value=f"{_format_time(0)} / {_format_time(duration)}")
    tk.Label(win, textvariable=time_var, font=("Menlo", 14),
             fg=FG, bg=BG).pack(pady=(6, 2))

    # Buttons
    btn_frame = tk.Frame(win, bg=BG)
    btn_frame.pack(pady=(2, 12))

    btn_style = dict(font=("Menlo", 16), width=5, bd=0,
                     fg=BG, activeforeground=BG,
                     highlightthickness=0, relief="flat", cursor="hand2")

    def _play_from(sample):
        sd.stop()
        chunk = audio[sample:]
        if chunk.shape[0] == 0:
            return
        state["start_sample"] = sample
        state["play_start_wall"] = _time.time()
        state["playing"] = True
        state["paused"] = False
        try:
            sd.play(chunk, samplerate=sr)
        except Exception as e:
            print(f"Audio playback error: {e}")
            state["playing"] = False

    def _play():
        _play_from(state["start_sample"])

    def _pause():
        if state["playing"] and not state["paused"]:
            state["start_sample"] = _current_sample()
            sd.stop()
            state["paused"] = True

    def _stop():
        sd.stop()
        state["playing"] = False
        state["paused"] = False
        state["start_sample"] = 0

    play_btn = tk.Button(btn_frame, text="▶", command=_play,
                         bg="#89b4fa", activebackground="#74c7ec", **btn_style)
    play_btn.pack(side="left", padx=6)

    pause_btn = tk.Button(btn_frame, text="⏸", command=_pause,
                          bg="#a6adc8", activebackground="#bac2de", **btn_style)
    pause_btn.pack(side="left", padx=6)

    stop_btn = tk.Button(btn_frame, text="■", command=_stop,
                         bg="#f38ba8", activebackground="#eba0ac", **btn_style)
    stop_btn.pack(side="left", padx=6)

    # ── Update loop (blitting — only redraws cursor) ──
    def _update():
        if state["closed"]:
            return

        pos = _current_sample()

        if state["playing"] and not state["paused"] and pos >= total_samples:
            state["playing"] = False
            state["start_sample"] = 0
            pos = 0


        # Blit: restore background, redraw only cursor
        t = (pos / sr) + start_time
        cursor_line.set_xdata([t, t])
        canvas_widget.restore_region(bg_snapshot)
        ax.draw_artist(cursor_line)
        canvas_widget.blit(ax.bbox)

        time_var.set(f"{_format_time(pos/sr)} / {_format_time(duration)}")
        state["after_id"] = win.after(30, _update)

    _update()

    def _on_close():
        state["closed"] = True
        if state["after_id"] is not None:
            win.after_cancel(state["after_id"])
        sd.stop()
        plt.close(fig)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", _on_close)
    win.wait_window()


def play_interactive(stimulus, xlim=None, ylim=None):
    """
    Unified interactive player: waveform / spectrogram / pitch + cursor
    + play/pause/stop + toggleable beats & onsets (visual + sonification).
    Switchable view modes: Waveform, Mel, Log, Linear, STFT, Pitch.
    """
    from matplotlib.figure import Figure
    import librosa
    import librosa.display

    y = stimulus.y
    sr = stimulus.sr

    # Handle xlim slicing
    start_time = 0.0
    if xlim is not None:
        start_time = float(xlim[0])
        end_time = float(xlim[1])
        s_start = int(start_time * sr)
        s_end = int(end_time * sr)
        if y.ndim == 2:
            y = y[:, s_start:s_end]
        else:
            y = y[s_start:s_end]

    # Prepare mono for display
    if y.ndim == 2:
        y_mono = np.mean(y, axis=0)
        audio_base = y.T.astype(np.float64)   # (n_samples, n_channels)
    else:
        y_mono = y.copy()
        audio_base = y.astype(np.float64)      # (n_samples,)

    total_samples = audio_base.shape[0]
    duration = total_samples / sr
    times = np.linspace(start_time, start_time + duration, len(y_mono))

    # Colors (Catppuccin Mocha)
    BG       = "#1e1e2e"
    SURFACE  = "#181825"
    FG       = "#cdd6f4"
    BORDER   = "#45475a"
    BEAT_CLR = "#f38ba8"
    ONSET_CLR = "#a6e3a1"
    WAVE_CLR = "#89b4fa"
    BTN_ON   = "#89b4fa"
    BTN_OFF  = "#45475a"
    VIEW_ON  = "#cba6f7"  # purple for active view mode

    # Click tracks (built lazily)
    tracks = {
        "beat_track": None,    # mono click track for beats
        "onset_track": None,   # mono ping track for onsets
    }

    def _build_beat_track():
        if tracks["beat_track"] is not None:
            return
        bt = stimulus.beat_times   # triggers lazy detection
        if xlim is not None:
            bt = bt[(bt >= start_time) & (bt < start_time + duration)] - start_time
        n = y_mono.shape[0]
        tracks["beat_track"] = mir_eval.sonify.clicks(bt, fs=sr, length=n)

    def _build_onset_track():
        if tracks["onset_track"] is not None:
            return
        if stimulus.onset_times is None:
            stimulus.detect_onsets()
        ot = stimulus.onset_times
        if xlim is not None:
            ot = ot[(ot >= start_time) & (ot < start_time + duration)] - start_time
        n = y_mono.shape[0]
        # High-pitch ping for onsets
        ping_len = int(0.05 * sr)
        t_ping = np.linspace(0, 0.05, ping_len)
        ping = np.sin(2 * np.pi * 1000 * t_ping) * np.exp(-t_ping * 100)
        tracks["onset_track"] = mir_eval.sonify.clicks(ot, fs=sr, length=n, click=ping)

    def _mix_audio():
        """Build the playback buffer based on current toggle state."""
        mixed = audio_base.copy()
        if state["beats_on"] and tracks["beat_track"] is not None:
            if mixed.ndim == 2:
                for ch in range(mixed.shape[1]):
                    mixed[:, ch] += tracks["beat_track"]
            else:
                mixed += tracks["beat_track"]
        if state["onsets_on"] and tracks["onset_track"] is not None:
            if mixed.ndim == 2:
                for ch in range(mixed.shape[1]):
                    mixed[:, ch] += tracks["onset_track"]
            else:
                mixed += tracks["onset_track"]
        # Normalize to original peak
        orig_peak = np.max(np.abs(audio_base))
        mix_peak = np.max(np.abs(mixed))
        if mix_peak > 0:
            mixed = mixed * (orig_peak / mix_peak)
        return mixed.astype(np.float32)

    # ── Playback state ────────────────────
    VIEW_MODES = ["Waveform", "Mel", "Log", "Linear", "Pitch"]
    state = {
        "playing": False,
        "paused": False,
        "start_sample": 0,
        "play_start_wall": 0,
        "closed": False,
        "after_id": None,
        "beats_on": False,
        "onsets_on": False,
        "beat_lines": None,
        "onset_lines": None,
        "audio": audio_base.astype(np.float32),
        "view_mode": "Waveform",
    }

    def _current_sample():
        if state["playing"] and not state["paused"]:
            elapsed = _time.time() - state["play_start_wall"]
            pos = state["start_sample"] + int(elapsed * sr)
            return min(pos, total_samples)
        return state["start_sample"]

    # Tkinter window
    root = _get_tk_root()
    win = tk.Toplevel(root)
    win.title(f"Player — {stimulus.audio_file_name}")
    win.configure(bg=BG)
    win.resizable(True, True)

    # Bring to front on macOS
    win.lift()
    win.attributes("-topmost", True)
    win.after(100, lambda: win.attributes("-topmost", False))
    try:
        import subprocess
        subprocess.Popen([
            "osascript", "-e",
            'tell application "System Events" to set frontmost of '
            'the first process whose unix id is '
            f'{os.getpid()} to true'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # Matplotlib figure
    fig = Figure(figsize=(12, 4.5), dpi=100, facecolor=BG)
    ax = fig.add_subplot(111)

    # View drawing functions

    def _draw_waveform():
        ax.plot(times, y_mono, color=WAVE_CLR, linewidth=0.4, alpha=0.85)
        ax.set_xlim(start_time, start_time + duration)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel("Amplitude", color=FG, fontsize=9)

    def _draw_spectrogram(scale):
        if scale == 'mel':
            S = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=128, fmax=sr // 2)
            S_dB = librosa.power_to_db(S, ref=np.max)
            y_axis = 'mel'
        else:
            S = np.abs(librosa.stft(y_mono)) ** 2
            S_dB = librosa.power_to_db(S, ref=np.max)
            y_axis = 'log' if scale == 'log' else 'hz'

        librosa.display.specshow(
            S_dB, sr=sr, x_axis='time', y_axis=y_axis,
            ax=ax, cmap='magma', fmax=sr // 2
        )
        ax.set_xlim(start_time, start_time + duration)
        ax.set_ylabel("Frequency (Hz)", color=FG, fontsize=9)


    def _draw_pitch():
        if stimulus._pitch_freq is None:
            status_var.set("Running CREPE pitch detection...")
            win.update_idletasks()
            from .genre import detect_pitch_crepe
            stimulus._pitch_time, stimulus._pitch_freq, stimulus._pitch_conf = detect_pitch_crepe(
                stimulus.audio_file_path
            )
            status_var.set("Pitch detection complete.")

        t = stimulus._pitch_time
        f = stimulus._pitch_freq
        c = stimulus._pitch_conf

        ax.scatter(t, f, c=c, cmap='magma', s=2, vmin=0, vmax=1, rasterized=True)
        ax.set_yscale('log')
        ax.set_ylim(50, 2000)
        ax.set_yticks([50, 100, 200, 440, 1000, 2000])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(0, t[-1] if len(t) > 0 else 1)
        ax.set_ylabel("Pitch (Hz)", color=FG, fontsize=9)

    def _redraw_view():
        """Clear and redraw the axes for the current view mode."""
        nonlocal cursor_line, bg_snapshot
        ax.clear()
        ax.set_facecolor(SURFACE)

        mode = state["view_mode"]
        if mode == "Waveform":
            _draw_waveform()
        elif mode == "Mel":
            _draw_spectrogram('mel')
        elif mode == "Log":
            _draw_spectrogram('log')
        elif mode == "Linear":
            _draw_spectrogram('linear')
        elif mode == "Pitch":
            _draw_pitch()

        ax.set_xlabel("Time (s)", color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.set_title(f"{mode} — {stimulus.audio_file_name}",
                     color=FG, fontsize=11, pad=8)
        fig.tight_layout(pad=1.5)

        # Re-add overlays (beats/onsets)
        state["beat_lines"] = None
        state["onset_lines"] = None
        _draw_overlay_beats()
        _draw_overlay_onsets()
        _rebuild_legend()

        # Re-add cursor
        cursor_line = ax.axvline(x=start_time, color=BEAT_CLR,
                                 linewidth=1.5, alpha=0.9, animated=True)

        canvas_widget.draw()
        bg_snapshot = canvas_widget.copy_from_bbox(ax.bbox)

    # ── Overlay draw helpers

    def _get_overlay_yrange():
        """Get y limits for overlay lines based on current view."""
        yl = ax.get_ylim()
        return yl[0], yl[1]

    def _draw_overlay_beats():
        if state["beat_lines"] is not None:
            state["beat_lines"].remove()
            state["beat_lines"] = None
        if state["beats_on"] and stimulus._beat_times is not None:
            bt = stimulus.beat_times
            if xlim is not None:
                bt = bt[(bt >= start_time) & (bt < start_time + duration)]
            ymin, ymax = _get_overlay_yrange()
            lc = ax.vlines(bt, ymin=ymin, ymax=ymax,
                           color=BEAT_CLR, linestyles="dashed",
                           linewidth=0.8, alpha=0.6, label="beats")
            state["beat_lines"] = lc

    def _draw_overlay_onsets():
        if state["onset_lines"] is not None:
            state["onset_lines"].remove()
            state["onset_lines"] = None
        if state["onsets_on"] and stimulus.onset_times is not None:
            ot = stimulus.onset_times
            if xlim is not None:
                ot = ot[(ot >= start_time) & (ot < start_time + duration)]
            ymin, ymax = _get_overlay_yrange()
            lc = ax.vlines(ot, ymin=ymin, ymax=ymax,
                           color=ONSET_CLR, linestyles="solid",
                           linewidth=1.2, alpha=0.7, label="onsets")
            state["onset_lines"] = lc

    def _rebuild_legend():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=8,
                      facecolor=SURFACE, edgecolor=BORDER,
                      labelcolor=FG)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    def _refresh_overlays():
        nonlocal bg_snapshot
        _draw_overlay_beats()
        _draw_overlay_onsets()
        _rebuild_legend()
        # Must redraw fully since overlays changed
        canvas_widget.draw()
        # Re-snapshot for blitting
        bg_snapshot = canvas_widget.copy_from_bbox(ax.bbox)

    # Initial draw
    _draw_waveform()
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("Time (s)", color=FG, fontsize=9)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_title(f"Waveform — {stimulus.audio_file_name}",
                 color=FG, fontsize=11, pad=8)
    fig.tight_layout(pad=1.5)

    # Cursor line
    cursor_line = ax.axvline(x=start_time, color=BEAT_CLR, linewidth=1.5, alpha=0.9,
                             animated=True)

    # Embed figure
    canvas_widget = FigureCanvasTkAgg(fig, master=win)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))

    # Save background for blitting
    bg_snapshot = canvas_widget.copy_from_bbox(ax.bbox)

    # Click on plot to seek
    def _on_click(event):
        if event.inaxes == ax:
            click_time = event.xdata
            if click_time is not None:
                rel_time = click_time - start_time
                sample = int(rel_time * sr)
                sample = max(0, min(sample, total_samples - 1))
                if state["playing"] and not state["paused"]:
                    _play_from(sample)
                else:
                    state["start_sample"] = sample

    fig.canvas.mpl_connect("button_press_event", _on_click)

    # Time label
    time_var = tk.StringVar(value=f"{_format_time(0)} / {_format_time(duration)}")
    tk.Label(win, textvariable=time_var, font=("Menlo", 14),
             fg=FG, bg=BG).pack(pady=(6, 2))

    # Status label
    status_var = tk.StringVar(value="")
    tk.Label(win, textvariable=status_var, font=("Menlo", 10),
             fg="#a6adc8", bg=BG).pack(pady=(2, 0))

    # View mode buttons
    view_frame = tk.Frame(win, bg=BG)
    view_frame.pack(pady=(4, 2))

    view_style = dict(font=("Menlo", 11), width=9, bd=0,
                      highlightthickness=0, relief="flat", cursor="hand2")

    view_buttons = {}

    def _switch_view(mode):
        if state["view_mode"] == mode:
            return
        state["view_mode"] = mode
        status_var.set(f"Switching to {mode}...")
        win.update_idletasks()
        _redraw_view()
        _update_view_btn_colors()
        status_var.set("")

    def _update_view_btn_colors():
        for name, btn in view_buttons.items():
            if name == state["view_mode"]:
                btn.configure(bg=VIEW_ON, fg=BG,
                              activebackground=VIEW_ON, activeforeground=BG)
            else:
                btn.configure(bg=BTN_OFF, fg=FG,
                              activebackground=BTN_OFF, activeforeground=FG)

    for mode_name in VIEW_MODES:
        btn = tk.Button(view_frame, text=mode_name,
                        command=lambda m=mode_name: _switch_view(m),
                        **view_style)
        btn.pack(side="left", padx=3)
        view_buttons[mode_name] = btn

    _update_view_btn_colors()

    # Playback controls
    ctrl_frame = tk.Frame(win, bg=BG)
    ctrl_frame.pack(pady=(4, 4))

    ctrl_style = dict(font=("Menlo", 16), width=5, bd=0,
                      fg=BG, activeforeground=BG,
                      highlightthickness=0, relief="flat", cursor="hand2")

    def _play_from(sample):
        sd.stop()
        audio = state["audio"]
        chunk = audio[sample:]
        if chunk.shape[0] == 0:
            return
        state["start_sample"] = sample
        state["play_start_wall"] = _time.time()
        state["playing"] = True
        state["paused"] = False
        play_pause_btn.configure(text="⏸")
        try:
            sd.play(chunk, samplerate=sr, blocksize=1024)
        except Exception as e:
            print(f"Audio playback error: {e}")
            state["playing"] = False
            play_pause_btn.configure(text="▶")

    def _play_pause():
        if state["playing"] and not state["paused"]:
            # Pause
            state["start_sample"] = _current_sample()
            sd.stop()
            state["paused"] = True
            play_pause_btn.configure(text="▶")
        else:
            # Play / Resume
            _play_from(state["start_sample"])

    def _stop():
        sd.stop()
        state["playing"] = False
        state["paused"] = False
        state["start_sample"] = 0
        play_pause_btn.configure(text="▶")

    play_pause_btn = tk.Button(ctrl_frame, text="▶", command=_play_pause,
                               bg="#89b4fa", activebackground="#74c7ec", **ctrl_style)
    play_pause_btn.pack(side="left", padx=6)

    stop_btn = tk.Button(ctrl_frame, text="■", command=_stop,
                         bg="#f38ba8", activebackground="#eba0ac", **ctrl_style)
    stop_btn.pack(side="left", padx=6)

    # Toggle buttons
    toggle_frame = tk.Frame(win, bg=BG)
    toggle_frame.pack(pady=(4, 12))

    toggle_style = dict(font=("Menlo", 13), width=12, bd=0,
                        highlightthickness=0, relief="flat", cursor="hand2")

    def _update_toggle_colors():
        beats_btn.configure(
            bg=BTN_ON if state["beats_on"] else BTN_OFF,
            fg=BG if state["beats_on"] else FG,
            activebackground=BTN_ON if state["beats_on"] else BTN_OFF,
            activeforeground=BG if state["beats_on"] else FG,
        )
        onsets_btn.configure(
            bg=BTN_ON if state["onsets_on"] else BTN_OFF,
            fg=BG if state["onsets_on"] else FG,
            activebackground=BTN_ON if state["onsets_on"] else BTN_OFF,
            activeforeground=BG if state["onsets_on"] else FG,
        )

    def _remix_if_playing():
        """Re-mix audio and restart from current pos if playing."""
        state["audio"] = _mix_audio()
        if state["playing"] and not state["paused"]:
            pos = _current_sample()
            _play_from(pos)

    def _toggle_beats():
        if not state["beats_on"]:
            if stimulus._beat_times is None:
                status_var.set("Detecting beats...")
                win.update_idletasks()
                _ = stimulus.beat_times
                status_var.set(f"Detected {len(stimulus.beat_times)} beats.")
            _build_beat_track()
            state["beats_on"] = True
        else:
            state["beats_on"] = False
            status_var.set("")
        _update_toggle_colors()
        _refresh_overlays()
        _remix_if_playing()

    def _toggle_onsets():
        if not state["onsets_on"]:
            if stimulus.onset_times is None:
                if stimulus.separated_drums is None:
                    status_var.set("⚠️  Please confirm in the terminal")
                else:
                    status_var.set("Detecting onsets...")
                win.update_idletasks()
                stimulus.detect_onsets()
                status_var.set(f"Detected {len(stimulus.onset_times)} onsets.")
            _build_onset_track()
            state["onsets_on"] = True
        else:
            state["onsets_on"] = False
            status_var.set("")
        _update_toggle_colors()
        _refresh_overlays()
        _remix_if_playing()

    beats_btn = tk.Button(toggle_frame, text="Beats",
                          command=_toggle_beats, **toggle_style)
    beats_btn.pack(side="left", padx=8)

    onsets_btn = tk.Button(toggle_frame, text="Onsets",
                           command=_toggle_onsets, **toggle_style)
    onsets_btn.pack(side="left", padx=8)

    _update_toggle_colors()

    # Update loop (blitting cursor)
    def _update():
        if state["closed"]:
            return

        pos = _current_sample()

        if state["playing"] and not state["paused"] and pos >= total_samples:
            state["playing"] = False
            state["start_sample"] = 0
            play_pause_btn.configure(text="▶")
            pos = 0

        t = (pos / sr) + start_time
        canvas_widget.restore_region(bg_snapshot)
        
        if cursor_line is not None:
            cursor_line.set_xdata([t, t])
            ax.draw_artist(cursor_line)

        canvas_widget.blit(ax.bbox)

        time_var.set(f"{_format_time(pos / sr)} / {_format_time(duration)}")
        state["after_id"] = win.after(30, _update)

    _update()

    def _on_close():
        state["closed"] = True
        if state["after_id"] is not None:
            win.after_cancel(state["after_id"])
        sd.stop()
        plt.close(fig)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", _on_close)
    win.wait_window()


def sonify_beats(y, sr, beat_times, title="Beats", xlim=None, ylim=None):
    """
    Overlay a click track on *y* at *beat_times* and open the audio player.
    Preserves original channel format (stereo/mono).
    """
    start_time = 0.0
    
    # Handle xlim slicing
    if xlim is not None:
        start_time = float(xlim[0])
        end_time = float(xlim[1])
        s_start = int(start_time * sr)
        s_end = int(end_time * sr)
        
        # Slice audio
        if y.ndim == 2:
            y = y[:, s_start:s_end]
        else:
            y = y[s_start:s_end]
            
        # Filter and shift beats (only for click generation)
        # We need relative times for mir_eval
        beat_times_rel = beat_times[(beat_times >= start_time) & (beat_times < end_time)] - start_time
    else:
        beat_times_rel = beat_times

    # Create click track (mono) at beat positions
    n_samples = y.shape[-1] if y.ndim == 2 else len(y)
    
    if n_samples == 0:
        print("Warning: selected range is empty or invalid.")
        return

    # mir_eval expects beats in seconds. If beats_rel is empty, it handles it.
    click_track = mir_eval.sonify.clicks(beat_times_rel, fs=sr, length=n_samples)

    # Add click track to each channel of the original audio
    if y.ndim == 2:
        mixed = y.copy().astype(np.float64)
        for ch in range(mixed.shape[0]):
            mixed[ch] += click_track
    else:
        mixed = y.astype(np.float64) + click_track

    # Scale mix to match original audio's peak level
    original_peak = np.max(np.abs(y))
    mixed_peak = np.max(np.abs(mixed))
    if mixed_peak > 0:
        mixed = mixed * (original_peak / mixed_peak)

    play_audio(mixed, sr, title=title, beat_times=beat_times, start_time=start_time, ylim=ylim)

def sonify_beats_and_onsets(y, sr, beat_times, onset_times, title="Beats + Onsets", xlim=None, ylim=None):
    """
    Overlay beats (standard click) and onsets (high pitch ping) on audio.
    """
    start_time = 0.0
    
    # Handle xlim slicing
    if xlim is not None:
        start_time = float(xlim[0])
        end_time = float(xlim[1])
        s_start = int(start_time * sr)
        s_end = int(end_time * sr)
        
        # Slice audio
        if y.ndim == 2:
            y = y[:, s_start:s_end]
        else:
            y = y[s_start:s_end]
            
        # Filter and shift for click generation
        beat_times_rel = beat_times[(beat_times >= start_time) & (beat_times < end_time)] - start_time
        onset_times_rel = onset_times[(onset_times >= start_time) & (onset_times < end_time)] - start_time
    else:
        beat_times_rel = beat_times
        onset_times_rel = onset_times

    # 1. Standard click track for beats
    n_samples = y.shape[-1] if y.ndim == 2 else len(y)
    
    if n_samples == 0:
        print("Warning: selected range is empty or invalid.")
        return

    beat_track = mir_eval.sonify.clicks(beat_times_rel, fs=sr, length=n_samples)

    # 2. Custom high-pitch ping for onsets
    # Generate a short 1000Hz sine wave ping
    ping_len = int(0.05 * sr)
    t = np.linspace(0, 0.05, ping_len)
    ping = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 100)
    
    # Use mir_eval to place this ping at onset times
    onset_track = mir_eval.sonify.clicks(onset_times_rel, fs=sr, length=n_samples, click=ping)

    # 3. Mix
    if y.ndim == 2:
        mixed = y.copy().astype(np.float64)
        for ch in range(mixed.shape[0]):
            mixed[ch] += beat_track + onset_track
    else:
        mixed = y.astype(np.float64) + beat_track + onset_track

    # 4. Normalize
    original_peak = np.max(np.abs(y))
    mixed_peak = np.max(np.abs(mixed))
    if mixed_peak > 0:
        mixed = mixed * (original_peak / mixed_peak)

    # 5. Play
    # Pass beat_times and onset_times so the player shows both
    # Pass absolute times because play_audio will shift the x-axis to match
    play_audio(mixed, sr, title=title, beat_times=beat_times, onset_times=onset_times, start_time=start_time, ylim=ylim)
