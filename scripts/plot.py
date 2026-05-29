import tkinter as tk
import numpy as np
import librosa
import librosa.display

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from play import _show_figure, _get_tk_root


def plot_beats(stimulus, xlim=None, ylim=None):
    """Plot waveform with detected beat markers."""
    y_mono = stimulus._to_mono()
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y=y_mono, sr=stimulus.sr, ax=ax,
                             linewidth=1.0, alpha=0.85)
    ax.vlines(stimulus.beat_times, ymin=y_mono.min(), ymax=y_mono.max(),
              color="r", linestyles="dashed", alpha=0.8,
              label="detected beats")
              
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Beats — {stimulus.audio_file_name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _show_figure(fig, title=f"Beats — {stimulus.audio_file_name}")


def plot_waveform(stimulus, xlim=None, ylim=None):
    """Plot the raw waveform."""
    y_mono = stimulus._to_mono()
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y=y_mono, sr=stimulus.sr, ax=ax,
                             linewidth=1.0, alpha=0.85)
                             
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform — {stimulus.audio_file_name}")
    fig.tight_layout()
    _show_figure(fig, title=f"Waveform — {stimulus.audio_file_name}")


def plot_beats_and_onsets(stimulus, xlim=None, ylim=None):
    """Plot waveform with both detected beat markers (red) and onsets (green)."""
    y_mono = stimulus._to_mono()
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot waveform
    librosa.display.waveshow(y=y_mono, sr=stimulus.sr, ax=ax,
                             linewidth=1.0, alpha=0.6)
    
    # Plot beats (red dashed)
    ax.vlines(stimulus.beat_times, ymin=y_mono.min(), ymax=y_mono.max(),
              color="r", linestyles="dashed", alpha=0.8,
              label="detected beats")
              
    # Plot onsets (green solid)
    if stimulus.onset_times is not None:
        ax.vlines(stimulus.onset_times, ymin=y_mono.min(), ymax=y_mono.max(),
                  color="g", linestyles="solid", linewidth=1.5, alpha=0.7,
                  label="onsets")
    else:
        print("Warning: .onset_times is None. Call .detect_onsets() first.")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(f"Beats + Onsets — {stimulus.audio_file_name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _show_figure(fig, title=f"Beats + Onsets — {stimulus.audio_file_name}")


def plot_interactive(stimulus, xlim=None, ylim=None):
    """
    Unified interactive plot: waveform + toggleable beats & onsets.
    Opens a tkinter window with two toggle buttons.
    """
    import os

    y_mono = stimulus._to_mono()
    sr = stimulus.sr
    duration = len(y_mono) / sr
    times = np.linspace(0, duration, len(y_mono))

    # ── Colors (Catppuccin Mocha) ─────────
    BG       = "#1e1e2e"
    SURFACE  = "#181825"
    FG       = "#cdd6f4"
    BORDER   = "#45475a"
    BEAT_CLR = "#f38ba8"   # red-pink
    ONSET_CLR = "#a6e3a1"  # green
    WAVE_CLR = "#89b4fa"   # blue
    BTN_ON   = "#89b4fa"
    BTN_OFF  = "#45475a"

    # ── State ─────────────────────────────
    state = {
        "beats_on": stimulus._beat_times is not None,
        "onsets_on": stimulus.onset_times is not None,
        "beat_lines": None,    # LineCollection handle
        "onset_lines": None,
    }

    # ── Tkinter window ────────────────────
    root = _get_tk_root()
    win = tk.Toplevel(root)
    win.title(f"Plot — {stimulus.audio_file_name}")
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

    # ── Matplotlib figure ─────────────────
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    # Waveform (always visible)
    ax.plot(times, y_mono, color=WAVE_CLR, linewidth=0.5, alpha=0.85)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, duration)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("Time (s)", color=FG, fontsize=9)
    ax.set_ylabel("Amplitude", color=FG, fontsize=9)
    ax.set_title(stimulus.audio_file_name, color=FG, fontsize=11, pad=8)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    fig.tight_layout(pad=1.5)

    # ── Draw helpers ──────────────────────
    def _draw_beats():
        if state["beat_lines"] is not None:
            state["beat_lines"].remove()
            state["beat_lines"] = None
        if state["beats_on"] and stimulus._beat_times is not None:
            lc = ax.vlines(stimulus.beat_times,
                           ymin=y_mono.min(), ymax=y_mono.max(),
                           color=BEAT_CLR, linestyles="dashed",
                           linewidth=0.8, alpha=0.7, label="beats")
            state["beat_lines"] = lc

    def _draw_onsets():
        if state["onset_lines"] is not None:
            state["onset_lines"].remove()
            state["onset_lines"] = None
        if state["onsets_on"] and stimulus.onset_times is not None:
            lc = ax.vlines(stimulus.onset_times,
                           ymin=y_mono.min(), ymax=y_mono.max(),
                           color=ONSET_CLR, linestyles="solid",
                           linewidth=1.2, alpha=0.7, label="onsets")
            state["onset_lines"] = lc

    def _refresh():
        """Redraw overlays + legend, then update canvas."""
        _draw_beats()
        _draw_onsets()
        # Rebuild legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=8,
                      facecolor=SURFACE, edgecolor=BORDER,
                      labelcolor=FG)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        canvas.draw_idle()

    # Initial overlays
    _draw_beats()
    _draw_onsets()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8,
                  facecolor=SURFACE, edgecolor=BORDER,
                  labelcolor=FG)

    # ── Embed figure ──────────────────────
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))

    # ── Status label ──────────────────────
    status_var = tk.StringVar(value="")
    status_label = tk.Label(win, textvariable=status_var,
                            font=("Menlo", 10), fg="#a6adc8", bg=BG)
    status_label.pack(pady=(4, 0))

    # ── Toggle buttons ────────────────────
    btn_frame = tk.Frame(win, bg=BG)
    btn_frame.pack(pady=(4, 12))

    btn_style = dict(font=("Menlo", 13), width=12, bd=0,
                     highlightthickness=0, relief="flat", cursor="hand2")

    def _update_btn_colors():
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

    def _toggle_beats():
        if not state["beats_on"]:
            # Turning ON — trigger lazy detection if needed
            if stimulus._beat_times is None:
                status_var.set("Detecting beats...")
                win.update_idletasks()
                _ = stimulus.beat_times          # triggers detection
                status_var.set(f"Detected {len(stimulus.beat_times)} beats.")
            state["beats_on"] = True
        else:
            state["beats_on"] = False
            status_var.set("")
        _update_btn_colors()
        _refresh()

    def _toggle_onsets():
        if not state["onsets_on"]:
            # Turning ON — trigger lazy detection if needed
            if stimulus.onset_times is None:
                status_var.set("Detecting onsets (may run separation)...")
                win.update_idletasks()
                stimulus.detect_onsets()
                status_var.set(f"Detected {len(stimulus.onset_times)} onsets.")
            state["onsets_on"] = True
        else:
            state["onsets_on"] = False
            status_var.set("")
        _update_btn_colors()
        _refresh()

    beats_btn = tk.Button(btn_frame, text="Beats",
                          command=_toggle_beats, **btn_style)
    beats_btn.pack(side="left", padx=8)

    onsets_btn = tk.Button(btn_frame, text="Onsets",
                           command=_toggle_onsets, **btn_style)
    onsets_btn.pack(side="left", padx=8)

    _update_btn_colors()

    # ── Close ─────────────────────────────
    win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))
    win.wait_window()




def plot_session_boxplots(session):
    """
    Plot boxplots for BPM, LUFS, and Syncopation Score across all session.
    Shows min, max, median, quartiles, and mean (red diamond).
    """
    import numpy as np

    # Ensure per-item metrics are computed
    has_bpm = any(getattr(s, '_bpm', None) is not None for s in session.items)
    has_sync = any(s.toussaint_syncopation_score is not None for s in session.items)
    if not has_bpm and not has_sync:
        session._process_all_items()

    # Gather data — only include items that have values
    bpms = [s.bpm for s in session.items if getattr(s, '_bpm', None) is not None]
    lufs = [s.loudness_lufs for s in session.items]  # always available
    syncs = [s.toussaint_syncopation_score for s in session.items
             if s.toussaint_syncopation_score is not None]

    # Build list of (data, label, unit) for panels that have data
    panels = []
    if lufs:
        panels.append((lufs, 'LUFS', 'LUFS'))
    if bpms:
        panels.append((bpms, 'BPM', 'BPM'))
    if syncs:
        panels.append((syncs, 'Syncopation', 'Score (0-100)'))

    if not panels:
        print("No data to plot. Run process_and_save() first.")
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (data, label, unit) in zip(axes, panels):
        bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red',
                                       markeredgecolor='red', markersize=7),
                        boxprops=dict(facecolor='#D6EAF8', edgecolor='#2C3E50'),
                        medianprops=dict(color='#2C3E50', linewidth=2),
                        whiskerprops=dict(color='#2C3E50'),
                        capprops=dict(color='#2C3E50'))

        mean_val = np.mean(data)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_ylabel(unit)
        ax.tick_params(bottom=False, labelbottom=False)

        # Annotate min, max, mean
        ax.text(1.15, np.min(data), f"min: {np.min(data):.1f}", va='center', fontsize=9)
        ax.text(1.15, np.max(data), f"max: {np.max(data):.1f}", va='center', fontsize=9)
        ax.text(1.15, mean_val, f"mean: {mean_val:.1f}", va='center', fontsize=9,
                color='red')

    fig.suptitle("Session Distribution", fontsize=14, fontweight='bold')
    fig.tight_layout()
    _show_figure(fig, title="Session Boxplots")


def plot_spectrogram(stimulus, xlim=None, ylim=None, scale='mel'):
    """
    Display a spectrogram in a Catppuccin-themed Tkinter window.

    Args:
        scale: 'mel' (default), 'linear', or 'log'.
    """
    y_mono = stimulus._to_mono()
    sr = stimulus.sr
    duration = len(y_mono) / sr

    # ── Colors (Catppuccin Mocha) ─────────
    BG      = "#1e1e2e"
    FG      = "#cdd6f4"
    BORDER  = "#45475a"

    # ── Compute spectrogram ───────────────
    if scale == 'mel':
        S = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=128, fmax=sr // 2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        y_axis = 'mel'
        title_scale = 'Mel'
    else:
        # Linear STFT for both 'linear' and 'log'
        S = np.abs(librosa.stft(y_mono)) ** 2
        S_dB = librosa.power_to_db(S, ref=np.max)
        y_axis = 'log' if scale == 'log' else 'hz'
        title_scale = 'Log' if scale == 'log' else 'Linear'

    # ── Tkinter window ────────────────────
    root = _get_tk_root()
    win = tk.Toplevel(root)
    win.title(f"Spectrogram ({title_scale}) — {stimulus.audio_file_name}")
    win.configure(bg=BG)
    win.resizable(True, True)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    img = librosa.display.specshow(
        S_dB, sr=sr, x_axis='time', y_axis=y_axis,
        ax=ax, cmap='magma', fmax=sr // 2
    )

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color=FG)
    cbar.outline.set_edgecolor(BORDER)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, duration)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(f"{title_scale} Spectrogram — {stimulus.audio_file_name}",
                 color=FG, fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (s)", color=FG)
    ax.set_ylabel("Frequency (Hz)", color=FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))
    win.lift()
    win.focus_force()


def plot_pitch(stimulus, xlim=None):
    """
    Plot CREPE pitch detection over time, colored by confidence.
    Similar to the CREPE activation plot — pitch vs time with
    confidence mapped to color intensity (magma colormap).
    """
    BG     = "#1e1e2e"
    FG     = "#cdd6f4"
    BORDER = "#45475a"

    time = stimulus._pitch_time
    freq = stimulus._pitch_freq
    conf = stimulus._pitch_conf

    root = _get_tk_root()
    win = tk.Toplevel(root)
    win.title(f"Pitch (CREPE) — {stimulus.audio_file_name}")
    win.configure(bg=BG)
    win.resizable(True, True)

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Scatter: pitch vs time, colored by confidence
    sc = ax.scatter(
        time, freq,
        c=conf, cmap='magma', s=2, vmin=0, vmax=1,
        rasterized=True
    )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, label='Confidence')
    cbar.ax.yaxis.set_tick_params(color=FG)
    cbar.ax.yaxis.label.set_color(FG)
    cbar.outline.set_edgecolor(BORDER)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG)

    # Y-axis: log scale for pitch
    ax.set_yscale('log')
    ax.set_ylim(50, 2000)
    ax.set_yticks([50, 100, 200, 440, 1000, 2000])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, time[-1] if len(time) > 0 else 1)

    ax.set_title(f"CREPE Pitch — {stimulus.audio_file_name}",
                 color=FG, fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (s)", color=FG)
    ax.set_ylabel("Pitch (Hz)", color=FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))
    win.lift()
    win.focus_force()
