#!/usr/bin/env python3
"""
yt_download.py
~~~~~~~~~~~~~~
Reads a session-log CSV (from Spotify) and downloads each unique track from
YouTube via yt-dlp, validating duration against the Spotify metadata.

Usage:
    python yt_download.py <csv_file> [--format mp3] [--quality 192]

Example:
    python yt_download.py PS01/session_log_11-17_22h-16m.csv

Output goes to:
    downloaded_session_music/<csv_filename>/
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
import unicodedata


# ── paths (adjust if your environment differs) ──────────────────────────────
YT_DLP   = "/Users/atlyk/miniconda3/envs/yt_download/bin/yt-dlp"
FFMPEG   = "/Users/atlyk/miniconda3/envs/yt_download/bin"
FFPROBE  = "/Users/atlyk/miniconda3/envs/yt_download/bin/ffprobe"
COOKIES  = "youtube_cookies.txt"

# ── validation settings ─────────────────────────────────────────────────────
DURATION_TOLERANCE_SEC = 10   # ±10 s is acceptable
SEARCH_RESULTS        = 5    # check top 5 YouTube results
SLEEP_BETWEEN_TRACKS  = 3    # seconds to wait between downloads (avoids rate limit)


def _sanitize(name):
    """Make a string safe for filenames: lowercase, underscores, no specials."""
    # Transliterate accented chars to ASCII (e.g. Ó→O, ü→u)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.lower()
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r'[^a-z0-9_]', '', name)   # drop anything else
    name = re.sub(r'_+', '_', name)           # collapse multiple underscores
    return name.strip('_')


def parse_duration(dur_str):
    """Convert a Spotify duration string like '4:59.529' to seconds (float)."""
    parts = dur_str.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return float(dur_str)


def get_file_duration(filepath):
    """Use ffprobe to get the duration of a downloaded audio file in seconds."""
    command = [
        FFPROBE,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        filepath,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info['format']['duration'])
    except (KeyError, ValueError, json.JSONDecodeError):
        pass
    return None


def _search_youtube(query):
    """
    Run a yt-dlp search and return a list of (url, duration_sec) tuples.
    """
    probe_cmd = [
        YT_DLP,
        f"ytsearch{SEARCH_RESULTS}:{query}",
        '--cookies', COOKIES,
        '--print', '%(webpage_url)s %(duration)s',
        '--force-ipv4', '--quiet', '--no-warnings',
        '--sleep-interval', '1',
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    # Exit code may be non-zero if some results are age-restricted,
    # but other results can still be returned — so only check stdout.
    if not result.stdout.strip():
        return []

    candidates = []
    for line in result.stdout.strip().splitlines():
        parts = line.rsplit(None, 1)          # "url duration"
        if len(parts) != 2:
            continue
        url, dur_str = parts
        try:
            dur = float(dur_str)
        except ValueError:
            continue
        candidates.append((url, dur))
    return candidates


def _find_best_video(artist, title, album, expected_dur_sec):
    """
    Cascading YouTube search to find the best duration match:
      1. Search by artist + track    → exact duration match?  Use it.
      2. Search by artist + track + album → exact duration match?  Use it.
      3. Search by track name only   → exact duration match?  Use it.
      4. Fallback: pick the closest duration from all results seen.
    Returns (url, yt_duration) or (None, None).
    """
    all_candidates = []   # collect everything as (url, dur, diff)

    # ── Cascading search queries (most likely → broadest) ────────────────
    queries = [
        f"{artist} {title}",                # artist + track (most reliable)
        f"{artist} {title} {album}",        # + album (more specific)
        title,                              # track name only (last resort)
    ]

    for query in queries:
        results = _search_youtube(query)
        for url, dur in results:
            diff = abs(dur - expected_dur_sec)
            all_candidates.append((url, dur, diff))

            # Exact match found — use it immediately
            if diff <= DURATION_TOLERANCE_SEC:
                return url, dur

    # ── Fallback: closest duration from everything we found ──────────────
    if not all_candidates:
        return None, None

    all_candidates.sort(key=lambda c: c[2])
    best_url, best_dur, _ = all_candidates[0]
    return best_url, best_dur


def download_audio_by_search(artist, title, album, expected_dur_sec, outdir,
                             track_num=1, audio_format="mp3", quality="192"):
    """
    Search YouTube for *artist – title – album*, pick the result whose
    duration best matches the Spotify metadata, and download it.
    Returns (filepath, actual_duration, status) where status is one of
    'ok', 'mismatch', or 'failed'.
    """
    os.makedirs(outdir, exist_ok=True)
    safe_name = f"{track_num:02d}_{_sanitize(artist)}_{_sanitize(title)}"
    outpath = os.path.join(outdir, f"{safe_name}.%(ext)s")

    print(f"🎵 Downloading: {artist} - {title}... ", end="", flush=True)

    # ── Step 1: find the best-matching video URL ─────────────────────────
    best_url, yt_dur = _find_best_video(artist, title, album, expected_dur_sec)

    if best_url is None:
        print("❌ (No results found)")
        return None, None, "failed"

    # ── Step 2: download that specific video ─────────────────────────────
    dl_cmd = [
        YT_DLP,
        best_url,
        '--cookies', COOKIES,
        '-x', '--audio-format', audio_format,
        '--audio-quality', quality,
        '-o', outpath,
        '--ffmpeg-location', FFMPEG,
        '--force-ipv4', '--quiet', '--no-warnings',
        '--sleep-interval', '1',
    ]
    result = subprocess.run(dl_cmd)

    if result.returncode != 0:
        print("❌ (Download failed)")
        return None, None, "failed"

    # ── Find the downloaded file ─────────────────────────────────────────
    expected_file = os.path.join(outdir, f"{safe_name}.{audio_format}")
    glob_pattern = os.path.join(outdir, f"{safe_name}.*")
    matches = [f for f in glob.glob(glob_pattern)
               if not f.endswith('.part') and not f.endswith('.ytdl')]

    filepath = expected_file if os.path.isfile(expected_file) else (
        matches[0] if matches else None
    )

    if not filepath or not os.path.isfile(filepath):
        print("❌ (File not found after download)")
        return None, None, "failed"

    # ── Validate duration ────────────────────────────────────────────────
    actual_dur = get_file_duration(filepath)
    if actual_dur is None:
        actual_dur = yt_dur  # fall back to yt-dlp reported duration

    if actual_dur is None:
        print("✅ (duration check skipped)")
        return filepath, None, "ok"

    diff = abs(actual_dur - expected_dur_sec)
    if diff <= DURATION_TOLERANCE_SEC:
        print(f"✅ ({actual_dur:.0f}s ≈ {expected_dur_sec:.0f}s)")
        return filepath, actual_dur, "ok"
    else:
        print(f"⚠️  duration mismatch: got {actual_dur:.0f}s, "
              f"expected {expected_dur_sec:.0f}s (diff {diff:+.0f}s)")
        return filepath, actual_dur, "mismatch"


def read_tracks_from_csv(csv_path):
    """
    Parse the session-log CSV and return a list of unique
    (artist_name, album_name, track_name, total_track_duration) tuples.
    """
    tracks = []
    seen = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            artist   = row["artist_name"].strip()
            album    = row["album_name"].strip()
            title    = row["track_name"].strip()
            duration = row["total_track_duration"].strip()
            key = (artist.lower(), title.lower())

            if key not in seen:
                seen.add(key)
                tracks.append((artist, album, title, duration))

    return tracks


def process_csv(csv_path, audio_format="mp3", quality="192", parent_folder=None):
    """Download all tracks from a single session-log CSV."""
    # ── build output dir ─────────────────────────────────────────────────
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    if parent_folder:
        outdir = os.path.join("downloaded_session_music", parent_folder, csv_name)
    else:
        outdir = os.path.join("downloaded_session_music", csv_name)

    # ── read tracks ──────────────────────────────────────────────────────
    tracks = read_tracks_from_csv(csv_path)
    print(f"\n📋 Found {len(tracks)} unique track(s) in {csv_path}")
    print(f"📂 Output → {outdir}/\n")

    if not tracks:
        print("Nothing to download.")
        return

    # ── show tracklist before downloading ────────────────────────────────
    for i, (artist, album, title, duration) in enumerate(tracks, 1):
        print(f"  {i:>3}. {artist} – {title}  ({album})  [{duration}]")
    print()

    # ── download ─────────────────────────────────────────────────────────
    os.makedirs(outdir, exist_ok=True)
    mismatches = []
    failures   = []

    for i, (artist, album, title, duration) in enumerate(tracks, 1):
        expected_sec = parse_duration(duration)
        filepath, actual_dur, status = download_audio_by_search(
            artist, title, album, expected_sec,
            outdir=outdir,
            track_num=i,
            audio_format=audio_format,
            quality=quality,
        )
        if status == "mismatch":
            mismatches.append((artist, title, expected_sec, actual_dur))
        elif status == "failed":
            failures.append((artist, title))

        # Brief pause between tracks to avoid YouTube rate-limiting
        time.sleep(SLEEP_BETWEEN_TRACKS)

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    total = len(tracks)
    ok = total - len(mismatches) - len(failures)
    print(f"🎉 Done!  {ok}/{total} tracks matched, "
          f"{len(mismatches)} mismatched, {len(failures)} failed.\n")

    if mismatches:
        print("⚠️  Duration mismatches (may need manual check):")
        for artist, title, expected, actual in mismatches:
            print(f"   • {artist} - {title}: "
                  f"expected {expected:.0f}s, got {actual:.0f}s")
        print()

    if failures:
        print("❌ Failed downloads:")
        for artist, title in failures:
            print(f"   • {artist} - {title}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download tracks listed in a session-log CSV (or folder of CSVs) via yt-dlp."
    )
    parser.add_argument("input", help="Path to a CSV file or a folder containing CSV files.")
    parser.add_argument("--format", default="mp3", dest="audio_format",
                        help="Audio format (default: mp3).")
    parser.add_argument("--quality", default="0",
                        help="Audio quality / bitrate (default: 0 = best).")
    args = parser.parse_args()

    # ── collect CSV files ────────────────────────────────────────────────
    if os.path.isdir(args.input):
        csv_files = sorted(glob.glob(os.path.join(args.input, "*.csv")))
        if not csv_files:
            sys.exit(f"❌ No CSV files found in {args.input}")
        folder_name = os.path.basename(os.path.normpath(args.input))
        print(f"📁 Found {len(csv_files)} CSV file(s) in {args.input}/")
    elif os.path.isfile(args.input):
        csv_files = [args.input]
        folder_name = None
    else:
        sys.exit(f"❌ Path not found: {args.input}")

    # ── process each CSV ─────────────────────────────────────────────────
    for csv_path in csv_files:
        process_csv(csv_path, audio_format=args.audio_format,
                    quality=args.quality, parent_folder=folder_name)


if __name__ == "__main__":
    main()
