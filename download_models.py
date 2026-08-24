"""
Download the Essentia classifier models that power rewardio's genre, mood,
voice/instrumental, and CREPE pitch features (~45 MB total).

Usage:
    python download_models.py

Files are saved to rewardio/models/. Already-downloaded files are skipped,
so re-running is always safe. Uses only the Python standard library.
"""
import os
import sys
import urllib.request

BASE = "https://essentia.upf.edu/models"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "rewardio", "models")

# (url path under BASE, local filename)
FILES = [
    # Shared Discogs-EffNet embedding model
    ("feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
     "discogs-effnet-bs64-1.pb"),
    # Classification heads (.pb weights + .json class labels)
    ("classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
     "genre_discogs400-discogs-effnet-1.pb"),
    ("classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
     "genre_discogs400-discogs-effnet-1.json"),
    ("classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb",
     "voice_instrumental-discogs-effnet-1.pb"),
    ("classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.json",
     "voice_instrumental-discogs-effnet-1.json"),
    ("classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
     "mood_happy-discogs-effnet-1.pb"),
    ("classification-heads/mood_happy/mood_happy-discogs-effnet-1.json",
     "mood_happy-discogs-effnet-1.json"),
    ("classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb",
     "mood_sad-discogs-effnet-1.pb"),
    ("classification-heads/mood_sad/mood_sad-discogs-effnet-1.json",
     "mood_sad-discogs-effnet-1.json"),
    ("classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
     "mood_aggressive-discogs-effnet-1.pb"),
    ("classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.json",
     "mood_aggressive-discogs-effnet-1.json"),
    ("classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb",
     "mood_relaxed-discogs-effnet-1.pb"),
    ("classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.json",
     "mood_relaxed-discogs-effnet-1.json"),
    # CREPE pitch tracker
    ("pitch/crepe/crepe-medium-1.pb",
     "crepe-medium-1.pb"),
]


def download(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done // total
                print(f"\r    {done / 1048576:6.1f} / {total / 1048576:.1f} MB  ({pct}%)",
                      end="", flush=True)
    os.replace(tmp, dest)  # only land the file once fully downloaded
    print()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading Essentia models → {MODELS_DIR}\n")

    failed = []
    for path, name in FILES:
        dest = os.path.join(MODELS_DIR, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print(f"  [skip] {name} (already present)")
            continue
        print(f"  [get ] {name}")
        try:
            download(f"{BASE}/{path}", dest)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append(name)

    print()
    missing = [name for _, name in FILES
               if not os.path.isfile(os.path.join(MODELS_DIR, name))]
    if failed or missing:
        print(f"✗ {len(set(failed + missing))} file(s) missing — re-run this "
              f"script to retry. Genre/mood/pitch features need all files.")
        return 1
    print(f"✓ All {len(FILES)} model files present — genre, mood, "
          f"voice/instrumental, and pitch features are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
