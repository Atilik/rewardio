"""
Essentia-based audio classification and pitch detection.

All classification heads share the same Discogs-EffNet embeddings,
so running multiple classifiers on the same file is fast after the
first embedding computation.

Models are loaded lazily and cached for reuse across calls.
"""

import os
import json
import numpy as np

# ── Model paths ───────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")

_EFFNET_PATH = os.path.join(_MODELS_DIR, "discogs-effnet-bs64-1.pb")

# Classification heads  (all use Discogs-EffNet embeddings)
_HEAD_FILES = {
    "genre":            "genre_discogs400-discogs-effnet-1",
    "voice_instrumental": "voice_instrumental-discogs-effnet-1",
    "mood_happy":       "mood_happy-discogs-effnet-1",
    "mood_sad":         "mood_sad-discogs-effnet-1",
    "mood_aggressive":  "mood_aggressive-discogs-effnet-1",
    "mood_relaxed":     "mood_relaxed-discogs-effnet-1",
}

_CREPE_PATH = os.path.join(_MODELS_DIR, "crepe-medium-1.pb")

# ── Lazy caches ───────────────────────────────
_embedding_model = None
_head_models = {}          
_head_classes = {}         
_crepe_model = None
_essentia_silenced = False


def _silence_essentia():
    """Suppress ALL Essentia + TensorFlow log spam."""
    global _essentia_silenced
    if not _essentia_silenced:
        # Silence TF/ABSL C++ logs BEFORE any TF import
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
        import essentia
        essentia.log.warningActive = False
        essentia.log.infoActive = False
        _essentia_silenced = True


def _get_embedding_model():
    """Load Discogs-EffNet embedding model (once, cached)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    _silence_essentia()

    if not os.path.isfile(_EFFNET_PATH):
        raise FileNotFoundError(
            f"Embedding model not found: {_EFFNET_PATH}\n"
            f"Run the download script or see genre.py for instructions."
        )

    from essentia.standard import TensorflowPredictEffnetDiscogs
    _embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=_EFFNET_PATH,
        output="PartitionedCall:1"
    )
    return _embedding_model


def _get_head_model(name):
    """Load a classification head (once, cached). Returns (model, classes)."""
    if name in _head_models:
        return _head_models[name], _head_classes[name]

    _silence_essentia()

    base = _HEAD_FILES[name]
    pb_path = os.path.join(_MODELS_DIR, f"{base}.pb")
    json_path = os.path.join(_MODELS_DIR, f"{base}.json")

    if not os.path.isfile(pb_path):
        raise FileNotFoundError(f"Model not found: {pb_path}")

    from essentia.standard import TensorflowPredict2D

    # genre_discogs400 uses different node names than the others
    if name == "genre":
        model = TensorflowPredict2D(
            graphFilename=pb_path,
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0"
        )
    else:
        model = TensorflowPredict2D(
            graphFilename=pb_path,
            output="model/Softmax"
        )

    classes = []
    if os.path.isfile(json_path):
        with open(json_path) as f:
            meta = json.load(f)
            classes = meta.get("classes", [])

    _head_models[name] = model
    _head_classes[name] = classes
    return model, classes


def _load_audio_16k(audio_file_path):
    """Load audio at 16kHz mono (required by Discogs-EffNet)."""
    _silence_essentia()
    from essentia.standard import MonoLoader
    return MonoLoader(
        filename=audio_file_path,
        sampleRate=16000,
        resampleQuality=4
    )()


def _get_embeddings(audio_file_path):
    """Compute Discogs-EffNet embeddings for an audio file."""
    audio = _load_audio_16k(audio_file_path)
    model = _get_embedding_model()
    return model(audio)


def _classify(name, audio_file_path, top_n=5, embeddings=None):
    """
    Run a classification head on an audio file.

    Returns list of (label, confidence) tuples.
    """
    if embeddings is None:
        embeddings = _get_embeddings(audio_file_path)

    head, classes = _get_head_model(name)
    predictions = head(embeddings)
    avg = np.mean(predictions, axis=0)

    top_idx = np.argsort(avg)[::-1][:top_n]
    return [(classes[i] if i < len(classes) else str(i), float(avg[i])) for i in top_idx]


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def classify_genre(audio_file_path, top_n=5):
    """Genre classification (400 Discogs sub-genres)."""
    return _classify("genre", audio_file_path, top_n=top_n)

def classify_voice_instrumental(audio_file_path):
    """Voice vs instrumental classification."""
    return _classify("voice_instrumental", audio_file_path, top_n=2)

def classify_mood(audio_file_path):
    """
    Classify mood across 4 dimensions: happy, sad, aggressive, relaxed.
    Returns dict of {mood: confidence}.
    Runs all 4 binary classifiers on shared embeddings.
    """
    embeddings = _get_embeddings(audio_file_path)
    return _classify_mood_from_embeddings(audio_file_path, embeddings)


def _classify_mood_from_embeddings(audio_file_path, embeddings):
    """Internal: classify mood using pre-computed embeddings."""
    moods = {}
    for mood_name in ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed"]:
        short_name = mood_name.replace("mood_", "")
        results = _classify(mood_name, audio_file_path, top_n=2, embeddings=embeddings)
        for label, conf in results:
            if label.lower() == short_name:
                moods[short_name] = conf
                break
        else:
            moods[short_name] = results[0][1] if results[0][0].lower() == short_name else 1.0 - results[0][1]
    return moods


def classify_all(audio_file_path):
    """
    Run ALL classifiers on shared embeddings (fast — one embedding pass).

    Returns dict:
        genre:            list of (label, confidence)
        voice_instrumental: list of (label, confidence)
        mood:             dict {happy, sad, aggressive, relaxed}
    """
    embeddings = _get_embeddings(audio_file_path)
    return {
        "genre":              _classify("genre", audio_file_path, top_n=5, embeddings=embeddings),
        "voice_instrumental": _classify("voice_instrumental", audio_file_path, top_n=2, embeddings=embeddings),
        "mood":               _classify_mood_from_embeddings(audio_file_path, embeddings),
    }


def detect_pitch_crepe(audio_file_path):
    """
    Pitch detection using CREPE (medium model).

    Returns:
        (time, frequency, confidence) — numpy arrays.
        time: timestamps in seconds
        frequency: pitch in Hz (0 where unvoiced)
        confidence: detection confidence [0, 1]
    """
    global _crepe_model
    _silence_essentia()

    if _crepe_model is None:
        if not os.path.isfile(_CREPE_PATH):
            raise FileNotFoundError(f"CREPE model not found: {_CREPE_PATH}")
        from essentia.standard import PitchCREPE
        _crepe_model = PitchCREPE(graphFilename=_CREPE_PATH)

    audio = _load_audio_16k(audio_file_path)
    time, frequency, confidence, activations = _crepe_model(audio)
    return time, frequency, confidence


def detect_key(audio_file_path):
    """
    Detect musical key and scale using Essentia's KeyExtractor.

    Returns:
        (key, scale, strength) — e.g. ("C", "minor", 0.85)
        key: str — tonic pitch class (C, C#, D, …, B)
        scale: str — "major" or "minor"
        strength: float — confidence of the key estimation [0, 1]
    """
    _silence_essentia()
    from essentia.standard import MonoLoader, KeyExtractor

    audio = MonoLoader(
        filename=audio_file_path,
        sampleRate=44100,
        resampleQuality=4
    )()

    extractor = KeyExtractor()
    key, scale, strength = extractor(audio)
    return key, scale, float(strength)
