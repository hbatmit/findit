"""Demo configuration. Everything is an environment variable so the same code
runs with mock models on a laptop and real models on the CSAIL GPU server."""
from __future__ import annotations

import os
from pathlib import Path


def _list(name: str, default: str) -> list[str]:
    return [s.strip() for s in os.environ.get(name, default).split(",") if s.strip()]


# Model backends: "mock" | "owlv2" | "yoloe"   and   "mock" | "siglip2"
DETECTOR = os.environ.get("FINDIT_DETECTOR", "mock")
EMBEDDER = os.environ.get("FINDIT_EMBEDDER", "mock")

# VLM is reached through an OpenAI-compatible endpoint (vLLM, llama.cpp server).
# Leave FINDIT_VLM_URL empty to use the mock describer.
VLM_URL = os.environ.get("FINDIT_VLM_URL", "")
VLM_MODEL = os.environ.get("FINDIT_VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")

# Open-vocabulary prompt list for the detector. Keep it short for OWLv2 speed.
VOCAB = _list(
    "FINDIT_VOCAB",
    "sunglasses,glasses,keys,headphones,earbuds,book,scissors,phone,wallet,"
    "remote control,mug,bottle,laptop,pen,notebook,hat,bag",
)

# Frame sampling rate for uploaded clips.
SAMPLE_FPS = float(os.environ.get("FINDIT_FPS", "1.0"))

# Detection / tracking thresholds.
DET_THRESHOLD = float(os.environ.get("FINDIT_DET_THRESHOLD", "0.25"))
TRACK_SIM = float(os.environ.get("FINDIT_TRACK_SIM", "0.80"))   # cosine on crop embeddings
TRACK_IOU = float(os.environ.get("FINDIT_TRACK_IOU", "0.10"))
QUERY_MIN_SIM = float(os.environ.get("FINDIT_QUERY_MIN_SIM", "0.05"))

# Storage. Only embeddings, text and tiny object crops go here; never frames.
DATA_DIR = Path(os.environ.get("FINDIT_DATA_DIR", "./data"))
DB_PATH = DATA_DIR / "memory.sqlite"
UPLOAD_DIR = DATA_DIR / "uploads"       # clips are deleted right after processing
THUMB_SIZE = int(os.environ.get("FINDIT_THUMB", "96"))
