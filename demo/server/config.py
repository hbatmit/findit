"""Configuration: one TOML file, optional env overrides, built-in mock defaults.

Resolution order for every key:
    env FINDIT_<SECTION>_<KEY>  >  config.toml  >  DEFAULTS below

The file path comes from FINDIT_CONFIG (default ./config.toml relative to the
working directory). Module-level names (DETECTOR, VOCAB, ...) are kept for the
rest of the code; `python -m server.config` prints the resolved config and
`python -m server.config --shell` prints it as export lines for shell scripts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Ubuntu 22.04 ships 3.10
    import tomli as tomllib

DEFAULTS: dict[str, dict] = {
    "server": {"host": "0.0.0.0", "port": 8000, "password_hash": "", "public_hostname": "", "data_dir": "./data", "thumb_size": 96},
    "detector": {"backend": "mock", "owlv2_model": "google/owlv2-base-patch16-ensemble",
                 "yoloe_weights": "yoloe-11s-seg.pt", "threshold": 0.25,
                 "sam3_model": "facebook/sam3", "sam3_threshold": 0.5, "sam3_dtype": "float32", "sam3_nms_iou": 0.7,
                 "vocab": ["sunglasses", "glasses", "keys", "headphones", "earbuds", "book", "scissors",
                           "phone", "wallet", "remote control", "mug", "bottle", "laptop", "pen",
                           "notebook", "hat", "bag"]},
    "embedder": {"backend": "mock", "siglip2_model": "google/siglip2-base-patch16-224"},
    "vlm": {"backend": "mock", "model": "Qwen/Qwen3-VL-4B-Instruct", "dtype": "float16",
            "url": "http://127.0.0.1:8001/v1", "port": 8001,
            "gpu_memory_utilization": 0.6, "max_model_len": 8192},
    "pipeline": {"sample_fps": 1.0, "mask_crops": True, "track_sim": 0.80, "track_iou": 0.10, "query_min_sim": 0.05,
                 "scene_note_every_s": 5.0, "ask_max_images": 6},
    "deploy": {"gpu": [-1], "venv": "./.venv", "hf_home": "", "torch_index": ""},
}

CONFIG_PATH = Path(os.environ.get("FINDIT_CONFIG", "config.toml"))
PATH_KEYS = {"data_dir", "venv", "hf_home"}   # "~" and "$USER" are expanded in these


def _coerce(value: str, like):
    if isinstance(like, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, list):
        return [s.strip() for s in value.split(",") if s.strip()]
    return value


def load(path: Path = CONFIG_PATH) -> dict[str, dict]:
    cfg = {s: dict(v) for s, v in DEFAULTS.items()}
    if path.exists():
        with path.open("rb") as f:
            for section, values in tomllib.load(f).items():
                cfg.setdefault(section, {}).update(values)
    for section, values in cfg.items():
        for key, default in values.items():
            env = os.environ.get(f"FINDIT_{section}_{key}".upper())
            if env is not None:
                values[key] = _coerce(env, default)
            if key in PATH_KEYS and isinstance(values[key], str) and values[key]:
                values[key] = os.path.expandvars(os.path.expanduser(values[key]))
    g = cfg["deploy"]["gpu"]
    g = [g] if isinstance(g, int) else list(g)
    cfg["deploy"]["gpu"] = [int(x) for x in g]
    return cfg


CFG = load()

# ---- names used by the rest of the code -----------------------------------
DETECTOR: str = CFG["detector"]["backend"]
OWLV2_MODEL: str = CFG["detector"]["owlv2_model"]
YOLOE_WEIGHTS: str = CFG["detector"]["yoloe_weights"]
DET_THRESHOLD: float = CFG["detector"]["threshold"]
VOCAB: list[str] = CFG["detector"]["vocab"]
SAM3_MODEL: str = CFG["detector"]["sam3_model"]
SAM3_THRESHOLD: float = CFG["detector"]["sam3_threshold"]
SAM3_DTYPE: str = CFG["detector"]["sam3_dtype"]
SAM3_NMS_IOU: float = CFG["detector"]["sam3_nms_iou"]

EMBEDDER: str = CFG["embedder"]["backend"]
SIGLIP2_MODEL: str = CFG["embedder"]["siglip2_model"]

VLM_BACKEND: str = CFG["vlm"]["backend"]
VLM_URL: str = CFG["vlm"]["url"]
VLM_MODEL: str = CFG["vlm"]["model"]
VLM_DTYPE: str = CFG["vlm"]["dtype"]

SAMPLE_FPS: float = CFG["pipeline"]["sample_fps"]
MASK_CROPS: bool = CFG["pipeline"]["mask_crops"]
TRACK_SIM: float = CFG["pipeline"]["track_sim"]
TRACK_IOU: float = CFG["pipeline"]["track_iou"]
QUERY_MIN_SIM: float = CFG["pipeline"]["query_min_sim"]
SCENE_NOTE_EVERY_S: float = CFG["pipeline"]["scene_note_every_s"]
ASK_MAX_IMAGES: int = CFG["pipeline"]["ask_max_images"]

HOST: str = CFG["server"]["host"]
PASSWORD_HASH: str = CFG["server"]["password_hash"]
PORT: int = CFG["server"]["port"]
PUBLIC_HOSTNAME: str = CFG["server"]["public_hostname"]
DATA_DIR = Path(CFG["server"]["data_dir"])
DB_PATH = DATA_DIR / "memory.sqlite"
UPLOAD_DIR = DATA_DIR / "uploads"
THUMB_SIZE: int = CFG["server"]["thumb_size"]
GPUS: list[int] = CFG["deploy"]["gpu"]          # [-1] means every visible GPU


def _main(argv: list[str]) -> None:
    if "--shell" in argv:
        for section, values in CFG.items():
            for key, v in values.items():
                if key == "password_hash":
                    continue
                if isinstance(v, list):
                    v = ",".join(map(str, v))
                print(f"export FINDIT_{section.upper()}_{key.upper()}='{v}'")
        return
    print(f"# config file: {CONFIG_PATH} ({'found' if CONFIG_PATH.exists() else 'not found, defaults'})")
    for section, values in CFG.items():
        print(f"[{section}]")
        for key, v in values.items():
            print(f"  {key} = {'***' if key == 'password_hash' and v else v!r}")


if __name__ == "__main__":
    _main(sys.argv[1:])
