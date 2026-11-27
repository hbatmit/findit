#!/usr/bin/env bash
# One-shot setup for the findit demo on a CSAIL GPU host. Idempotent, no root.
#
#   cd findit/demo && bash scripts/csail_setup.sh
#
# All settings come from config.toml (copied from config.example.toml on first
# run); edit that file rather than this script.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f config.toml ] || { cp config.example.toml config.toml; echo "created config.toml from config.example.toml; edit it if needed"; }
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
eval "$("$PY" -m server.config --shell)"
VENV="$FINDIT_DEPLOY_VENV"

# --- GPU check ------------------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: expected an NVIDIA GPU host." >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo "deploy.gpu = [$FINDIT_DEPLOY_GPU] in config.toml (-1 = all): one worker with its own model copy (~12 GB) per listed GPU."

# --- model cache location (home dirs at CSAIL are AFS: slow and quota-limited) --
[ -n "$FINDIT_DEPLOY_HF_HOME" ] && export HF_HOME="$FINDIT_DEPLOY_HF_HOME"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if ! mkdir -p "$HF_HOME" 2>/dev/null || [ ! -w "$HF_HOME" ]; then
  echo "ERROR: cannot write model cache at $HF_HOME. Set deploy.hf_home in config.toml to a local disk:" >&2
  df -h 2>/dev/null | grep -E "^/dev" >&2 || true
  exit 1
fi
case "$HF_HOME" in /afs/*) echo "WARNING: model cache is under /afs (~15 GB). Set deploy.hf_home to a local disk in config.toml." ;; esac
echo "HF cache: $HF_HOME"

# --- python env -------------------------------------------------------------
echo "python: $PY ($("$PY" --version))"
case "$VENV" in /afs/*) echo "WARNING: deploy.venv is under /afs; set it to local disk (e.g. /scratch/$USER/findit-venv) in config.toml" ;; esac
mkdir -p "$(dirname "$VENV")"
if [ ! -x "$VENV/bin/python" ]; then
  # On Ubuntu, `python3.X -m venv` fails at ensurepip when python3.X-venv is not
  # installed. Create the venv without pip and bootstrap pip ourselves instead.
  if ! "$PY" -m venv "$VENV" 2>/dev/null; then
    rm -rf "$VENV"
    "$PY" -m venv --without-pip "$VENV"
    curl -sSL https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python"
  fi
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt
# --- torch wheel selection ----------------------------------------------------
# Default PyPI torch wheels are built for CUDA 13 (driver >= 580) and, from 2.15,
# no longer run on Volta (V100, compute capability 7.0). Older drivers or Volta
# GPUs need the CUDA 12.6 wheels, whose last release is torch 2.14.
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
MIN_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1)
TORCH_SPEC="torch torchvision"
TORCH_INDEX=""
if [ -n "$FINDIT_DEPLOY_TORCH_INDEX" ]; then
  TORCH_INDEX="--index-url $FINDIT_DEPLOY_TORCH_INDEX"
elif [ "${DRIVER_MAJOR:-0}" -lt 580 ] || [ "$(echo "${MIN_CC:-9} < 8.0" | bc)" = 1 ]; then
  TORCH_SPEC="torch==2.14.* torchvision==0.29.*"
  TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu126"
fi
echo "driver $DRIVER_MAJOR, min compute capability $MIN_CC -> pip install $TORCH_SPEC $TORCH_INDEX"
# shellcheck disable=SC2086
pip install -q $TORCH_SPEC $TORCH_INDEX
pip install -q "transformers>=4.57" pillow scipy accelerate huggingface_hub
python - <<'PYEOF'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "CUDA build", torch.version.cuda, "| GPU visible:", ok)
if not ok:
    raise SystemExit("ERROR: torch cannot see the GPU. Check the driver / wheel pairing (deploy.torch_index in config.toml).")
PYEOF
if [ "$FINDIT_VLM_BACKEND" = "openai" ]; then
  pip install -q vllm || echo "WARNING: vllm install failed; set vlm.backend = \"transformers\" in config.toml instead."
fi

# --- pre-download the models named in config.toml ----------------------------
python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
repos = [os.environ["FINDIT_EMBEDDER_SIGLIP2_MODEL"], os.environ["FINDIT_VLM_MODEL"]]
if os.environ["FINDIT_DETECTOR_BACKEND"] == "owlv2":
    repos.append(os.environ["FINDIT_DETECTOR_OWLV2_MODEL"])
if os.environ["FINDIT_DETECTOR_BACKEND"] == "sam3":
    repos.append(os.environ["FINDIT_DETECTOR_SAM3_MODEL"])
for repo in repos:
    print("fetching", repo)
    try:
        snapshot_download(repo)
    except Exception as e:  # gated repo without access, or offline
        if "sam3" in repo:
            raise SystemExit(f"cannot download {repo}: {e.__class__.__name__}. SAM 3 is gated: accept the terms at "
                             f"https://huggingface.co/{repo} and run `hf auth login` in this venv, then re-run setup.")
        raise
PYEOF

echo "$VENV" > .findit-venv   # lets csail_run.sh find the venv before reading config.toml
echo
echo "Setup done. Start with:  bash scripts/csail_run.sh"
