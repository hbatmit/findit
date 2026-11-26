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
echo "deploy.gpu = $FINDIT_DEPLOY_GPU in config.toml (-1 = all). Pick an index with ~0 MiB used on a shared machine."

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
pip install -q torch torchvision "transformers>=4.57" pillow accelerate huggingface_hub
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
for repo in repos:
    print("fetching", repo)
    snapshot_download(repo)
PYEOF

echo "$VENV" > .findit-venv   # lets csail_run.sh find the venv before reading config.toml
echo
echo "Setup done. Start with:  bash scripts/csail_run.sh"
