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
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
[ "$VRAM_MB" -ge 14000 ] || echo "NOTE: < 14 GB GPU memory; consider vlm.model = \"Qwen/Qwen3-VL-2B-Instruct\" in config.toml"

# --- model cache location (home dirs at CSAIL are often quota-limited) ------
if [ -n "$FINDIT_DEPLOY_HF_HOME" ]; then
  export HF_HOME="$FINDIT_DEPLOY_HF_HOME"
elif [ -z "${HF_HOME:-}" ] && [ -d /scratch ] && [ -w /scratch ]; then
  export HF_HOME="/scratch/$USER/hf"
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo "HF cache: $HF_HOME   (set deploy.hf_home in config.toml to change)"

# --- python env -------------------------------------------------------------
echo "python: $PY ($("$PY" --version))"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q torch torchvision transformers pillow accelerate huggingface_hub
pip install -q vllm || echo "WARNING: vllm install failed. Use llama.cpp's llama-server instead (see README)."

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

echo
echo "Setup done. Start with:  bash scripts/csail_run.sh"
