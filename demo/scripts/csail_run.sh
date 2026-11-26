#!/usr/bin/env bash
# Start / stop the VLM server and the demo API in tmux sessions.
# Settings come from config.toml.
#
#   bash scripts/csail_run.sh [start|stop|logs]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f config.toml ] || { echo "no config.toml: run scripts/csail_setup.sh first" >&2; exit 1; }
eval "$(python3 -m server.config --shell)"
VENV="$FINDIT_DEPLOY_VENV"
[ -d "$VENV" ] || { echo "no venv at $VENV: run scripts/csail_setup.sh first" >&2; exit 1; }
# shellcheck disable=SC1091
. "$VENV/bin/activate"
[ -n "$FINDIT_DEPLOY_HF_HOME" ] && export HF_HOME="$FINDIT_DEPLOY_HF_HOME"
[ -z "${HF_HOME:-}" ] && [ -d "/scratch/$USER/hf" ] && export HF_HOME="/scratch/$USER/hf"
mkdir -p logs

case "${1:-start}" in
  stop)
    tmux kill-session -t findit-api 2>/dev/null || true
    tmux kill-session -t findit-vllm 2>/dev/null || true
    echo "stopped"; exit 0 ;;
  logs) exec tail -n 40 -f logs/vllm.log logs/api.log ;;
  start) ;;
  *) echo "usage: $0 [start|stop|logs]" >&2; exit 1 ;;
esac
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }

VLM_HEALTH="http://127.0.0.1:${FINDIT_VLM_PORT}/v1/models"
if [ -n "$FINDIT_VLM_URL" ] && ! curl -sf "$VLM_HEALTH" >/dev/null; then
  tmux kill-session -t findit-vllm 2>/dev/null || true
  tmux new-session -d -s findit-vllm \
    "source '$VENV/bin/activate' && export HF_HOME='${HF_HOME:-}' && \
     vllm serve '$FINDIT_VLM_MODEL' --port $FINDIT_VLM_PORT --host 127.0.0.1 \
       --max-model-len $FINDIT_VLM_MAX_MODEL_LEN --limit-mm-per-prompt '{\"image\":2}' \
       --gpu-memory-utilization $FINDIT_VLM_GPU_MEMORY_UTILIZATION 2>&1 | tee logs/vllm.log"
  echo -n "waiting for vLLM (first start takes a few minutes)"
  for _ in $(seq 1 120); do curl -sf "$VLM_HEALTH" >/dev/null && break; echo -n "."; sleep 5; done; echo
  curl -sf "$VLM_HEALTH" >/dev/null || { echo "vLLM did not come up; see logs/vllm.log" >&2; exit 1; }
fi
[ -n "$FINDIT_VLM_URL" ] && echo "VLM up on :${FINDIT_VLM_PORT}" || echo "vlm.url is empty: using mock descriptions"

tmux kill-session -t findit-api 2>/dev/null || true
tmux new-session -d -s findit-api \
  "source '$VENV/bin/activate' && export HF_HOME='${HF_HOME:-}' && \
   uvicorn server.app:app --host $FINDIT_SERVER_HOST --port $FINDIT_SERVER_PORT 2>&1 | tee logs/api.log"
for _ in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:${FINDIT_SERVER_PORT}/" && break; sleep 2; done

echo
echo "findit demo:  http://${FINDIT_SERVER_PUBLIC_HOSTNAME:-$(hostname -f)}:${FINDIT_SERVER_PORT}/"
echo "logs:         bash scripts/csail_run.sh logs"
echo "stop:         bash scripts/csail_run.sh stop"
