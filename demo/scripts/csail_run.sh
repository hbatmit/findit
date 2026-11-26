#!/usr/bin/env bash
# Start (or restart) the VLM server and the demo API in tmux sessions.
#
#   bash scripts/csail_run.sh          start both
#   bash scripts/csail_run.sh stop     stop both
#   bash scripts/csail_run.sh logs     tail both logs
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env.csail ] || { echo "run scripts/csail_setup.sh first" >&2; exit 1; }
# shellcheck disable=SC1091
. .env.csail
# shellcheck disable=SC1091
. "$FINDIT_VENV/bin/activate"
mkdir -p logs data

case "${1:-start}" in
  stop)
    tmux kill-session -t findit-api 2>/dev/null || true
    tmux kill-session -t findit-vllm 2>/dev/null || true
    echo "stopped"; exit 0 ;;
  logs)
    exec tail -n 40 -f logs/vllm.log logs/api.log ;;
  start) ;;
  *) echo "usage: $0 [start|stop|logs]" >&2; exit 1 ;;
esac

command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }

# --- VLM ----------------------------------------------------------------------
if ! curl -sf "http://127.0.0.1:${FINDIT_VLM_PORT}/v1/models" >/dev/null; then
  tmux kill-session -t findit-vllm 2>/dev/null || true
  tmux new-session -d -s findit-vllm \
    "source .env.csail && source $FINDIT_VENV/bin/activate && \
     vllm serve $FINDIT_VLM_MODEL --port $FINDIT_VLM_PORT --host 127.0.0.1 \
       --max-model-len 8192 --limit-mm-per-prompt '{\"image\":2}' --gpu-memory-utilization 0.6 \
       2>&1 | tee logs/vllm.log"
  echo -n "waiting for vLLM (first start takes a few minutes)"
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${FINDIT_VLM_PORT}/v1/models" >/dev/null && break
    echo -n "."; sleep 5
  done
  echo
  curl -sf "http://127.0.0.1:${FINDIT_VLM_PORT}/v1/models" >/dev/null || { echo "vLLM did not come up; see logs/vllm.log" >&2; exit 1; }
fi
echo "VLM up on :${FINDIT_VLM_PORT}"

# --- API ----------------------------------------------------------------------
tmux kill-session -t findit-api 2>/dev/null || true
tmux new-session -d -s findit-api \
  "source .env.csail && source $FINDIT_VENV/bin/activate && \
   uvicorn server.app:app --host 0.0.0.0 --port $FINDIT_API_PORT 2>&1 | tee logs/api.log"
for _ in $(seq 1 60); do
  curl -s -o /dev/null "http://127.0.0.1:${FINDIT_API_PORT}/" && break; sleep 2
done
echo
echo "findit demo:  http://$(hostname -f):${FINDIT_API_PORT}/   (CSAIL network or VPN)"
echo "logs:         bash scripts/csail_run.sh logs"
echo "stop:         bash scripts/csail_run.sh stop"
