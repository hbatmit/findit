#!/usr/bin/env bash
# Start / stop the VLM server and the demo API in tmux sessions.
# Settings come from config.toml.
#
#   bash scripts/csail_run.sh [start|stop|logs]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f config.toml ] || { echo "no config.toml: run scripts/csail_setup.sh first" >&2; exit 1; }
[ -f .findit-venv ] || { echo "no .findit-venv: run scripts/csail_setup.sh first" >&2; exit 1; }
VENV="$(cat .findit-venv)"
[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV: run scripts/csail_setup.sh first" >&2; exit 1; }
# shellcheck disable=SC1091
. "$VENV/bin/activate"
eval "$(python -m server.config --shell)"   # venv python: has tomllib/tomli
[ -n "$FINDIT_DEPLOY_HF_HOME" ] && export HF_HOME="$FINDIT_DEPLOY_HF_HOME"
[ -z "${HF_HOME:-}" ] && [ -d "/scratch/$USER/hf" ] && export HF_HOME="/scratch/$USER/hf"
mkdir -p logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID            # make CUDA indices match nvidia-smi
[ "$FINDIT_DEPLOY_GPU" != "-1" ] && export CUDA_VISIBLE_DEVICES="$FINDIT_DEPLOY_GPU"
# tmux runs commands through the login shell (tcsh at CSAIL), so each service
# gets its own bash launcher file instead of an inline command string.
launcher() {  # $1 name, $2 command
  cat > "logs/$1.sh" <<LAUNCH
#!/usr/bin/env bash
set -o pipefail
cd "$(pwd)"
source '$VENV/bin/activate'
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' HF_HOME='${HF_HOME:-}'
exec $2 2>&1 | tee logs/$1.log
LAUNCH
  chmod +x "logs/$1.sh"
}

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
if [ "$FINDIT_VLM_BACKEND" = "openai" ] && ! curl -sf "$VLM_HEALTH" >/dev/null; then
  tmux kill-session -t findit-vllm 2>/dev/null || true
  launcher vllm "vllm serve '$FINDIT_VLM_MODEL' --port $FINDIT_VLM_PORT --host 127.0.0.1 --dtype $FINDIT_VLM_DTYPE --max-model-len $FINDIT_VLM_MAX_MODEL_LEN --limit-mm-per-prompt '{\"image\":2}' --gpu-memory-utilization $FINDIT_VLM_GPU_MEMORY_UTILIZATION"
  tmux new-session -d -s findit-vllm "$(pwd)/logs/vllm.sh"
  echo -n "waiting for vLLM (first start takes a few minutes)"
  for _ in $(seq 1 120); do curl -sf "$VLM_HEALTH" >/dev/null && break; echo -n "."; sleep 5; done; echo
  curl -sf "$VLM_HEALTH" >/dev/null || { echo "vLLM did not come up; see logs/vllm.log" >&2; exit 1; }
fi
echo "VLM backend: $FINDIT_VLM_BACKEND ($FINDIT_VLM_MODEL, $FINDIT_VLM_DTYPE)"

tmux kill-session -t findit-api 2>/dev/null || true
launcher api "uvicorn server.app:app --host $FINDIT_SERVER_HOST --port $FINDIT_SERVER_PORT"
tmux new-session -d -s findit-api "$(pwd)/logs/api.sh"
echo -n "waiting for the API (loads the models; a few minutes on first start)"
for _ in $(seq 1 150); do
  curl -s -o /dev/null "http://127.0.0.1:${FINDIT_SERVER_PORT}/" && break
  tmux has-session -t findit-api 2>/dev/null || { echo; echo "API exited. Last log lines:"; tail -n 30 logs/api.log; exit 1; }
  echo -n "."; sleep 4
done

echo
echo "findit demo:  http://${FINDIT_SERVER_PUBLIC_HOSTNAME:-$(hostname -f)}:${FINDIT_SERVER_PORT}/"
echo "logs:         bash scripts/csail_run.sh logs"
echo "stop:         bash scripts/csail_run.sh stop"
