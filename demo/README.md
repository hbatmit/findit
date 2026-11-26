# findit demo

Upload a clip on a web page → a server processes it into object-location
embeddings → ask "where are my brown sunglasses?" and get the last known
location. The clip is deleted once processed; only per-object embeddings, a
one-sentence location, timestamps and a small object-only crop are kept.

```
browser ──upload──▶ FastAPI ──▶ pipeline: sample 1 fps → detect → embed crops
                      │                   → link tracks → VLM describes last sighting
                      │                   → SQLite (embeddings + text + thumb), clip deleted
        ◀──query────  └──▶ text embedding → cosine search → latest matching track
```

## Run locally with mock models (no GPU)

```bash
cd demo
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn server.app:app --reload --port 8000
```

Open http://localhost:8000. The mock detector only finds saturated coloured
blobs, so use `tests/test_pipeline.py::make_clip` to generate a clip, or move
straight to real models.

Command-line, no web:

```bash
python -m server.pipeline path/to/clip.mp4 "brown sunglasses" "keys"
```

## Run on the CSAIL GPU server with real models

1. Install the model stack:

   ```bash
   pip install -r requirements.txt torch torchvision transformers pillow vllm
   ```

2. Start the VLM behind an OpenAI-compatible endpoint (one GPU, ~10 GB):

   ```bash
   vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8001 --max-model-len 8192 --limit-mm-per-prompt image=2
   ```

   Smaller option: `Qwen/Qwen3-VL-2B-Instruct`, or llama.cpp's `llama-server`
   with a GGUF build and `--mmproj`.

3. Start the demo server with real backends:

   ```bash
   FINDIT_DETECTOR=owlv2 FINDIT_EMBEDDER=siglip2 \
   FINDIT_VLM_URL=http://127.0.0.1:8001/v1 FINDIT_VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct \
   FINDIT_VOCAB="sunglasses,keys,headphones,earbuds,book,scissors,phone,wallet,mug,remote control" \
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```

   `FINDIT_DETECTOR=yoloe` is much faster than OWLv2 but pulls in ultralytics
   (AGPL-3.0).

4. Expose it: CSAIL machines are reachable over the CSAIL VPN; for off-campus
   viewers put nginx with TLS in front, or use a Tailscale/Cloudflare tunnel.
   Keep the `/api/upload` size limit in nginx (`client_max_body_size 500m`).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `FINDIT_DETECTOR` | `mock` | `mock`, `owlv2`, `yoloe` |
| `FINDIT_EMBEDDER` | `mock` | `mock`, `siglip2` |
| `FINDIT_VLM_URL` | empty (mock) | OpenAI-compatible base URL, e.g. `http://127.0.0.1:8001/v1` |
| `FINDIT_VLM_MODEL` | `Qwen/Qwen3-VL-4B-Instruct` | model name sent to the endpoint |
| `FINDIT_VOCAB` | common household items | comma-separated detector prompts |
| `FINDIT_FPS` | `1.0` | frame sampling rate |
| `FINDIT_DET_THRESHOLD` | `0.25` | detector confidence |
| `FINDIT_TRACK_SIM` / `FINDIT_TRACK_IOU` | `0.80` / `0.10` | track linking thresholds |
| `FINDIT_QUERY_MIN_SIM` | `0.05` | minimum query similarity to answer |
| `FINDIT_THUMB` | `96` | thumbnail size in px; `0` disables storing any pixels |
| `FINDIT_DATA_DIR` | `./data` | SQLite + transient upload dir |

## Layout

```
demo/
  server/config.py     env-driven settings
  server/models.py     Detector / Embedder / Describer interfaces, mock + real backends
  server/pipeline.py   clip → tracks → store; query answering; CLI entry point
  server/store.py      SQLite object memory (no column can hold a frame)
  server/app.py        FastAPI routes + static page
  web/index.html       upload, poll, query UI
  tests/               synthetic-clip end-to-end tests
```

## What is deliberately simple here

- Tracking is greedy label + embedding + IoU linking at 1 fps; no ByteTrack.
- No person masking before embedding (add before showing real household clips).
- Search is a numpy cosine over one job's tracks; swap in sqlite-vec when
  multiple clips accumulate.
- One worker thread; one clip processed at a time.

See `../docs/ARCHITECTURE.md` for where this grows into the full system.

## CSAIL GPU host (mapmaker-gpu2)

```bash
ssh mapmaker-gpu2.csail.mit.edu
git clone -b claude/object-location-tracking-arch-5c03bb https://github.com/hbatmit/findit
cd findit/demo
bash scripts/csail_setup.sh      # venv, deps, model download (~10 min, ~15 GB in /scratch/$USER/hf)
bash scripts/csail_run.sh        # vLLM + API in tmux; prints the URL
```

Then open http://mapmaker-gpu2.csail.mit.edu:8000/ from the CSAIL network or VPN.
`bash scripts/csail_run.sh logs` tails both services; `stop` shuts them down.
Setup picks Qwen3-VL-4B if the GPU has ≥ 14 GB, else the 2B model; override
with `FINDIT_VLM_MODEL=...` before running setup. If the GPU is shared, lower
`--gpu-memory-utilization` in `scripts/csail_run.sh`.
