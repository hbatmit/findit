# findit demo

Upload a clip on a web page → a server processes it into object-location
embeddings → ask "where are my brown sunglasses?" and get the last known
location. The clip is deleted once processed; only per-object embeddings, a
one-sentence location, timestamps, and a small object-only crop are kept.

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

3. Copy `config.example.toml` to `config.toml`, set `detector.backend`,
   `embedder.backend`, `vlm.url` and `vlm.model`, then start the server:

   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```

   `detector.backend = "yoloe"` is much faster than OWLv2 but pulls in
   ultralytics (AGPL-3.0).

4. Expose it: CSAIL machines are reachable over the CSAIL VPN; for off-campus
   viewers put nginx with TLS in front, or use a Tailscale/Cloudflare tunnel.
   Keep the `/api/upload` size limit in nginx (`client_max_body_size 500m`).

## Configuration

Everything lives in `config.toml` (start from `config.example.toml`; the file
is gitignored). With no file, built-in defaults run the mock models. Any key
can be overridden by an env var named `FINDIT_<SECTION>_<KEY>`, for example
`FINDIT_DETECTOR_BACKEND=mock` or `FINDIT_VLM_URL=`. `python -m server.config`
prints the resolved settings; `--shell` prints them as export lines, which is
how the scripts read them.

| Section | Keys |
|---|---|
| `server` | `host`, `port`, `public_hostname` (for the printed URL), `data_dir`, `thumb_size` (0 stores no pixels) |
| `detector` | `backend` (mock, owlv2, yoloe), model names, `threshold`, `vocab` |
| `embedder` | `backend` (mock, siglip2), `siglip2_model` |
| `vlm` | `backend` (mock, transformers, openai), `model`, `dtype`, and for openai: `url`, `port`, `gpu_memory_utilization`, `max_model_len` |
| `pipeline` | `sample_fps`, `track_sim`, `track_iou`, `query_min_sim` |
| `deploy` | `gpu` (nvidia-smi index, -1 for all), `venv`, `hf_home` (used by the scripts only) |

## Layout

```
demo/
  server/config.py     loads config.toml (+ env overrides, built-in defaults)
  config.example.toml  template for config.toml
  scripts/             setup and run scripts for a GPU host
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

## GPU host (e.g. a CSAIL machine)

```bash
ssh <gpu-host>
git clone -b claude/object-location-tracking-arch-5c03bb https://github.com/hbatmit/findit
cd findit/demo
cp config.example.toml config.toml   # set deploy.gpu, deploy.venv/hf_home (local disk), server.public_hostname
bash scripts/csail_setup.sh          # venv, deps, model download into /scratch/$USER/hf when available
bash scripts/csail_run.sh            # API (and vLLM if vlm.backend = "openai") in tmux; prints the URL
```

The page is served at `http://<server.public_hostname>:<server.port>/`, reachable
from the CSAIL network or VPN. `bash scripts/csail_run.sh logs` tails both
services; `stop` shuts them down. For a GPU under ~14 GB set
`vlm.model = "Qwen/Qwen3-VL-2B-Instruct"`; on a shared GPU lower
`vlm.gpu_memory_utilization`.
