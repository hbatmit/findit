"""FastAPI demo server.

    POST /api/upload            multipart clip  -> {job_id}
    GET  /api/jobs/{job_id}     -> {status, frames, tracks}
    GET  /api/jobs/{job_id}/tracks
    GET  /api/tracks/{id}/thumb -> JPEG of the object crop
    POST /api/query             {job_id, text} -> last known location
    GET  /                      the web page

Run:  uvicorn server.app:app --host 0.0.0.0 --port 8000   (from demo/)
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .models import make_describer, make_detector, make_embedder
from .pipeline import Models, answer_query, process_clip
from .store import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

store = Store(config.DB_PATH)
models: Models | None = None
# One worker: the GPU is shared and the demo processes one clip at a time.
executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global models
    models = Models(make_detector(), make_embedder(), make_describer())
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="findit demo", lifespan=lifespan)


def _run_job(job_id: str, path: Path):
    store.update_job(job_id, status="processing")
    try:
        process_clip(path, job_id, models, store,
                     progress=lambda f, t: store.update_job(job_id, frames=f, tracks=t))
        store.update_job(job_id, status="done")
    except Exception as e:  # noqa: BLE001
        store.update_job(job_id, status="error", message=f"{e.__class__.__name__}: {e}")
        path.unlink(missing_ok=True)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("video/") and not file.filename.lower().endswith((".mp4", ".mov", ".webm", ".mkv")):
        raise HTTPException(400, "please upload a video file")
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    dest = config.UPLOAD_DIR / f"{job_id}{suffix}"
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    store.create_job(job_id, time.time())
    executor.submit(_run_job, job_id, dest)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/jobs/{job_id}/tracks")
def job_tracks(job_id: str):
    if not store.get_job(job_id):
        raise HTTPException(404, "unknown job")
    return [{"track_id": r.id, "label": r.label, "first_seen_s": round(r.first_t, 2),
             "last_seen_s": round(r.last_t, 2), "observations": r.n_obs, "location": r.location,
             "has_thumb": r.thumb is not None} for r in store.tracks(job_id)]


@app.get("/api/tracks/{track_id}/thumb")
def track_thumb(track_id: int):
    cur = store._conn.execute("SELECT thumb FROM track WHERE id=?", (track_id,))
    row = cur.fetchone()
    if not row or row[0] is None:
        raise HTTPException(404, "no thumbnail")
    return Response(content=row[0], media_type="image/jpeg")


class Query(BaseModel):
    job_id: str
    text: str


@app.post("/api/query")
def query(q: Query):
    job = store.get_job(q.job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    res = answer_query(q.job_id, q.text, models, store)
    if res.get("found"):
        res["thumb_url"] = f"/api/tracks/{res['track_id']}/thumb"
    return res


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
