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

import json
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .models import gpu_devices, make_describer, make_detector, make_embedder
from .pipeline import Models, answer_query, ask, nearby_phrases, process_clip
from .store import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

store = Store(config.DB_PATH)


class ModelSet:
    """One copy of the three models on one device, used by one worker thread.
    `busy` is held while a clip is processed so query routes prefer idle sets."""

    def __init__(self, device: str):
        self.device = device
        self.models = Models(make_detector(device), make_embedder(device), make_describer(device))
        self.busy = threading.Lock()


class Pool:
    """One worker thread per model set, all pulling from one FIFO job queue."""

    def __init__(self):
        self.sets: list[ModelSet] = []
        self.jobs: queue.Queue[tuple[str, Path] | None] = queue.Queue()
        self.threads: list[threading.Thread] = []

    def start(self):
        for dev in gpu_devices():
            ms = ModelSet(dev)
            self.sets.append(ms)
            th = threading.Thread(target=self._worker, args=(ms,), daemon=True, name=f"worker-{dev}")
            th.start()
            self.threads.append(th)

    def submit(self, job_id: str, path: Path):
        self.jobs.put((job_id, path))

    def idle_models(self) -> Models:
        """Models for a query: an idle set if there is one, else the first set
        (the call then shares the GPU with that worker's clip)."""
        for ms in self.sets:
            if ms.busy.acquire(blocking=False):
                ms.busy.release()
                return ms.models
        return self.sets[0].models

    def wait_idle(self, timeout: float = 60.0) -> None:
        """Block until the queue is drained and no worker is busy (tests)."""
        self.jobs.join()
        end = time.time() + timeout
        while time.time() < end:
            if all(ms.busy.acquire(blocking=False) and (ms.busy.release() or True) for ms in self.sets):
                return
            time.sleep(0.05)

    def _worker(self, ms: ModelSet):
        while True:
            item = self.jobs.get()
            if item is None:
                self.jobs.task_done()
                return
            job_id, path = item
            with ms.busy:
                _run_job(job_id, path, ms.models)
            self.jobs.task_done()


pool = Pool()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    store.requeue_interrupted()
    pool.start()
    print(f"findit: {len(pool.sets)} worker(s) on {[ms.device for ms in pool.sets]}", flush=True)
    yield


app = FastAPI(title="findit demo", lifespan=lifespan)


def _run_job(job_id: str, path: Path, models: Models):
    store.update_job(job_id, status="processing")
    try:
        process_clip(path, job_id, models, store,
                     progress=lambda f, t, msg: store.update_job(job_id, frames=f, tracks=t, message=msg))
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
    pool.submit(job_id, dest)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    job["workers"] = len(pool.sets)
    return job


@app.get("/api/jobs/{job_id}/tracks")
def job_tracks(job_id: str):
    if not store.get_job(job_id):
        raise HTTPException(404, "unknown job")
    return [{"track_id": r.id, "label": r.label, "first_seen_s": round(r.first_t, 2),
             "last_seen_s": round(r.last_t, 2), "observations": r.n_obs, "location": r.location,
             "nearby": nearby_phrases(json.loads(r.neighbors) if r.neighbors else []),
             "has_thumb": r.thumb is not None} for r in store.tracks(job_id)]


@app.get("/api/tracks/{track_id}/thumb")
def track_thumb(track_id: int):
    thumb = store.thumb(track_id)
    if thumb is None:
        raise HTTPException(404, "no thumbnail")
    return Response(content=thumb, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/notes")
def job_notes(job_id: str):
    if not store.get_job(job_id):
        raise HTTPException(404, "unknown job")
    return [{"t": round(t, 2), "text": text} for t, text in store.scene_notes(job_id)]


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
    res = answer_query(q.job_id, q.text, pool.idle_models(), store)
    if res.get("found"):
        res["thumb_url"] = f"/api/tracks/{res['track_id']}/thumb"
    return res


class Question(BaseModel):
    job_id: str
    question: str


@app.post("/api/ask")
def ask_route(q: Question):
    job = store.get_job(q.job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    res = ask(q.job_id, q.question.strip(), pool.idle_models(), store)
    res["thumb_urls"] = [f"/api/tracks/{tid}/thumb" for tid in res["used_tracks"]]
    return res


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
