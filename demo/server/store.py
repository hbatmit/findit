"""Object memory for the demo: SQLite holding one row per object track.

Only embeddings, text, timestamps, boxes, and a tiny object-only crop are
stored. The schema has no place for a frame; the pipeline deletes the uploaded
clip once processing finishes.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS job (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,            -- queued | processing | done | error
    message TEXT,
    frames INTEGER DEFAULT 0,
    tracks INTEGER DEFAULT 0,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS track (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    label TEXT NOT NULL,
    first_t REAL NOT NULL,           -- seconds into the clip
    last_t REAL NOT NULL,
    last_box TEXT NOT NULL,          -- "x1,y1,x2,y2"
    n_obs INTEGER NOT NULL,
    embedding BLOB NOT NULL,         -- float32[d], L2-normalised mean of crop embeddings
    dim INTEGER NOT NULL,
    location TEXT,                   -- VLM one-sentence description
    thumb BLOB,                      -- small JPEG of the object crop only (optional)
    neighbors TEXT                   -- JSON: other objects in the last frame, relative position
);
CREATE INDEX IF NOT EXISTS track_job ON track(job_id);
CREATE TABLE IF NOT EXISTS scene_note (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    t REAL NOT NULL,                 -- seconds into the clip
    text TEXT NOT NULL               -- VLM one-line description of the scene (no people)
);
CREATE INDEX IF NOT EXISTS scene_note_job ON scene_note(job_id);
"""


@dataclass
class TrackRow:
    id: int
    job_id: str
    label: str
    first_t: float
    last_t: float
    last_box: tuple[int, int, int, int]
    n_obs: int
    embedding: np.ndarray
    location: str | None
    thumb: bytes | None
    neighbors: str | None = None


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(track)")}
        if "neighbors" not in cols:  # databases created before this column existed
            self._conn.execute("ALTER TABLE track ADD COLUMN neighbors TEXT")
            self._conn.commit()

    # ---- jobs
    def create_job(self, job_id: str, created_at: float) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO job(id,status,created_at) VALUES(?,?,?)",
                               (job_id, "queued", created_at))
            self._conn.commit()

    def update_job(self, job_id: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE job SET {cols} WHERE id=?", (*fields.values(), job_id))
            self._conn.commit()

    def get_job(self, job_id: str) -> dict | None:
        cur = self._conn.execute("SELECT id,status,message,frames,tracks,created_at FROM job WHERE id=?", (job_id,))
        row = cur.fetchone()
        if not row:
            return None
        job = dict(zip(["id", "status", "message", "frames", "tracks", "created_at"], row))
        if job["status"] == "queued":
            job["ahead"] = self._conn.execute(
                "SELECT COUNT(*) FROM job WHERE status='queued' AND created_at<?", (job["created_at"],)).fetchone()[0]
        del job["created_at"]
        return job

    def requeue_interrupted(self) -> None:
        """Jobs left in 'processing' by a previous server run can never finish."""
        with self._lock:
            self._conn.execute("UPDATE job SET status='error', message='server restarted during processing' WHERE status='processing'")
            self._conn.commit()

    # ---- tracks
    def add_track(self, job_id: str, label: str, first_t: float, last_t: float,
                  last_box: tuple[int, int, int, int], n_obs: int, embedding: np.ndarray,
                  location: str | None, thumb: bytes | None, neighbors: str | None = None) -> int:
        emb = np.asarray(embedding, np.float32)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO track(job_id,label,first_t,last_t,last_box,n_obs,embedding,dim,location,thumb,neighbors)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, label, first_t, last_t, ",".join(map(str, last_box)), n_obs,
                 emb.tobytes(), emb.shape[0], location, thumb, neighbors))
            self._conn.commit()
            return int(cur.lastrowid)

    def tracks(self, job_id: str) -> list[TrackRow]:
        cur = self._conn.execute(
            "SELECT id,job_id,label,first_t,last_t,last_box,n_obs,embedding,dim,location,thumb,neighbors"
            " FROM track WHERE job_id=? ORDER BY last_t", (job_id,))
        out = []
        for r in cur.fetchall():
            emb = np.frombuffer(r[7], np.float32, count=r[8])
            out.append(TrackRow(r[0], r[1], r[2], r[3], r[4], tuple(int(v) for v in r[5].split(",")),
                                r[6], emb, r[9], r[10], r[11]))
        return out

    def thumb(self, track_id: int) -> bytes | None:
        row = self._conn.execute("SELECT thumb FROM track WHERE id=?", (track_id,)).fetchone()
        return row[0] if row else None

    # ---- scene notes
    def add_scene_note(self, job_id: str, t: float, text: str) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO scene_note(job_id,t,text) VALUES(?,?,?)", (job_id, t, text))
            self._conn.commit()

    def scene_notes(self, job_id: str) -> list[tuple[float, str]]:
        cur = self._conn.execute("SELECT t,text FROM scene_note WHERE job_id=? ORDER BY t", (job_id,))
        return [(r[0], r[1]) for r in cur.fetchall()]

    def search(self, job_id: str, query_emb: np.ndarray, k: int = 5) -> list[tuple[TrackRow, float]]:
        """Cosine similarity over all tracks of a job. Fine for demo sizes."""
        rows = self.tracks(job_id)
        if not rows:
            return []
        mat = np.stack([r.embedding for r in rows])
        sims = mat @ np.asarray(query_emb, np.float32)
        order = np.argsort(-sims)[:k]
        return [(rows[i], float(sims[i])) for i in order]
