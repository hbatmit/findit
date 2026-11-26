"""Clip -> object memory.

    sample frames  ->  detect  ->  embed crops  ->  link into tracks
                   ->  describe each track's last sighting with the VLM
                   ->  write tracks to the store, delete the clip

Frames only ever live in this process's memory.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config
from .models import Detection
from .store import Store


@dataclass
class Track:
    label: str
    first_t: float
    last_t: float
    last_box: tuple[int, int, int, int]
    embeddings: list[np.ndarray]
    last_crop: np.ndarray
    last_context: np.ndarray
    n_obs: int = 1
    misses: int = 0
    neighbors: list[dict] = field(default_factory=list)   # other objects in the last frame

    @property
    def mean_embedding(self) -> np.ndarray:
        m = np.mean(np.stack(self.embeddings), axis=0)
        return m / max(float(np.linalg.norm(m)), 1e-8)


@dataclass
class Models:
    detector: object
    embedder: object
    describer: object


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def relation(a, b) -> tuple[str, float]:
    """Describe where box b sits relative to box a, in image terms.
    Returns (phrase, distance in units of a's size)."""
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dx, dy = bx - ax, by - ay
    size = max(a[2] - a[0], a[3] - a[1], 1)
    if abs(dx) >= abs(dy):
        phrase = "to the right" if dx > 0 else "to the left"
    else:
        phrase = "below" if dy > 0 else "above"
    return phrase, math.hypot(dx, dy) / size


def neighbors_of(box, others: list[Detection], max_n: int = 4) -> list[dict]:
    out = []
    for o in others:
        phrase, dist = relation(box, o.box)
        out.append({"label": o.label, "rel": phrase, "dist": round(dist, 1)})
    out.sort(key=lambda n: n["dist"])
    return out[:max_n]


def crop(frame: np.ndarray, box, pad: float = 0.0, square: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if square:
        side = max(bw, bh)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, x2 = cx - side / 2, cx + side / 2
        y1, y2 = cy - side / 2, cy + side / 2
        bw = bh = side
    x1, x2 = int(x1 - pad * bw), int(x2 + pad * bw)
    y1, y2 = int(y1 - pad * bh), int(y2 + pad * bh)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
    return frame[y1:y2, x1:x2]


def sample_frames(path: Path, fps: float):
    """Yield (t_seconds, frame_bgr) at roughly `fps` frames per second."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native / fps)))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield i / native, frame
            i += 1
    finally:
        cap.release()


def link(tracks: list[Track], dets: list[Detection], embs: np.ndarray, crops, contexts, t: float,
         sim_thr: float, iou_thr: float) -> None:
    """Greedy association of this frame's detections to open tracks."""
    used = set()
    for i, (det, emb, c, ctx) in enumerate(zip(dets, embs, crops, contexts)):
        nbrs = neighbors_of(det.box, [d for j, d in enumerate(dets) if j != i])
        best, best_score = None, -1.0
        for k, tr in enumerate(tracks):
            if k in used or tr.label != det.label:
                continue
            sim = float(tr.mean_embedding @ emb)
            ov = iou(tr.last_box, det.box)
            if sim >= sim_thr and (ov >= iou_thr or tr.misses > 0):
                score = sim + ov
                if score > best_score:
                    best, best_score = k, score
        if best is None:
            tracks.append(Track(det.label, t, t, det.box, [emb], c, ctx, neighbors=nbrs))
        else:
            used.add(best)
            tr = tracks[best]
            tr.last_t, tr.last_box, tr.n_obs, tr.misses = t, det.box, tr.n_obs + 1, 0
            tr.embeddings.append(emb)
            tr.last_crop, tr.last_context, tr.neighbors = c, ctx, nbrs
    for k, tr in enumerate(tracks):
        if k not in used:
            tr.misses += 1


def thumb_jpeg(img: np.ndarray, size: int) -> bytes:
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def downscale(frame: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return frame if s >= 1 else cv2.resize(frame, (int(w * s), int(h * s)))


def process_clip(path: Path, job_id: str, models: Models, store: Store,
                 delete_after: bool = True, progress=None) -> int:
    """Run the pipeline on one clip. Returns number of tracks written."""
    tracks: list[Track] = []
    n_frames = 0
    last_note_t = -1e9
    for t, frame in sample_frames(path, config.SAMPLE_FPS):
        n_frames += 1
        # Scene notes: a one-line text description of the whole frame every few
        # seconds, so free-form questions have more than object tracks to go on.
        if config.SCENE_NOTE_EVERY_S > 0 and t - last_note_t >= config.SCENE_NOTE_EVERY_S:
            last_note_t = t
            try:
                store.add_scene_note(job_id, t, models.describer.caption_scene(downscale(frame)))
            except Exception as e:  # noqa: BLE001
                store.add_scene_note(job_id, t, f"(no scene note: {e.__class__.__name__})")
        dets = models.detector.detect(frame)
        if dets:
            crops = [crop(frame, d.box, pad=0.1, square=True) for d in dets]
            contexts = [crop(frame, d.box, pad=1.5, square=True) for d in dets]
            embs = models.embedder.embed_images(crops, [d.label for d in dets])
            link(tracks, dets, embs, crops, contexts, t, config.TRACK_SIM, config.TRACK_IOU)
        if progress:
            progress(n_frames, len(tracks), f"sampling frames ({n_frames} so far)")

    # Describe each track's last sighting. This is the only VLM call per object.
    for i, tr in enumerate(tracks, 1):
        if progress:
            progress(n_frames, len(tracks), f"describing object {i} of {len(tracks)} with the VLM")
        try:
            tr_location = models.describer.describe(tr.last_crop, tr.last_context, tr.label)
        except Exception as e:  # VLM down: keep the track, back-fill later
            tr_location = f"(no description: {e.__class__.__name__})"
        store.add_track(job_id, tr.label, tr.first_t, tr.last_t, tr.last_box, tr.n_obs,
                        tr.mean_embedding, tr_location,
                        thumb_jpeg(tr.last_crop, config.THUMB_SIZE) if config.THUMB_SIZE else None,
                        neighbors=json.dumps(tr.neighbors))

    if delete_after:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    store.update_job(job_id, frames=n_frames, tracks=len(tracks), message=None)
    return len(tracks)


TIME_WINDOW_S = 3.0


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def nearby_phrases(neighbors: list[dict]) -> list[str]:
    out = []
    for n in neighbors:
        closeness = "right next to it" if n["dist"] < 1.5 else "nearby" if n["dist"] < 4 else "further away"
        out.append(f"{n['label']} {n['rel']}, {closeness}")
    return out


def answer_query(job_id: str, text: str, models: Models, store: Store, k: int = 5) -> dict:
    """Return the most recently seen track matching the query.

    "Where" in a single clip means: when it was last seen (a +/- window of
    seconds into the clip), what else was around it in that frame (from the
    detector boxes), and the VLM's one-line description of the surroundings."""
    q = models.embedder.embed_text([text])[0]
    hits = store.search(job_id, q, k=k)
    hits = [(r, s) for r, s in hits if s >= config.QUERY_MIN_SIM]
    if not hits:
        return {"found": False, "query": text, "candidates": []}
    # Prefer the latest sighting among strong matches (within 0.05 of the best score).
    best_sim = hits[0][1]
    strong = [h for h in hits if h[1] >= best_sim - 0.05]
    row, sim = max(strong, key=lambda h: h[0].last_t)
    neighbors = json.loads(row.neighbors) if row.neighbors else []
    t0, t1 = max(0.0, row.last_t - TIME_WINDOW_S), row.last_t + TIME_WINDOW_S
    nearby = nearby_phrases(neighbors)
    summary = f"Last seen around {mmss(row.last_t)} (between {mmss(t0)} and {mmss(t1)})."
    if nearby:
        summary += " Nearby: " + "; ".join(nearby) + "."
    if row.location:
        summary += " " + row.location
    return {
        "found": True,
        "query": text,
        "label": row.label,
        "summary": summary,
        "location": row.location,
        "nearby": nearby,
        "neighbors": neighbors,
        "last_seen_s": round(row.last_t, 2),
        "window_s": [round(t0, 2), round(t1, 2)],
        "first_seen_s": round(row.first_t, 2),
        "observations": row.n_obs,
        "similarity": round(sim, 3),
        "track_id": row.id,
        "candidates": [{"track_id": r.id, "label": r.label, "last_seen_s": round(r.last_t, 2),
                        "similarity": round(s, 3)} for r, s in hits],
    }


def memory_context(job_id: str, store: Store) -> str:
    """Everything remembered about a clip, as text for the VLM."""
    rows = store.tracks(job_id)
    notes = store.scene_notes(job_id)
    end = max([r.last_t for r in rows] + [t for t, _ in notes] + [0.0])
    lines = [f"Clip length: about {mmss(end)}. Frames were sampled about once per second.", "",
             "Objects remembered (id, label, when seen, sightings, where, what was nearby):"]
    if not rows:
        lines.append("(none detected)")
    for r in rows:
        nb = nearby_phrases(json.loads(r.neighbors) if r.neighbors else [])
        line = f"#{r.id} {r.label} — seen {mmss(r.first_t)}–{mmss(r.last_t)} ({r.n_obs} sightings)"
        if r.location:
            line += f" — {r.location}"
        if nb:
            line += " — nearby: " + "; ".join(nb)
        lines.append(line)
    if notes:
        lines += ["", "Scene notes over time:"]
        lines += [f"{mmss(t)} — {text}" for t, text in notes]
    return "\n".join(lines)


def ask(job_id: str, question: str, models: Models, store: Store) -> dict:
    """Free-form question about the clip, answered by the VLM from the memory
    (object tracks, scene notes) plus the thumbnails of the most relevant objects."""
    context = memory_context(job_id, store)
    rows = store.tracks(job_id)
    images, used = [], []
    if rows and config.ASK_MAX_IMAGES > 0:
        q = models.embedder.embed_text([question])[0]
        for row, _sim in store.search(job_id, q, k=config.ASK_MAX_IMAGES):
            if row.thumb:
                img = cv2.imdecode(np.frombuffer(row.thumb, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
                    used.append(row.id)
    if used:
        context += "\n\nAttached images, in order: " + ", ".join(
            f"image {i + 1} = object #{tid}" for i, tid in enumerate(used))
    answer = models.describer.answer(question, context, images)
    return {"question": question, "answer": answer, "used_tracks": used}


def run_cli():
    """python -m server.pipeline clip.mp4 "brown sunglasses" """
    import sys
    from .models import make_describer, make_detector, make_embedder

    clip = Path(sys.argv[1])
    queries = sys.argv[2:]
    store = Store(config.DB_PATH)
    models = Models(make_detector(), make_embedder(), make_describer())
    job_id = f"cli-{int(time.time())}"
    store.create_job(job_id, time.time())
    n = process_clip(clip, job_id, models, store, delete_after=False)
    print(f"{n} tracks")
    for r in store.tracks(job_id):
        print(f"  #{r.id} {r.label:12s} {r.first_t:6.1f}s -> {r.last_t:6.1f}s  x{r.n_obs}  {r.location}")
    for q in queries:
        a = answer_query(job_id, q, models, store)
        print(q, "->", a.get("summary", a))


if __name__ == "__main__":
    run_cli()
