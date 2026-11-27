"""End-to-end test with the mock models on a synthetic clip.

A red blob ("sunglasses" in the mock detector) sits top-left for 3 s, vanishes,
then reappears bottom-right for the last 2 s. A green blob ("keys") stays put.
The query for sunglasses must return the bottom-right, later sighting.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import config
from server.models import MockDescriber, MockDetector, MockEmbedder
from server.pipeline import Models, answer_query, process_clip
from server.store import Store


def make_clip(path: Path, fps: int = 10, seconds: int = 6) -> None:
    w, h = 320, 240
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(fps * seconds):
        t = i / fps
        frame = np.full((h, w, 3), 230, np.uint8)
        cv2.rectangle(frame, (20, 200), (60, 225), (0, 200, 0), -1)  # green "keys", static
        if t < 3:
            cv2.rectangle(frame, (30, 30), (80, 60), (0, 0, 220), -1)  # red "sunglasses" top-left
        elif t >= 4:
            cv2.rectangle(frame, (240, 170), (290, 200), (0, 0, 220), -1)  # ... bottom-right
        out.write(frame)
    out.release()


@pytest.fixture
def models():
    return Models(MockDetector(), MockEmbedder(), MockDescriber())


def test_last_location_is_latest_sighting(tmp_path, models):
    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    store = Store(tmp_path / "mem.sqlite")
    store.create_job("j1", 0.0)
    n = process_clip(clip, "j1", models, store, delete_after=True)
    assert not clip.exists(), "clip must be deleted after processing"
    # The sunglasses re-appear elsewhere with the same appearance, so they are
    # linked to the same track and only its last location moves.
    labels = sorted(r.label for r in store.tracks("j1"))
    assert labels == ["keys", "sunglasses"], labels
    assert n == 2

    res = answer_query("j1", "brown sunglasses", models, store)
    assert res["found"]
    assert res["label"] == "sunglasses"
    assert res["last_seen_s"] >= 4.0
    row = next(r for r in store.tracks("j1") if r.id == res["track_id"])
    assert row.last_box[0] > 200  # bottom-right placement
    assert res["window_s"] == [res["last_seen_s"] - 3.0, res["last_seen_s"] + 3.0]
    assert any(n.startswith("keys to the left") for n in res["nearby"]), res["nearby"]
    assert "Last seen around" in res["summary"] and "keys to the left" in res["summary"]

    res = answer_query("j1", "my keys", models, store)
    assert res["found"] and res["label"] == "keys"

    # scene notes were written every 5 s (clip is 6 s long: notes at 0 s and 5 s)
    notes = store.scene_notes("j1")
    assert [round(t) for t, _ in notes] == [0, 5], notes

    from server.pipeline import ask, memory_context
    ctx = memory_context("j1", store)
    assert "#1 sunglasses" in ctx or "#2 sunglasses" in ctx
    assert "Scene notes over time" in ctx
    a = ask("j1", "what was near the keys?", models, store)
    assert a["answer"] and len(a["used_tracks"]) == 2


def test_http_roundtrip(tmp_path, monkeypatch, models):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "mem.sqlite")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    import importlib
    from server import app as app_module
    importlib.reload(app_module)

    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    with TestClient(app_module.app) as client:
        r = client.post("/api/upload", files={"file": ("clip.mp4", clip.read_bytes(), "video/mp4")})
        assert r.status_code == 200, r.text
        job = r.json()["job_id"]
        # a second upload queues behind the first (one mock worker)
        r2 = client.post("/api/upload", files={"file": ("clip2.mp4", clip.read_bytes(), "video/mp4")})
        job2 = r2.json()["job_id"]
        app_module.pool.wait_idle()
        assert client.get(f"/api/jobs/{job}").json()["status"] == "done"
        assert client.get(f"/api/jobs/{job2}").json()["status"] == "done"
        assert not list((tmp_path / "uploads").iterdir()), "upload must be deleted"
        tracks = client.get(f"/api/jobs/{job}/tracks").json()
        assert len(tracks) == 2
        a = client.post("/api/query", json={"job_id": job, "text": "sunglasses"}).json()
        assert a["found"] and a["last_seen_s"] >= 4.0
        t = client.get(a["thumb_url"])
        assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
        q = client.post("/api/ask", json={"job_id": job, "question": "what did you see?"}).json()
        assert q["answer"] and len(q["thumb_urls"]) == 2
        assert len(client.get(f"/api/jobs/{job}/notes").json()) == 2
        assert client.get("/").status_code == 200


def test_password(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "mem.sqlite")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    from server.passwd import hash_password, verify_password, write_hash
    h = hash_password("hunter22", iterations=1000)
    cfg = tmp_path / "c.toml"
    cfg.write_text("[server]\nport = 8000\n\n[vlm]\nmodel = \"x\"\n")
    write_hash(cfg, 'password_hash = "A"')
    write_hash(cfg, 'password_hash = "B"')            # replaces, does not duplicate
    assert cfg.read_text() == "[server]\npassword_hash = \"B\"\nport = 8000\n\n[vlm]\nmodel = \"x\"\n"
    write_hash(tmp_path / "new.toml", 'password_hash = "C"')   # no [server] yet
    assert "[server]\npassword_hash = \"C\"" in (tmp_path / "new.toml").read_text()
    assert h.startswith("pbkdf2_sha256$1000$") and verify_password("hunter22", h) and not verify_password("x", h)
    monkeypatch.setattr(config, "PASSWORD_HASH", h)
    import importlib
    from server import app as app_module
    importlib.reload(app_module)
    with TestClient(app_module.app) as client:
        r = client.get("/")
        assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic")
        assert client.get("/api/jobs/x").status_code == 401
        assert client.get("/", auth=("anyone", "wrong")).status_code == 401
        assert client.get("/", auth=("anyone", "hunter22")).status_code == 200
        assert client.get("/api/jobs/x", auth=("anyone", "hunter22")).status_code == 404


def test_merge_across_labels_and_masked_crop():
    from server.models import Detection, merge_across_labels
    from server.pipeline import masked_frame

    a = Detection("sunglasses", 0.9, (10, 10, 50, 30))
    b = Detection("glasses", 0.6, (11, 10, 50, 31))      # same region, other prompt
    c = Detection("keys", 0.7, (100, 100, 140, 120))
    kept = merge_across_labels([b, a, c], iou_thr=0.7)
    assert [d.label for d in kept] == ["sunglasses", "keys"]

    frame = np.full((40, 60, 3), 200, np.uint8)
    mask = np.zeros((40, 60), bool); mask[10:30, 10:50] = True
    out = masked_frame(frame, mask, dilate_px=0)
    assert out[20, 20].tolist() == [200, 200, 200] and out[5, 5].tolist() == [128, 128, 128]
    assert masked_frame(frame, None) is frame
