"""Model wrappers behind three tiny interfaces:

    Detector.detect(frame_bgr) -> list[Detection]
    Embedder.embed_images(crops_bgr) -> np.ndarray [n, d]   (L2-normalised)
    Embedder.embed_text(texts)      -> np.ndarray [n, d]   (L2-normalised)
    Describer.describe(crop_bgr, context_bgr, label) -> str

Each has a mock implementation so the whole demo runs without a GPU, and a
real implementation that is only imported when selected. Swap models by
changing the factory functions at the bottom.
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


# --------------------------------------------------------------------------- mock


class MockDetector:
    """Finds saturated coloured blobs and labels them by hue. Good enough to
    exercise the pipeline on synthetic clips (see tests/)."""

    HUES = [("sunglasses", 0, 15), ("book", 15, 40), ("keys", 40, 80),
            ("mug", 80, 130), ("headphones", 130, 180)]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 80), (180, 255, 255))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 200:
                continue
            hue = int(np.median(hsv[y:y + h, x:x + w, 0]))
            label = next((name for name, lo, hi in self.HUES if lo <= hue < hi), "object")
            out.append(Detection(label, 0.9, (int(x), int(y), int(x + w), int(y + h))))
        return out


class MockEmbedder:
    """Deterministic pseudo-embeddings: mean colour + label hash for images,
    keyword hash for text, arranged so that text and image for the same label
    land near each other. Dimension 64."""

    DIM = 64

    @staticmethod
    def _label_vec(label: str) -> np.ndarray:
        h = hashlib.sha256(label.lower().encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return rng.standard_normal(MockEmbedder.DIM)

    def embed_images(self, crops: list[np.ndarray], labels: list[str] | None = None) -> np.ndarray:
        labels = labels or ["object"] * len(crops)
        vecs = []
        for crop, label in zip(crops, labels):
            v = self._label_vec(label).copy()
            mean = crop.reshape(-1, 3).mean(0) / 255.0  # BGR
            v[:3] += mean * 2.0  # colour nudge so "brown" vs "black" differs a bit
            vecs.append(v)
        return _l2(np.stack(vecs)) if vecs else np.zeros((0, self.DIM), np.float32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            words = t.lower().replace(",", " ").split()
            # Use the longest word as the "label"; crude but fine for the mock.
            label = max(words, key=len) if words else "object"
            vecs.append(self._label_vec(label))
        return _l2(np.stack(vecs))


class MockDescriber:
    def describe(self, crop: np.ndarray, context: np.ndarray, label: str) -> str:
        h, w = context.shape[:2]
        ch, cw = crop.shape[:2]
        return f"a {label} resting on a surface (mock description, crop {cw}x{ch} in a {w}x{h} context)"


# --------------------------------------------------------------------------- real


class Owlv2Detector:
    """OWLv2 from transformers. Slow (~0.5-1 s/frame on a modest GPU) but zero
    setup and Apache-2.0. Fine for a 1 fps demo."""

    def __init__(self, vocab: list[str], threshold: float):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        name = "google/owlv2-base-patch16-ensemble"
        self.processor = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to(self.device).eval()
        self.vocab = vocab
        self.threshold = threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        from PIL import Image

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=[self.vocab], images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target = self.torch.tensor([[img.height, img.width]], device=self.device)
        res = self.processor.post_process_object_detection(outputs, threshold=self.threshold, target_sizes=target)[0]
        out = []
        for score, lab, box in zip(res["scores"], res["labels"], res["boxes"]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            out.append(Detection(self.vocab[int(lab)], float(score), (x1, y1, x2, y2)))
        return out


class YoloeDetector:
    """YOLOE via ultralytics (AGPL-3.0; check licensing before productising).
    Much faster than OWLv2."""

    def __init__(self, vocab: list[str], threshold: float):
        from ultralytics import YOLOE

        self.model = YOLOE("yoloe-11s-seg.pt")
        self.model.set_classes(vocab, self.model.get_text_pe(vocab))
        self.threshold = threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, conf=self.threshold, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            out.append(Detection(res.names[int(b.cls)], float(b.conf), (x1, y1, x2, y2)))
        return out


class Siglip2Embedder:
    """SigLIP2 base: one model for both crop and query embeddings."""

    def __init__(self, name: str = "google/siglip2-base-patch16-224"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def embed_images(self, crops: list[np.ndarray], labels: list[str] | None = None) -> np.ndarray:
        from PIL import Image

        if not crops:
            return np.zeros((0, self.model.config.projection_dim), np.float32)
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return _l2(feats.float().cpu().numpy())

    def embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, padding="max_length", max_length=64,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return _l2(feats.float().cpu().numpy())


class OpenAIVlmDescriber:
    """Talks to any OpenAI-compatible multimodal endpoint (vLLM serving
    Qwen3-VL, llama.cpp server, ...). Sends the object crop and a context crop,
    never the full frame."""

    PROMPT = (
        "Image 1 is a close-up of a {label}. Image 2 shows its surroundings. "
        "In one short sentence, describe where the {label} is located relative to "
        "nearby furniture or objects (for example: 'on the wooden desk, left of the laptop'). "
        "Mention its colour. Do not describe people."
    )

    def __init__(self, url: str, model: str):
        import httpx

        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=120)
        self.model = model

    @staticmethod
    def _data_url(img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    def describe(self, crop: np.ndarray, context: np.ndarray, label: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": 80,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._data_url(crop)}},
                    {"type": "image_url", "image_url": {"url": self._data_url(context)}},
                    {"type": "text", "text": self.PROMPT.format(label=label)},
                ],
            }],
        }
        r = self.client.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- factories


def _l2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def make_detector():
    if config.DETECTOR == "owlv2":
        return Owlv2Detector(config.VOCAB, config.DET_THRESHOLD)
    if config.DETECTOR == "yoloe":
        return YoloeDetector(config.VOCAB, config.DET_THRESHOLD)
    return MockDetector()


def make_embedder():
    if config.EMBEDDER == "siglip2":
        return Siglip2Embedder()
    return MockEmbedder()


def make_describer():
    if config.VLM_URL:
        return OpenAIVlmDescriber(config.VLM_URL, config.VLM_MODEL)
    return MockDescriber()
