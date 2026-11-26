# findit — Architecture and Implementation Strategy

Private, edge-only "where did I leave my X?" system for homes and offices.
Static per-room cameras plus mobile cameras (phone, smart glasses) feed a local
perception pipeline that remembers the **last known location of objects as
embeddings and metadata only**. No video, no frames, no face data ever touch disk.

This document proposes the architecture, model choices, privacy design, and a
phased implementation plan. Model names and sizes are current as of late 2026
but move fast; the *roles* in the pipeline are the stable part, and every model
is behind a swappable interface.

---

## 1. Goals and constraints

| Requirement | Design consequence |
|---|---|
| Track last known location of everyday objects (keys, sunglasses, earbuds, books, scissors, …), small and large | Open-vocabulary detection + instance-level re-identification; event-based "placed / picked up" memory, not continuous tracking |
| Indoors, occasionally garage / backyard | Room-level + zone-level localization ("kitchen counter, left of the toaster"), metric 3D only where a mobile device supplies pose |
| Mixed cameras: static per-room, phone, smart glasses | One perception pipeline, two ingestion modes: continuous (static) and session-based egocentric (mobile) |
| All inference on-prem; V1 on a user-controlled server, later distributed across camera GPUs/NPUs | Stage-based pipeline where every stage is a separately deployable service with an ONNX/TensorRT/Hailo export path |
| **No video stored, ever** | Frames exist only in RAM on the processing node; the persistence layer physically cannot accept pixel data (type-level enforcement, see §7) |
| Lightweight models | Nothing above ~4B parameters anywhere; the per-frame path is ≤ 50M parameters total; the VLM is asynchronous and called a few times per minute at most |

---

## 2. Design principles

1. **Perceive, then forget.** Every frame is processed into a handful of small
   records (detections, crop embeddings, a short caption) and discarded. The
   system's memory is a table of *object observations*, not a recording.
2. **Event-driven, not frame-driven.** A home is static 99% of the time. Cheap
   motion gating decides when the detector runs; the detector decides when the
   embedder runs; the embedder decides when the VLM runs. Heavy models see a few
   crops per minute, not 30 frames per second.
3. **Two-tier object identity.** Tier 1 is the *class* ("sunglasses") from an
   open-vocabulary detector. Tier 2 is the *instance* ("Hari's black Ray-Bans")
   from an appearance embedding matched against enrolled prototypes. The system
   answers queries at whichever tier it is confident in.
4. **Placement is the event that matters.** "Last known location" changes when
   an object is *put down*, not while it is carried. Hand–object interaction
   (a hand releases an object and the object stays still) is the primary trigger
   on egocentric cameras; "object appears and stays stationary" is the trigger
   on static cameras.
5. **Location is semantic first, metric second.** Users want "on the hallway
   shelf", not coordinates. Zones (named polygons per static camera) and
   VLM-generated relational descriptions are the V1 location representation;
   metric 3D is layered on where a mobile device provides pose.
6. **People are never remembered.** Person detections are used transiently to
   attribute actions to "a hand" and to mask people out before embedding.
   No person crop, no face, no body embedding is ever written.
7. **Distribute by stage and by stream, not by tensor.** When compute moves onto
   the cameras, each camera runs the per-frame stages for its own stream and
   publishes small events; the VLM runs on whichever node(s) have the memory.
   Splitting one model's layers across Wi-Fi is a last resort (see §10).

---

## 3. System overview

```
                 ┌────────────────────────────────────────────────────────────┐
                 │                       HOME LAN (no WAN)                   │
                 │                                                            │
  Static cams    │   ┌──────────────┐   ┌──────────────┐                      │
  (RTSP/PoE) ────┼──▶│  Ingest      │──▶│  Perception  │──┐                   │
                 │   │  go2rtc /    │   │  Worker(s)   │  │  small events     │
  Phone app ─────┼──▶│  WebRTC /    │──▶│  (per stream)│  │  (JSON + vectors) │
  (ARKit/ARCore) │   │  gRPC stream │   │              │  │                   │
                 │   └──────────────┘   └──────────────┘  ▼                   │
  Glasses ──▶ phone  (frames never     motion → detect  ┌──────────────────┐  │
  (Meta DAT /       leave RAM)         → track → HOI    │  Event Bus       │  │
   other SDK)                          → embed → zone   │  (NATS / MQTT)   │  │
                 │                                      └────────┬─────────┘  │
                 │                                               │            │
                 │          ┌────────────────────┐   ┌───────────▼──────────┐ │
                 │          │  VLM Worker        │◀──│  Object Memory       │ │
                 │          │  (async, low-prio) │──▶│  Service             │ │
                 │          │  captions, relations│   │  identity matching,  │ │
                 │          │  query understanding│   │  placement state,    │ │
                 │          └────────────────────┘   │  SQLite + vector idx │ │
                 │                                   └───────────┬──────────┘ │
                 │                                               │            │
                 │   ┌───────────────┐   ┌──────────────┐   ┌────▼─────────┐  │
                 │   │  Voice (STT)  │──▶│  Query API   │◀──│  Web / app / │  │
                 │   │  whisper.cpp  │   │  (local LLM) │   │  Home Asst.  │  │
                 │   └───────────────┘   └──────────────┘   └──────────────┘  │
                 └────────────────────────────────────────────────────────────┘
```

Services (V1 all on one server; later spread across nodes):

| Service | Responsibility | Stateful? |
|---|---|---|
| **Ingest** | Pull RTSP / receive phone & glasses streams, decode to RAM frames, fan out to a perception worker per stream | No |
| **Perception worker** | Motion gate → open-vocab detector → tracker → hand–object interaction → person masking → crop embeddings → zone assignment → emit events | No (per-stream tracker state in RAM only) |
| **Event bus** | Transport for small events (≤ a few KB each) between services and nodes | No |
| **Object Memory** | Identity resolution (which known object is this?), placement state machine, last-known-location table, retention policy | Yes (the only persistent store) |
| **VLM worker** | Asynchronous: caption crops, describe spatial relations, disambiguate near-duplicate instances, parse natural-language queries | No |
| **Query API** | "Where are my keys?" → object resolution → answer with location, time, confidence, description | No |
| **Calibration / admin UI** | Name rooms, draw zones per static camera, enroll objects, set retention, privacy switches | Yes (config) |

---

## 4. Perception pipeline (per stream)

Each stage is a pure function `frame/crops → events`. Stages run in a
single process per stream in V1 (shared GPU), and become per-device
deployables later.

| # | Stage | Runs at | Model (primary) | Params | Notes |
|---|---|---|---|---|---|
| 0 | **Decode** | native fps (static: 10–15 fps at 1080p–4K; mobile: 5–10 fps) | — | — | Hardware decode (NVDEC / V4L2). Frame buffers are RAM-only, `mlock`ed, never written to disk. |
| 1 | **Motion / change gate** | every frame | OpenCV MOG2 or frame-diff on 320px downscale | 0 | Emits ROI boxes of change. Static cams spend >95% of time here. Hand-sized motion = candidate placement. |
| 2 | **Open-vocabulary detector** | on motion, ≤ 5 fps; plus a periodic full-frame sweep every ~30 s per static cam | **YOLOE-26-S** (or YOLOE-v8s/11s) in text-prompt mode with the user's object vocabulary + `person`, `hand` | ~10–15M | Runs on the full frame at 640–1280 px; for small objects on 4K static cams, additionally on a 2× tile around motion ROIs (SAHI-style) only when a hand was present. Alternatives: OV-DEIM (NMS-free, Apache-2.0), MM-Grounding-DINO-T, OWLv2-base (slow but accurate; good for the periodic sweep). |
| 3 | **Tracker** | with detector | ByteTrack (BoT-SORT if ReID needed) | 0 | Produces short tracklets so one placement yields one event, not 50. |
| 4 | **Hand–object interaction (HOI)** | when `hand` present | Hand boxes from stage 2 + contact heuristic (IoU + motion coherence); optional MediaPipe Hands for 21-pt hands on egocentric streams | ~1–4M | State per object tracklet: `free → held → free`. `held → free` with object stationary for N frames ⇒ **PLACED**. `free → held` then object leaves frame ⇒ **TAKEN** (object becomes "in transit"). |
| 5 | **Person masking** | before any embedding | `person` boxes from stage 2 | 0 | Pixels inside person boxes (minus the hand region touching the object) are zeroed before cropping/embedding. Person boxes are never emitted beyond this stage. |
| 6 | **Object crop embedding** | on new tracklet, on PLACED, and on periodic re-confirmation (~once/min per stationary object) | **MobileCLIP2-S2** (text-alignable, 512-d) **and** **DINOv2 ViT-S/14** or **DINOv3 ViT-S** (instance-discriminative, 384-d) | 36M + 21M | Two embeddings per crop: CLIP-style for open-text queries, DINO-style for "is this *my* pair of sunglasses" matching. Crops are padded square, 224 px. |
| 7 | **Scene / room context** | on PLACED and every ~5 min per static cam | Global MobileCLIP2 image embedding of the *person-masked* frame (static cams: optional; mobile: required for room recognition) | shared | kNN against per-room calibration embeddings answers "which room is this mobile camera in?" Upgrade path: a VPR model (MixVPR / SALAD / BoQ) if kNN on CLIP is not robust enough. |
| 8 | **Zone / position** | on PLACED | Static cams: point-in-polygon against user-drawn zones. Mobile: ARKit/ARCore 6-DoF pose + **Depth Anything V2 Small** (25M) for a metric-ish point; falls back to room + relational description | 25M (mobile only) | Position record = `(room, zone, optional xyz, camera_id)`. |
| 9 | **VLM enrichment** (async) | on PLACED, on new unknown object, on ambiguous identity; budget ~1–5 calls/min server-wide | **Qwen3-VL-2B-Instruct** (4-bit, ~1.5 GB) on the server; **MiniCPM-V 4.6 (1.3B)** / **SmolVLM2-2.2B** / **FastVLM-1.5B** as alternatives; Qwen3-VL-4B if the server has ≥ 8 GB VRAM and you want better spatial reasoning | 1.3–4B | Inputs: the object crop plus a person-masked, downscaled context crop (not the full frame). Outputs: a short attribute caption ("black wayfarer sunglasses, folded"), a relational location phrase ("on the kitchen counter, left of the toaster, next to a blue mug"), and a container judgment ("placed inside the top drawer" ⇒ object is hidden, location = container). The context crop is discarded after the call. |

Emitted events (all small; no pixels):

```jsonc
{ "type": "PLACED",
  "t": "2026-11-22T18:04:11Z",
  "camera": "kitchen-1",            // or "phone:hari" / "glasses:hari"
  "tracklet": "kitchen-1/8812",
  "label_candidates": [["sunglasses", 0.91], ["glasses", 0.07]],
  "bbox": [812, 440, 905, 488],       // pixels; kept for zone lookup, then dropped
  "zone": {"room": "kitchen", "zone": "counter-left", "xyz": null},
  "emb_clip": "<512 x int8>", "emb_dino": "<384 x int8>",
  "held_by": "hand",                  // never a person identity
  "caption": null                     // filled later by VLM worker
}
```

---

## 5. Object identity and memory

### 5.1 Identity resolution

The Object Memory service maintains a set of **object identities**, each with:

- label(s) and a human name (`"sunglasses"` / `"Hari's black Ray-Bans"`)
- a small bank of prototype embeddings (up to ~10 per identity, both CLIP- and DINO-style), updated online with exponential decay
- enrolled (user confirmed) vs. auto-discovered
- optional owner and "important" flag (affects retention and notifications)

Resolution of an incoming crop:

1. Candidate set = identities whose label matches the detector label (loose: synonyms via CLIP text similarity).
2. Score = `0.6 · cos(dino, prototypes) + 0.4 · cos(clip, prototypes)`; thresholds tuned per class (keys are hard; books are easy).
3. Above the high threshold ⇒ match. Between thresholds ⇒ queue to the VLM worker for a disambiguation call ("Is the object in image A the same as in image B? Reply with yes/no and the distinguishing feature"). Below ⇒ new auto-discovered identity.
4. Multi-camera merging: identities seen within a short window from different static cameras with high embedding similarity are merged (same object moved rooms).

**Enrollment** (optional but recommended for the user's top ~20 items): the user
photographs an object from 3–5 angles in the app; the embeddings become seed
prototypes. The app can also propose auto-discovered identities for naming
("I've seen this object 14 times — what is it?") using the VLM caption.

### 5.2 Placement state machine (per identity)

```
          seen stationary / PLACED
   ┌─────────────────────────────────┐
   ▼                                 │
AT_LOCATION ──TAKEN (hand)──▶ IN_TRANSIT ──PLACED──▶ AT_LOCATION (new place)
   │                                 │
   │ not seen where expected for T   │ not seen for T
   ▼                                 ▼
MISSING_FROM_LAST_LOCATION      UNKNOWN (last known = last PLACED)
```

The **last known location** is the most recent `PLACED` record. `TAKEN` does
not erase it; it marks it as stale so the answer can say "it was on the
counter at 6:04 pm, but someone picked it up at 6:30 and I haven't seen it
since." Static cams also emit `CONFIRMED_STILL_THERE` on the periodic sweep,
which refreshes the timestamp without creating a new placement.

Containers: when the VLM judges that the object went into a drawer, bag, or
cabinet, the record stores `inside: "kitchen top drawer"` and the state stays
`AT_LOCATION` even though the object is no longer visible.

### 5.3 Storage

SQLite (SQLCipher for encryption at rest) with a vector index
(`sqlite-vec`; LanceDB or Qdrant-embedded are drop-in alternatives). Tables:

| Table | Key columns |
|---|---|
| `object_identity` | id, labels[], name, enrolled, owner, importance, created_at |
| `prototype` | identity_id, emb_clip (int8[512]), emb_dino (int8[384]), weight, updated_at |
| `observation` | id, identity_id, t, camera_id, room, zone, xyz?, inside?, caption, confidence, event_type, emb_clip, emb_dino |
| `last_location` (materialized) | identity_id, t_placed, t_last_confirmed, room, zone, xyz?, inside?, caption, state |
| `zone` | camera_id, room, name, polygon |
| `room_prototype` | room, emb_clip (for mobile room recognition) |
| `thumbnail` (**off by default**) | identity_id, 64×64 JPEG of the *object crop only*, encrypted — see §7 |

Retention: raw `observation` rows older than N days (default 30) are deleted;
`last_location` and prototypes persist. Everything is a few KB per event, so a
busy household produces megabytes per month, not gigabytes.

---

## 6. Localization details

**Static cameras.** During setup the user draws named zones on a still from
each camera (the still is shown live and not saved). Pixel bbox → zone is a
point-in-polygon test on the bottom-center of the box. For a better answer than
"counter-left", the VLM relational phrase is attached. Optional one-time metric
calibration (a printed marker or a known-size object) lets the system estimate
a floor-plane homography so zones across two cameras of the same room can be
aligned; this is not needed for V1.

**Phone.** ARKit / ARCore world tracking gives 6-DoF pose within a session;
Depth Anything V2 Small (or the platform's LiDAR/ToF where present) gives the
object's depth along the ray, so the placement becomes a 3D point in session
coordinates. Cross-session persistence without the cloud: on iOS use
`ARWorldMap` saved locally (encrypted); on Android avoid Cloud Anchors and
instead rely on room recognition (stage 7) plus the relational description.
The system should always be able to answer at room + relational-phrase
granularity even if metric relocalization fails.

**Smart glasses.** Meta's Wearables Device Access Toolkit (1.0 as of Connect
2026) streams camera frames from Ray-Ban glasses to a companion iOS/Android app;
apps do not run on the glasses. Our phone app runs stages 1–6 on the phone NPU
(Core ML / LiteRT) and forwards only events + embeddings to the server over the
LAN (or over WireGuard/Tailscale when away, still to the home server). Frames
never leave the phone. Other glasses with raw camera access (Mentra, Vuzix,
Project Aria for research) plug into the same phone-side pipeline. Glasses have
no reliable pose, so localization is room recognition + relational phrase, plus
"the last static camera that saw this object" as a cross-check.

**Garage / backyard.** Same as a static room with weatherproof cameras; the
detector vocabulary is extended with outdoor items (garden tools, bikes).
Lighting variation argues for the periodic re-confirmation sweep to run more
often there.

---

## 7. Privacy architecture

**What is stored:** object-crop embeddings (int8), short text captions, labels,
timestamps, camera id, zone/room, optional xyz, optional encrypted 64×64
object-only thumbnail (off by default). Nothing else.

**What is never stored:** full frames, scene crops, any pixels of people,
person/face/body embeddings, audio, per-person activity logs. Person boxes
exist only inside the perception process and are masked before any embedding.

**Enforcement, not policy:**

- The persistence layer's schema has no column that can hold an image, and the
  `ObjectMemory` client API accepts only typed `Event` structs whose fields are
  scalars, strings ≤ 512 chars, and fixed-length int8 vectors. Writing pixels is
  a type error, not a configuration mistake.
- Frame buffers are allocated from a pinned, `mlock`ed pool; swap is disabled on
  the server; `/dev/shm` is used for inter-process frame hand-off and is
  tmpfs (RAM).
- Debug frame dumps are compiled out of release builds (`FINDIT_DEV_FRAMES`
  build flag); the CI privacy test runs the pipeline on a synthetic stream and
  asserts that no file on the filesystem grew by more than event-sized bytes
  and that no JPEG/PNG/H.264 magic bytes appear anywhere under the data dir.
- All inter-node traffic is on the LAN with mutual TLS; there is no outbound
  WAN rule in the server's firewall profile. Remote access (phone away from
  home) is via a user-run WireGuard/Tailscale tunnel to the home server.
- A physical/UI "pause" per camera and global; while paused, ingest is
  disconnected (not just ignored).

**Residual risk to state clearly to users:** CLIP-style embeddings leak some
information (approximate inversion attacks exist). Mitigations in order of
cost: embed only tight object crops (not scenes), quantize to int8, encrypt the
database at rest with a key held on the server, and offer a "DINO-only" mode
(self-supervised features are harder to invert and sufficient for enrolled
objects, at the cost of open-text queries). The optional thumbnail is the one
knob that stores real pixels; it is object-only, 64×64, encrypted, and the UI
says so.

**Development exception:** building and evaluating this system needs recorded
test sequences. Those live only on developers' machines, come from consenting
staged scenes or public egocentric datasets (Ego4D VQ2D, EPIC-KITCHENS VISOR),
and never pass through the product's ingest path. Keep that boundary explicit
in the repo layout (`dev-data/` is gitignored and the product binary cannot read
it).

---

## 8. Model selection summary

Criteria: ≤ a few tens of M params on the per-frame path, open weights,
exportable to ONNX/TensorRT/Core ML/Hailo, and a licence compatible with a
product (flagged below; verify before shipping).

| Role | Primary | Size | Alternatives | Licence notes |
|---|---|---|---|---|
| Motion gate | OpenCV MOG2 / frame diff | 0 | — | — |
| Open-vocab detection | **YOLOE-26-S** (text/visual prompts, prompt-free mode) | ~10–15M | OV-DEIM-S (Apache-2.0, NMS-free), MM-Grounding-DINO-T, OWLv2-base for the slow periodic sweep | YOLOE/Ultralytics lineage is AGPL-3.0 — needs an Ultralytics commercial licence or use OV-DEIM / MM-GDINO instead |
| Tracking | ByteTrack | 0 | BoT-SORT (+ReID) | MIT |
| Hands / HOI | detector `hand` class + contact heuristic | 0 | MediaPipe Hands (egocentric), EgoHOS-style contact model if heuristics fall short | Apache-2.0 |
| Object crop embedding (text-alignable) | **MobileCLIP2-S2** | 36M img enc. | MobileCLIP2-S0 (11M) on phones; SigLIP2-B/16 (Apache-2.0) on the server | Apple sample-code licence — verify |
| Object crop embedding (instance) | **DINOv2 ViT-S/14** | 21M | DINOv3 ViT-S (stronger retrieval; custom licence) | DINOv2 Apache-2.0 |
| Room recognition | kNN on MobileCLIP2 global embedding | shared | MixPVR / SALAD / BoQ | — |
| Depth (mobile only) | **Depth Anything V2 Small** | 25M | platform LiDAR/ToF | Small variant is Apache-2.0 (B/L are CC-BY-NC) |
| Optional segmentation for cleaner crops | MobileSAM | 10M | SAM 2.1 tiny; SAM 3 if a bigger server is available | Apache-2.0 |
| VLM (server) | **Qwen3-VL-2B-Instruct**, Q4 | 2B | Qwen3-VL-4B, MiniCPM-V 4.6 (1.3B), SmolVLM2-2.2B, Gemma 3n E2B/E4B, LFM2-VL-1.6B | Qwen3-VL Apache-2.0; others vary |
| VLM (phone, optional) | FastVLM-0.5B/1.5B or SmolVLM2-500M | ≤ 1.5B | MiniCPM-V 4.6 | — |
| Query NLU / answer phrasing | same VLM in text mode | — | Qwen3-1.7B text-only | — |
| Speech | whisper.cpp small / Moonshine | ≤ 250M | Home Assistant Assist pipeline | MIT |

Serving: TensorRT (NVIDIA) or ONNX Runtime / OpenVINO for stages 1–8;
llama.cpp (GGUF) or vLLM for the VLM. On Apple Silicon servers use MLX for the
VLM and Core ML for the rest.

Rough budget on a single RTX 3060-class GPU: 6 static cams at 1080p with
motion gating + 5 fps detection ≈ 20–30% GPU; embeddings negligible; Qwen3-VL-2B
Q4 ≈ 1.5 GB VRAM, ~0.5–1 s per enrichment call. A Jetson Orin NX (16 GB)
handles 3–4 cams plus the 2B VLM.

---

## 9. Hardware

**V1 server (user-controlled):** mini PC with an NVIDIA GPU (8–12 GB) or a
Jetson Orin NX/AGX, or a Mac mini (M-series). 16 GB RAM, swap disabled, NVMe
only for the encrypted DB and models.

**Static cameras:** any RTSP/ONVIF PoE camera (1080p is enough for placement
events; 4K helps small items on wide views). Mount at ~1.5–2 m looking across
surfaces, not down from ceiling corners — a 45° view of counters and tables
dramatically improves small-object detection and hand visibility.

**Edge nodes for the distributed phase:** Raspberry Pi 5 + Camera Module + AI
HAT+ (Hailo-8L, 13 TOPS) runs stages 1–6 at ~10 fps; Jetson Orin Nano (8 GB)
runs stages 1–8 and can host the 2B VLM; reCamera / Axera / RK3588 boards are
viable for stages 1–6.

**Mobile:** iPhone (ARKit, Neural Engine, Core ML) and Android (ARCore, NNAPI /
LiteRT). Glasses via Meta DAT (Ray-Ban) or vendor SDKs.

---

## 10. Distributed inference (phase 3+)

Goal: no central server required; cameras collectively run the system.

1. **Stage-level distribution (first step).** Each camera node runs stages 0–8
   for its own stream and publishes events on the bus. Object Memory becomes a
   small replicated service (SQLite with Litestream-style replication to 2–3
   nodes, or a single elected leader via NATS JetStream). The VLM runs on the
   node(s) with enough memory; a scheduler routes enrichment jobs by queue
   depth. This alone removes the GPU server for most homes if at least one
   node is Jetson-class.
2. **Work-stealing for the VLM.** Enrichment calls are independent, latency-
   tolerant (seconds), and small (one crop + one context crop ≈ 100–300 KB).
   A NATS work queue with multiple VLM consumers on heterogeneous nodes is
   enough. Nodes advertise capability (VRAM, TOPS, supported models) via mDNS.
3. **Model sharding only if needed.** If no single node can hold the VLM,
   pipeline-parallel sharding across 2–3 Jetson-class nodes via llama.cpp RPC is
   workable for prefill-heavy, low-rate workloads (per-layer activation for a
   ~1k-token image prompt is a few MB over Wi-Fi 6). Tensor-parallel is not
   viable over Wi-Fi. NPU-only nodes (Hailo, Coral) cannot run transformer
   decoders and stay on stages 0–8.
4. **Degradation.** If the VLM is unreachable, the system still records
   placements with labels + zones; captions are back-filled when it returns.
   Nothing on the per-frame path depends on the VLM.

The service boundaries in §3 are chosen so that steps 1–3 require no change to
the perception code — only to deployment and the bus topology.

---

## 11. Implementation roadmap

### Phase 0 — Harness and data (2–3 weeks)
- Repo layout: `ingest/`, `perception/` (one module per stage with a common
  `Stage` interface and ONNX export), `memory/`, `vlm/`, `query/`, `app/`,
  `calibration/`, `dev-data/` (gitignored).
- Dev-only evaluation set: 20–30 staged sequences (place/take objects in 3
  rooms, 2 static cams + 1 phone), plus Ego4D VQ2D clips for egocentric.
  Annotate placement events (object, time, zone).
- Metrics: placement recall/precision, zone accuracy, identity confusion rate,
  time-to-answer, false "missing" rate, GPU utilization per camera.
- Privacy CI test from §7 from day one.

### Phase 1 — V1 single server, static cameras + phone (6–8 weeks)
- Ingest via go2rtc (RTSP → in-process frames) and a phone app that streams
  WebRTC to the server (frames in RAM only).
- Stages 1–8 with the primary models; VLM worker with Qwen3-VL-2B via llama.cpp.
- Object Memory with SQLCipher + sqlite-vec; placement state machine; retention.
- Calibration UI (rooms, zones, enrollment) and query UI (text + voice via
  whisper.cpp); Home Assistant integration over MQTT discovery so "where are
  my keys" works through Assist.
- Exit criteria on the eval set: ≥ 90% placement recall for enrolled objects
  ≥ 5 cm, ≥ 95% correct room, ≥ 85% correct zone, < 5% identity confusion
  among enrolled objects.

### Phase 2 — Egocentric and hands (4–6 weeks)
- Phone-side pipeline (stages 1–6 on Core ML / LiteRT) so the phone sends
  events, not frames; same code path serves glasses via Meta DAT.
- Hand–object contact model if the heuristic is < 85% precision on VQ2D-style
  clips; MediaPipe Hands on egocentric streams.
- ARKit/ARCore pose + depth → xyz; local ARWorldMap persistence on iOS.
- Cross-camera identity merging and the "taken by hand, last seen heading to
  the hallway" answer style.

### Phase 3 — Edge nodes (6–10 weeks)
- Export stages 1–8 to Hailo (HEF) and TensorRT; a `findit-node` image for
  Pi 5 + AI HAT and Jetson Orin Nano.
- NATS bus with mTLS, mDNS discovery, replicated Object Memory, VLM work queue.
- Field test: 4–6 nodes, no central server, one Jetson hosting the VLM.

### Phase 4 — Fully distributed VLM and polish
- llama.cpp RPC sharding across Jetson nodes where no single node fits the VLM.
- Online prototype refinement, user feedback loop ("that's not my keys"),
  notification rules ("tell me if my keys leave the house without me").

---

## 12. Known hard problems and mitigations

| Problem | Mitigation |
|---|---|
| Small objects (keys, earbuds) at distance on static cams | Camera placement at surface level; tiled inference around motion ROIs when a hand is present; rely on the *placement moment* (object is large in frame near the hand) rather than steady-state detection; egocentric camera sees the object up close |
| Object hidden in a container or under something | VLM container judgment on the context crop at placement; the answer reports "inside the top drawer (placed 6:04 pm)" |
| Instance confusion (two black phones, two sets of keys) | Enrollment prototypes; DINO embeddings; VLM pairwise disambiguation; ask the user once and store the answer as a prototype |
| Identity drift as objects change state (folded vs. open sunglasses, book front vs. spine) | Multi-prototype banks per identity with online update; label-tier answer when instance-tier is uncertain |
| Lighting changes, night | Periodic re-confirmation sweeps; IR-capable cameras; embeddings computed on normalized crops |
| Multi-person households, attributions | By design no person identity; location is what matters. "Who moved it" is out of scope and stays so |
| False "missing" alarms | Require missed re-confirmation across ≥ 2 sweeps and no `TAKEN` event before marking `MISSING_FROM_LAST_LOCATION` |
| Bandwidth / compute at scale | Motion gating; per-node perception in phase 3; VLM budgeted at a few calls/min |
| Licensing of YOLO-family models | Keep the detector interface model-agnostic; OV-DEIM / MM-GDINO as Apache-2.0 fallbacks |

---

## 13. Open decisions

1. Thumbnail storage default (off, recommended) vs. on-with-encryption for a
   better "is this it?" confirmation UX.
2. Whether to build ingest on Frigate (mature camera handling, Hailo/TensorRT
   detector abstraction) with recording hard-disabled, or keep a minimal
   in-house ingest for a stronger no-storage guarantee. Recommendation:
   in-house ingest using go2rtc, borrowing Frigate's motion-mask and detector
   plugin patterns.
3. DINOv2 (Apache) vs. DINOv3 (better, custom licence) for instance embeddings.
4. Single-vendor edge node (Jetson Orin Nano everywhere; simplest software) vs.
   mixed Pi + Hailo nodes (cheaper; two export toolchains).

---

## References (checked Nov 2026)

- YOLOE: Real-Time Seeing Anything — https://arxiv.org/abs/2503.07465
- YOLOE-26 (YOLO26 + YOLOE) — https://arxiv.org/abs/2602.00168
- YOLO26 analysis — https://arxiv.org/abs/2601.12882
- OV-DEIM (DETR-style open-vocab, NMS-free) — https://arxiv.org/abs/2603.07022
- MobileCLIP2 — https://arxiv.org/abs/2508.20691
- ILIAS instance-level retrieval benchmark (DINOv3 vs SigLIP numbers) — https://vrg.fel.cvut.cz/ilias/
- Object-to-image ratio in instance retrieval (why tight crops matter) — https://arxiv.org/abs/2610.11489
- MiniCPM-V 4.6 (1.3B, phone-targeted) — https://rits.shanghai.nyu.edu/ai/minicpm-v-4-6-a-1-3b-multimodal-model-built-for-phones/
- Meta Wearables Device Access Toolkit — https://developers.meta.com/blog/introducing-meta-wearables-device-access-toolkit/ and https://wearables.developer.meta.com/docs/develop/dat/build-overview/
- Multimodal RAG / embedding leakage evaluation — https://arxiv.org/abs/2601.17644
