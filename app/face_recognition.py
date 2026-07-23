from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import time
from typing import Iterable

import cv2
import numpy as np


DEFAULT_FACE_THRESHOLD = 0.45


@dataclass(frozen=True)
class FaceCandidate:
    face_id: str
    box: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass(frozen=True)
class RegisteredFace:
    identity: str
    embedding: np.ndarray
    threshold: float
    source_image: str | None
    source_face_id: str | None
    created_at: float


@dataclass(frozen=True)
class FaceMatch:
    found: bool
    identity: str
    threshold: float
    similarity: float | None
    face_id: str | None
    box: tuple[int, int, int, int] | None
    frame_size: tuple[int, int]
    offset: tuple[float, float] | None
    offset_ratio: tuple[float, float] | None
    match_count: int
    face_count: int
    miss_reason: str | None


class FaceRecognitionService:
    def __init__(
        self,
        registry_path: str | Path = Path("data") / "face_registry.json",
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
        candidate_ttl_seconds: int = 600,
    ):
        self.registry_path = Path(registry_path)
        self.model_name = model_name
        self.providers = providers or ["CPUExecutionProvider"]
        self.det_size = det_size
        self.candidate_ttl_seconds = candidate_ttl_seconds
        self._app = None
        self._lock = Lock()
        self._candidates: dict[str, tuple[float, FaceCandidate]] = {}

    def detect_faces(self, image: np.ndarray) -> list[FaceCandidate]:
        self._prune_candidates()
        height, width = image.shape[:2]
        faces = self._face_app().get(image)
        candidates: list[FaceCandidate] = []
        for face in faces:
            embedding = _normalize(np.asarray(face.embedding, dtype=np.float32))
            x1, y1, x2, y2 = [int(round(v)) for v in face.bbox.tolist()]
            x1, y1, x2, y2 = _clamp_box((x1, y1, x2, y2), width=width, height=height)
            candidate = FaceCandidate(
                face_id=str(uuid.uuid4()),
                box=(x1, y1, x2, y2),
                confidence=float(face.det_score),
                embedding=embedding,
            )
            self._candidates[candidate.face_id] = (time(), candidate)
            candidates.append(candidate)
        return candidates

    def register_candidate(
        self,
        identity: str,
        face_id: str,
        threshold: float = DEFAULT_FACE_THRESHOLD,
        source_image: str | None = None,
    ) -> RegisteredFace:
        identity = identity.strip()
        if not identity:
            raise ValueError("identity is required")

        self._prune_candidates()
        candidate_entry = self._candidates.get(face_id)
        if candidate_entry is None:
            raise KeyError(f"Unknown or expired face_id: {face_id}")

        _, candidate = candidate_entry
        registered = RegisteredFace(
            identity=identity,
            embedding=candidate.embedding,
            threshold=threshold,
            source_image=source_image,
            source_face_id=face_id,
            created_at=time(),
        )
        registry = self._load_registry()
        registry[identity] = registered
        self._save_registry(registry)
        return registered

    def recognize_identity(
        self,
        image: np.ndarray,
        identity: str,
        threshold: float | None = None,
    ) -> FaceMatch:
        height, width = image.shape[:2]
        registry = self._load_registry()
        registered = registry.get(identity)
        if registered is None:
            return FaceMatch(
                found=False,
                identity=identity,
                threshold=threshold or DEFAULT_FACE_THRESHOLD,
                similarity=None,
                face_id=None,
                box=None,
                frame_size=(width, height),
                offset=None,
                offset_ratio=None,
                match_count=0,
                face_count=0,
                miss_reason="identity_not_registered",
            )

        effective_threshold = threshold if threshold is not None else registered.threshold
        candidates = self.detect_faces(image)
        scored = [
            (candidate, cosine_similarity(candidate.embedding, registered.embedding))
            for candidate in candidates
        ]
        matches = [(candidate, score) for candidate, score in scored if score >= effective_threshold]
        if not matches:
            best_similarity = max((score for _, score in scored), default=None)
            return FaceMatch(
                found=False,
                identity=identity,
                threshold=effective_threshold,
                similarity=best_similarity,
                face_id=None,
                box=None,
                frame_size=(width, height),
                offset=None,
                offset_ratio=None,
                match_count=0,
                face_count=len(candidates),
                miss_reason="no_faces" if not candidates else "no_identity_match",
            )

        best, best_similarity = max(matches, key=lambda item: item[1])
        center_x, center_y = best.center
        offset_x = center_x - width / 2
        offset_y = center_y - height / 2
        return FaceMatch(
            found=True,
            identity=identity,
            threshold=effective_threshold,
            similarity=best_similarity,
            face_id=best.face_id,
            box=best.box,
            frame_size=(width, height),
            offset=(offset_x, offset_y),
            offset_ratio=(offset_x / width, offset_y / height),
            match_count=len(matches),
            face_count=len(candidates),
            miss_reason=None,
        )

    def identities(self) -> list[dict]:
        registry = self._load_registry()
        return [
            {
                "identity": face.identity,
                "threshold": face.threshold,
                "source_image": face.source_image,
                "source_face_id": face.source_face_id,
                "created_at": face.created_at,
            }
            for face in registry.values()
        ]

    def _face_app(self):
        if self._app is None:
            with self._lock:
                if self._app is None:
                    try:
                        from insightface.app import FaceAnalysis
                    except ImportError as exc:
                        raise RuntimeError(
                            "InsightFace is not installed. Run: pip install -r requirements.txt"
                        ) from exc

                    app = FaceAnalysis(name=self.model_name, providers=self.providers)
                    app.prepare(ctx_id=0, det_size=self.det_size)
                    self._app = app
        return self._app

    def _load_registry(self) -> dict[str, RegisteredFace]:
        if not self.registry_path.exists():
            return {}
        with self.registry_path.open("r", encoding="utf-8") as registry_file:
            raw = json.load(registry_file)

        registry: dict[str, RegisteredFace] = {}
        for identity, payload in raw.get("identities", {}).items():
            registry[identity] = RegisteredFace(
                identity=identity,
                embedding=_normalize(_decode_embedding(payload["embedding"])),
                threshold=float(payload.get("threshold", DEFAULT_FACE_THRESHOLD)),
                source_image=payload.get("source_image"),
                source_face_id=payload.get("source_face_id"),
                created_at=float(payload.get("created_at", 0)),
            )
        return registry

    def _save_registry(self, registry: dict[str, RegisteredFace]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "model": self.model_name,
            "identities": {
                identity: {
                    "embedding": _encode_embedding(face.embedding),
                    "threshold": face.threshold,
                    "source_image": face.source_image,
                    "source_face_id": face.source_face_id,
                    "created_at": face.created_at,
                }
                for identity, face in registry.items()
            },
        }
        with self.registry_path.open("w", encoding="utf-8") as registry_file:
            json.dump(payload, registry_file, ensure_ascii=False, indent=2)

    def _prune_candidates(self) -> None:
        now = time()
        expired = [
            face_id
            for face_id, (created_at, _) in self._candidates.items()
            if now - created_at > self.candidate_ttl_seconds
        ]
        for face_id in expired:
            self._candidates.pop(face_id, None)


def draw_faces(
    image: np.ndarray,
    faces: Iterable[FaceCandidate],
    label: str = "face",
) -> np.ndarray:
    output = image.copy()
    for index, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = face.box
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 180, 255), 2)
        text = f"{label}_{index} {face.confidence:.2f}"
        cv2.putText(output, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 180, 255), 2)
    return output


def draw_face_match(image: np.ndarray, match: FaceMatch) -> np.ndarray:
    output = image.copy()
    if not match.found or match.box is None:
        return output
    x1, y1, x2, y2 = match.box
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 80), 2)
    similarity = 0.0 if match.similarity is None else match.similarity
    text = f"{match.identity} {similarity:.2f}"
    cv2.putText(output, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
    return output


def candidate_to_dict(candidate: FaceCandidate) -> dict:
    return {
        "face_id": candidate.face_id,
        "box": candidate.box,
        "confidence": candidate.confidence,
        "center": candidate.center,
    }


def registered_to_dict(face: RegisteredFace) -> dict:
    return {
        "identity": face.identity,
        "threshold": face.threshold,
        "source_image": face.source_image,
        "source_face_id": face.source_face_id,
        "created_at": face.created_at,
    }


def match_to_dict(match: FaceMatch) -> dict:
    return {
        "found": match.found,
        "identity": match.identity,
        "threshold": match.threshold,
        "similarity": match.similarity,
        "face_id": match.face_id,
        "box": match.box,
        "frame_size": match.frame_size,
        "offset": match.offset,
        "offset_ratio": match.offset_ratio,
        "match_count": match.match_count,
        "face_count": match.face_count,
        "miss_reason": match.miss_reason,
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = _normalize(left)
    right = _normalize(right)
    return float(np.dot(left, right))


def _normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)


def _encode_embedding(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode("ascii")


def _decode_embedding(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value.encode("ascii")), dtype=np.float32)


def _clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )
