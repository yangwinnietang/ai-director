from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.detector import YoloDetector, load_image
from app.face_recognition import (
    FaceRecognitionService,
    candidate_to_dict,
    draw_face_match,
    draw_faces,
    match_to_dict,
    registered_to_dict,
)
from app.labels import resolve_target_classes


OUTPUT_DIR = Path("images") / "output"

app = FastAPI(title="Director Vision Tool", version="0.1.0")
detector = YoloDetector(
    model_path=settings.yolo_model,
    confidence=settings.yolo_confidence,
    image_size=settings.yolo_image_size,
    end2end=settings.yolo_end2end,
)
face_service = FaceRecognitionService(
    registry_path=settings.face_registry_path,
    model_name=settings.face_model,
)


class CameraAimRequest(BaseModel):
    camera_index: int = 0
    target: str
    tolerance_ratio: float = 0.08


class FaceRegisterRequest(BaseModel):
    identity: str
    face_id: str
    threshold: float = settings.face_threshold
    source_image: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.yolo_model, "face_model": settings.face_model}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> list[dict]:
    image = await _read_upload(file)
    return [asdict(detection) for detection in detector.detect(image)]


@app.post("/robot/detect/image")
async def robot_detect_image(
    file: UploadFile = File(...),
    target: str | None = Form(None),
) -> dict:
    image = await _read_upload(file)
    detections = detector.detect(image)
    if target:
        accepted_labels = set(resolve_target_classes(target))
        detections = [detection for detection in detections if detection.label in accepted_labels]

    return {
        "object_count": len(detections),
        "objects": [_robot_object_from_detection(detection) for detection in detections],
    }


@app.post("/aim/image")
async def aim_image(
    target: str = Form(...),
    tolerance_ratio: float = Form(0.08),
    file: UploadFile = File(...),
) -> dict:
    image = await _read_upload(file)
    return asdict(
        detector.aim_at(
            image,
            target=target,
            tolerance_ratio=tolerance_ratio,
            diagnostic_confidence=settings.yolo_diagnostic_confidence,
        )
    )


@app.post("/aim/camera")
def aim_camera(request: CameraAimRequest) -> dict:
    capture = cv2.VideoCapture(request.camera_index)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok:
        raise HTTPException(status_code=500, detail="Could not read camera frame")

    return asdict(
        detector.aim_at(
            frame,
            target=request.target,
            tolerance_ratio=request.tolerance_ratio,
            diagnostic_confidence=settings.yolo_diagnostic_confidence,
        )
    )


@app.post("/faces/candidates")
async def face_candidates(
    file: UploadFile = File(...),
    annotate: bool = Form(False),
    output_name: str | None = Form(None),
) -> dict:
    image = await _read_upload(file)
    try:
        candidates = face_service.detect_faces(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    output_path = None
    if annotate:
        output_path = _write_output_image(
            draw_faces(image, candidates),
            output_name or _default_output_name(file.filename, "faces"),
        )

    return {
        "face_count": len(candidates),
        "faces": [candidate_to_dict(candidate) for candidate in candidates],
        "output_path": output_path,
    }


@app.post("/faces/register")
def register_face(request: FaceRegisterRequest) -> dict:
    try:
        registered = face_service.register_candidate(
            identity=request.identity,
            face_id=request.face_id,
            threshold=request.threshold,
            source_image=request.source_image,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return registered_to_dict(registered)


@app.post("/faces/recognize")
async def recognize_face(
    identity: str = Form(...),
    threshold: float | None = Form(None),
    annotate: bool = Form(False),
    output_name: str | None = Form(None),
    file: UploadFile = File(...),
) -> dict:
    image = await _read_upload(file)
    try:
        match = face_service.recognize_identity(
            image=image,
            identity=identity,
            threshold=threshold,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    output_path = None
    if annotate:
        output_path = _write_output_image(
            draw_face_match(image, match),
            output_name or _default_output_name(file.filename, identity),
        )

    result = match_to_dict(match)
    result["output_path"] = output_path
    return result


@app.get("/faces/identities")
def face_identities() -> dict:
    return {"identities": face_service.identities()}


async def _read_upload(file: UploadFile):
    suffix = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return load_image(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _write_output_image(image, output_name: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / Path(output_name).name
    if not cv2.imwrite(str(output_path), image):
        raise HTTPException(status_code=500, detail=f"Could not write output image: {output_path}")
    return str(output_path)


def _default_output_name(filename: str | None, suffix: str) -> str:
    image_path = Path(filename or "upload.jpg")
    image_suffix = image_path.suffix or ".jpg"
    return f"{image_path.stem}_{suffix}{image_suffix}"


def _robot_object_from_detection(detection) -> dict:
    x1, y1, x2, y2 = detection.box
    return {
        "object_name": detection.label,
        "confidence": detection.confidence,
        "top_left": {"x": x1, "y": y1},
        "bottom_right": {"x": x2, "y": y2},
    }
