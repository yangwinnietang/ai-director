from __future__ import annotations

import os
import tempfile
from dataclasses import asdict

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.detector import YoloDetector, load_image


MODEL_PATH = os.getenv("YOLO_MODEL", "yolo26s.pt")
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.2"))
IMAGE_SIZE = int(os.getenv("YOLO_IMGSZ")) if os.getenv("YOLO_IMGSZ") else None
END2END = os.getenv("YOLO_END2END")
END2END_FLAG = None if END2END is None else END2END.lower() in {"1", "true", "yes", "on"}
DIAGNOSTIC_CONFIDENCE = (
    float(os.getenv("YOLO_DIAGNOSTIC_CONFIDENCE")) if os.getenv("YOLO_DIAGNOSTIC_CONFIDENCE") else None
)

app = FastAPI(title="Director Vision Tool", version="0.1.0")
detector = YoloDetector(
    model_path=MODEL_PATH,
    confidence=CONFIDENCE,
    image_size=IMAGE_SIZE,
    end2end=END2END_FLAG,
)


class CameraAimRequest(BaseModel):
    camera_index: int = 0
    target: str
    tolerance_ratio: float = 0.08


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> list[dict]:
    image = await _read_upload(file)
    return [asdict(detection) for detection in detector.detect(image)]


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
            diagnostic_confidence=DIAGNOSTIC_CONFIDENCE,
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
            diagnostic_confidence=DIAGNOSTIC_CONFIDENCE,
        )
    )


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
