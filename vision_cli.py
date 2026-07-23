from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import asdict
import json
import sys

import cv2

from app.detector import YoloDetector, draw_detections, load_image


def main() -> None:
    output_dir = Path("images") / "output"
    parser = argparse.ArgumentParser(description="Run YOLO detection or aiming on one image.")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument(
        "--target",
        nargs="+",
        help="One or more target labels or aliases, e.g. person bottle mouse",
    )
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.2, help="Minimum detection confidence")
    parser.add_argument("--imgsz", type=int, help="YOLO inference image size, e.g. 960 or 1280")
    parser.add_argument(
        "--diagnostic-confidence",
        type=float,
        help="Lower confidence used only to report filtered-out target matches",
    )
    parser.add_argument(
        "--end2end",
        dest="end2end",
        action="store_true",
        default=None,
        help="Force YOLO end-to-end one-to-one prediction head when supported",
    )
    parser.add_argument(
        "--no-end2end",
        dest="end2end",
        action="store_false",
        help="Use YOLO one-to-many prediction head when supported",
    )
    parser.add_argument("--classes", action="store_true", help="Print model class names and exit")
    parser.add_argument(
        "--output",
        nargs="?",
        const="",
        help="Write an annotated image with detection boxes under images/output",
    )
    parser.add_argument("--tolerance", type=float, default=0.08, help="Centering tolerance as frame ratio")
    args = parser.parse_args()

    try:
        detector = YoloDetector(
            model_path=args.model,
            confidence=args.confidence,
            image_size=args.imgsz,
            end2end=args.end2end,
        )
        if args.classes:
            print(json.dumps(detector.class_names, ensure_ascii=False, indent=2))
            return
        image = load_image(args.image)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    detections_for_output = None

    if args.target:
        aim_results = [
            detector.aim_at(
                image,
                target=target,
                tolerance_ratio=args.tolerance,
                diagnostic_confidence=args.diagnostic_confidence,
            )
            for target in args.target
        ]
        result = [asdict(aim_result) for aim_result in aim_results]
        detections_for_output = [aim_result.detection for aim_result in aim_results if aim_result.detection]
    else:
        detections_for_output = detector.detect(image)
        result = [asdict(detection) for detection in detections_for_output]

    if args.output is not None:
        output_path = _resolve_output_path(args.image, args.output, output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(image, detections_for_output)
        if not cv2.imwrite(str(output_path), annotated):
            print(f"Error: could not write annotated image: {output_path}", file=sys.stderr)
            raise SystemExit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


def _resolve_output_path(image_path: str, output_name: str, output_dir: Path) -> Path:
    if output_name:
        return output_dir / Path(output_name).name

    image = Path(image_path)
    suffix = image.suffix or ".jpg"
    return output_dir / f"{image.stem}_annotated{suffix}"


if __name__ == "__main__":
    main()
