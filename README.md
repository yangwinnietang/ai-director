# Director Vision Tool

Small YOLO-based vision service for robotic camera movement. It detects objects in an image or camera frame and returns a simple aiming command for the robot controller.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Aim at a target in an uploaded image:

```powershell
curl -X POST http://127.0.0.1:8000/aim/image -F "target=person" -F "file=@sample.jpg"
```

Aim from camera index 0:

```powershell
curl -X POST http://127.0.0.1:8000/aim/camera -H "Content-Type: application/json" -d '{"target":"person","camera_index":0}'
```

## CLI

```powershell
python vision_cli.py sample.jpg --target person
```

To save an annotated image with detection boxes:

```powershell
python vision_cli.py sample.jpg --output
```

Annotated images are written to `images\output`. You can also choose the output filename:

```powershell
python vision_cli.py sample.jpg --output annotated.jpg
```

You can combine this with target aiming too:

```powershell
python vision_cli.py sample.jpg --target person --output annotated.jpg
```

To see why a target was not found, lower the diagnostic threshold or increase the inference image size:

```powershell
python vision_cli.py sample.jpg --target person --diagnostic-confidence 0.05 --imgsz 1280
```

Useful target-debugging options:

- `--classes`: print the class names supported by the loaded model.
- `--imgsz 1280`: run inference at a larger image size, which can help with small or distant people.
- `--diagnostic-confidence 0.05`: report target detections that were filtered out by the main confidence threshold.
- `--no-end2end`: ask supported YOLO models to use the one-to-many head.

Multiple targets can be checked in one call:

```powershell
python vision_cli.py sample.jpg --target person bottle mouse
```

If model download is interrupted, delete the partial weight file and run again. When using the bundled local model, this should not be needed:

```powershell
Remove-Item .\yolo26s.pt
```

## Output Contract

The robot controller should mainly use these fields:

- `found`: whether the target is visible.
- `accepted_labels`: model class names accepted for the requested target.
- `confidence_threshold`: minimum confidence used for this result.
- `image_size`: inference image size passed to YOLO, or `null` for the model default.
- `detection_count`: number of detections returned before target filtering.
- `match_count`: number of detections matching the accepted target labels.
- `labels_seen`: object labels YOLO actually returned at the main confidence threshold.
- `miss_reason`: why `found` is false, such as `no_detections`, `no_matching_labels`, or `only_low_confidence_matches`.
- `low_confidence_matches`: target matches found only below `confidence_threshold` when `--diagnostic-confidence` is used.
- `command`: one of `search`, `hold`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`.
- `centered`: whether the target is close enough to the frame center.
- `offset_ratio`: target center offset from image center. Positive x means target is on the right, positive y means target is lower.
- `detection.box`: target bounding box as `[x1, y1, x2, y2]`.

For production scenes with named actors, train a custom YOLO model or add a face/person re-identification module. Generic YOLO can detect `person`, but it cannot reliably know which person is the male lead or female lead without extra identity logic.

## Ubuntu Deployment

For a robot-side Ubuntu machine, copy this project directory to the machine and run:

```bash
bash deploy/install_ubuntu.sh
```

The script installs the app into `/opt/director-vision`, creates a `director` service user, installs Python dependencies, and registers a systemd service named `director-vision`.

Useful commands:

```bash
sudo systemctl status director-vision
sudo journalctl -u director-vision -f
curl http://127.0.0.1:8000/health
```

Configuration lives in `/opt/director-vision/.env`:

```bash
YOLO_MODEL=yolo26s.pt
YOLO_CONFIDENCE=0.2
```

If you train a custom model, put the `.pt` file on the Ubuntu machine and set `YOLO_MODEL=/path/to/best.pt`.
