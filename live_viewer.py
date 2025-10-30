# live_view.py
# CPU-only YOLO live viewer (GUI required; no headless path)
# Fixed classes: Capacitor, IC, Processor, Tan_Cap

import argparse, sys, time
from pathlib import Path
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ["Capacitor", "IC", "Processor", "Tan_Cap"]
PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

def assert_highgui() -> None:
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
    except cv2.error:
        print("[FATAL] OpenCV was installed without GUI support.\n"
              "Install a GUI build:\n"
              "    pip uninstall -y opencv-python-headless opencv-contrib-python-headless\n"
              "    pip install --no-cache-dir opencv-python\n")
        sys.exit(3)

def recent_best_weights() -> Path | None:
    runs_dir = PROJECT_ROOT / "yolo_data" / "runs"
    if not runs_dir.exists():
        return None
    bests = sorted(runs_dir.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    lasts = sorted(runs_dir.glob("**/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return bests[0] if bests else (lasts[0] if lasts else None)

def is_centered(xyxy, W, H, frac=0.30):
    x1, y1, x2, y2 = xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    fx, fy = 0.5 * W, 0.5 * H
    hw, hh = 0.5 * frac * W, 0.5 * frac * H
    return (fx - hw) <= cx <= (fx + hw) and (fy - hh) <= cy <= (fy + hh)

def parse_args():
    ap = argparse.ArgumentParser(description="YOLO live viewer (CPU, GUI required)")
    ap.add_argument("--model", type=str, default="", help="Weights path; default: latest yolo_data/runs/**/weights/(best|last).pt")
    ap.add_argument("--cam", type=int, default=0, help="Camera index")
    ap.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.800, help="Confidence threshold")
    ap.add_argument("--center-frac", type=float, default=0.30, help="Central box fraction")
    ap.add_argument("--width", type=int, default=0, help="Force capture width (0 keep default)")
    ap.add_argument("--height", type=int, default=0, help="Force capture height (0 keep default)")
    ap.add_argument("--show-fps", action="store_true", help="Overlay FPS")
    ap.add_argument("--save", type=str, default="", help="Optional output mp4 to record annotated feed")
    return ap.parse_args()

def main():
    args = parse_args()
    assert_highgui()

    model_path = Path(args.model) if args.model else (recent_best_weights() or Path("yolov8n.pt"))
    if not model_path.exists():
        print(f"[FATAL] Model not found: {model_path}")
        sys.exit(4)
    print(f"[info] Using model: {model_path}")
    print(f"[info] Classes: {CLASS_NAMES}")

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[FATAL] Could not open webcam index {args.cam}.")
        sys.exit(2)

    if args.width > 0:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(args.save, fourcc, fps_out, (W, H))
        print(f"[info] Recording to: {args.save} @ {fps_out:.1f}fps {W}x{H}")

    history = deque(maxlen=5)
    smoothed_fps = None
    cv2.namedWindow("Live Detection (press q to quit)", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] Frame grab failed; terminating.")
                break

            H, W = frame.shape[:2]
            t0 = time.time()

            res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False, device="cpu")
            preds = []
            for r in res:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(float(b.cls))
                    conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                    x1, y1, x2, y2 = map(int, b.xyxy.squeeze().cpu().numpy().tolist())
                    preds.append((cls_id, conf, (x1, y1, x2, y2)))

            fx, fy = int(0.5 * W), int(0.5 * H)
            hw, hh = int(0.5 * args.center_frac * W), int(0.5 * args.center_frac * H)
            cv2.rectangle(frame, (fx - hw, fy - hh), (fx + hw, fy + hh), (128, 128, 128), 1)

            any_yes = False
            for cls_id, conf, (x1, y1, x2, y2) in preds:
                centered = is_centered((x1, y1, x2, y2), W, H, args.center_frac)
                any_yes |= centered
                color = (0, 255, 0) if centered else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            history.append(1 if any_yes else 0)
            stable_yes = sum(history) >= 3
            status = "YES" if stable_yes else "NO"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if stable_yes else (0, 0, 255), 2)

            if args.show_fps:
                dt = max(1e-6, time.time() - t0)
                fps = 1.0 / dt
                smoothed_fps = fps if smoothed_fps is None else (0.9 * smoothed_fps + 0.1 * fps)
                cv2.putText(frame, f"{smoothed_fps:5.1f} FPS", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Live Detection (press q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == "__main__":
    main()
