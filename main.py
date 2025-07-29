# ---------------------------------- Importing Libraries ---------------------------------- #
import csv
import argparse
import cv2
import easyocr
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------- Configuration Setup ---------------------------------- #
CONFIG = {
    "model_path": "/home/ayaan/code/NumberPlateDetection-using-YOLO/Model/yolov8n.pt",
    "output_video": "output_video.mp4",
    "csv_path": "ocr_results.csv",
    "ocr_languages": ["en"],
    "confidence_threshold": 0.5,
    "frame_skip": 0,
    "verbose": True
}

# ---------------------------------- Argument Parsing ---------------------------------- #
def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLOv8 + EasyOCR License Plate Recognition')
    parser.add_argument('--input', default=CONFIG["input_video"], help='Path to input video')
    parser.add_argument('--output', default=CONFIG["output_video"], help='Path to save annotated video')
    parser.add_argument('--csv', default=CONFIG["csv_path"], help='Path to save OCR results CSV')
    parser.add_argument('--model', default=CONFIG["model_path"], help='Path to YOLOv8 model')
    parser.add_argument('--skip', type=int, default=CONFIG["frame_skip"], help='Frame skip interval')
    parser.add_argument('--threshold', type=float, default=CONFIG["confidence_threshold"], help='Confidence threshold')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

# ---------------------------------- Main Function ---------------------------------- #
def process_video():
    args = parse_arguments()
    CONFIG.update(vars(args))

    # Load YOLOv8 Model
    try:
        model = YOLO(CONFIG["model"])
        if CONFIG["verbose"]:
            print(f"[INFO] YOLOv8 model loaded: {CONFIG['model']}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Initialize EasyOCR
    try:
        reader = easyocr.Reader(CONFIG["ocr_languages"])
        if CONFIG["verbose"]:
            print("[INFO] EasyOCR initialized.")
    except Exception as e:
        print(f"[ERROR] OCR init failed: {e}")
        return

    # Setup video I/O
    cap = cv2.VideoCapture(CONFIG["input_video"])
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {CONFIG['input_video']}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(CONFIG["output_video"],
                          cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (frame_width, frame_height))

    # CSV setup
    csv_file = open(CONFIG["csv_path"], mode='w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=[
        "Timestamp", "Frame", "X1", "Y1", "X2", "Y2", "Confidence", "Detected_Text", "Processing_Time"
    ])
    writer.writeheader()

    frame_count = 0
    processed_frames = 0
    start_time = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if CONFIG["frame_skip"] > 0 and frame_count % (CONFIG["frame_skip"] + 1) != 0:
            continue

        processed_frames += 1
        try:
            results = model(frame)[0]  # single image inference
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < CONFIG["confidence_threshold"]:
                    continue

                cls = int(box.cls[0])
                if cls != 0:  # assuming class 0 is license plate
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.shape[0] < 10 or plate_crop.shape[1] < 10:
                    continue

                # OCR
                ocr_start = datetime.now()
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                text = reader.readtext(thresh, detail=0)
                ocr_time = (datetime.now() - ocr_start).total_seconds()
                text_str = " ".join(text).strip()

                if CONFIG["verbose"]:
                    print(f"[INFO] Frame {frame_count} | Text: {text_str} | Confidence: {conf:.2f} | OCR Time: {ocr_time:.2f}s")

                writer.writerow({
                    "Timestamp": datetime.now().isoformat(),
                    "Frame": frame_count,
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                    "Confidence": f"{conf:.2f}",
                    "Detected_Text": text_str,
                    "Processing_Time": ocr_time
                })

                color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
                label = f"{text_str} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)

        except Exception as e:
            print(f"[ERROR] Error at frame {frame_count}: {e}")

        if CONFIG["verbose"] and processed_frames % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            fps_calc = processed_frames / elapsed if elapsed > 0 else 0
            print(f"[PROGRESS] Processed {processed_frames}/{total_frames} frames ({fps_calc:.1f} FPS)")

    cap.release()
    out.release()
    csv_file.close()
    print(f"[INFO] Done. Output saved to {CONFIG['output_video']} and {CONFIG['csv_path']}")

if __name__ == "__main__":
    process_video()
