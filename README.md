
# Shipping Container Detection and Number Recognition

This project uses **YOLOv8** for detecting shipping containers in images and **OCR** (EasyOCR or Tesseract) to extract container numbers printed on them.

---

## Features

- Object detection using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Pre-labeled image support for training
- OCR integration to recognize container IDs (e.g., `XYZ123`)
- Output results saved with bounding boxes and detected numbers
- CSV export of OCR results

---

## Dataset Structure

Your dataset should follow this structure:

```

dataset/
├── images/
│   ├── 000001\_XYZ123.jpg
│   ├── 000002\_XYZ456.jpg
│   └── ...
├── labels/
│   ├── 000001\_XYZ123.txt
│   ├── 000002\_XYZ456.txt
│   └── ...
└── data.yaml

```

Each `.txt` file must contain bounding boxes in YOLO format:
```

\<class\_id> \<x\_center> \<y\_center> <width> <height>

````

> Note: `XYZ123` is the actual container number written on the image and is used only for OCR validation.

---

## 📥 Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-repo/container-detector.git
cd container-detector
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Basic dependencies:

* `ultralytics`
* `opencv-python`
* `easyocr` or `pytesseract`
* `torch`

---

## 🚀 Training the YOLOv8 Model

```bash
python train_yolo.py \
  --data dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device cuda \
  --name container-detector
```

Output will be saved in:

```
runs/detect/container-detector/
```

---

## 🔍 Running Detection + OCR

```bash
python detect_and_ocr_yolov8.py \
  --weights runs/detect/container-detector/weights/best.pt \
  --source test_images/ \
  --ocr_engine easyocr \
  --output output_video.mp4 \
  --csv ocr_results.csv
```

---

## 📄 OCR Output

OCR results are saved in a CSV format:

```
frame_number, container_number
1, XYZ123
2, XYZ456
```
