from ultralytics import YOLO
import argparse

# ---------------------------------- Argument Parsing ---------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for shipping container detection")
    parser.add_argument('--data', type=str, required=True, help='/home/ayaan/code/NumberPlateDetection/conf.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='/home/ayaan/code/NumberPlateDetection-using-YOLO/Models/yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--name', type=str, default='container-detector', help='Experiment name')
    return parser.parse_args()
# ---------------------------------- Model Training ---------------------------------- #
def main():
    args = parse_args()
    print(f"[INFO] Starting training for model {args.model}")

    model = YOLO(args.model)  # Load a YOLOv8 base model (e.g., yolov8n.pt)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name
    )

    print(f"[INFO] Training complete. Model saved in 'runs/detect/{args.name}'")

if __name__ == "__main__":
    main()
