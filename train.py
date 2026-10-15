import argparse
from ultralytics import YOLO

def train():
    parser = argparse.ArgumentParser(description="YOLO Object Detection Training Pipeline")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="yolov8l.pt", help="Pretrained model backbone")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=256, help="Image resolution for training")
    args = parser.parse_args()

    # Load YOLO model backbone
    model = YOLO(args.model)

    # Start model training workflow
    print(f"[INFO] Starting training with {args.model} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="runs/detect",
        name="custom_yolo_experiment"
    )
    print("[INFO] Training completed successfully.")

if __name__ == "__main__":
    train()
