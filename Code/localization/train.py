from ultralytics import YOLO
from pathlib import Path

DATASET_YAML = Path("Dataset/dataset.yaml")
MODEL_CFG = "yolo11n.yaml"
OUTPUT_DIR = Path("localization/weights")

def main():
    model = YOLO(MODEL_CFG)

    model.train(
        data=str(DATASET_YAML),
        epochs=150,
        batch=16,
        imgsz=640,
        val=True,
        project=str(OUTPUT_DIR),
        name="localization"
    )

if __name__ == "__main__":
    main()