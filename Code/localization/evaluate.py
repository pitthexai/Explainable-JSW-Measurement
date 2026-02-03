from ultralytics import YOLO
from pathlib import Path

WEIGHTS = Path("Models/localization/YOLO11_final.pt")
DATASET_YAML = Path("Dataset/jsw_dataset_final/dataset.yaml")

def main():
    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test"
    )

    for k, v in metrics.results_dict.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()