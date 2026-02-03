from pathlib import Path
import cv2
from ultralytics import YOLO
import argparse

WEIGHTS = Path("Models/localization/YOLO11_final.pt")
OUTDIR = Path("Figures/localization/inference")  # <- folder to save annotated images
OUTDIR.mkdir(parents=True, exist_ok=True)


def main(args):
    # Load model
    model = YOLO(WEIGHTS)

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # Inference
    results = model(image, conf=args.conf)

    # Draw predictions
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Save annotated image
    out_path = OUTDIR / Path(args.image).name
    cv2.imwrite(str(out_path), image)
    print(f"✅ Saved annotated image to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")

    args = parser.parse_args()
    main(args)