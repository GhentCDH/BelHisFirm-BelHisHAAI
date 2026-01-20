from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

INPUT_DIR = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - JPEG2000")
OUTPUT_DIR = Path("/home/bas/Documents/Visual Code Data/BelHisHAAI/1909 - JPEG2000/processed")
MODEL_PATH = "/home/bas/Documents/Visual Code Data/BelHisHAAI/model/trained/title_detection_model/weights/best.pt"  # or path to your custom model
BATCH_SIZE = 16


def process_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model once
    model = YOLO(MODEL_PATH)

    # Collect all JP2 files
    files = list(INPUT_DIR.rglob("*.jp2"))
    print(f"Found {len(files)} images")

    # Process in batches
    for i in range(0, len(files), BATCH_SIZE):
        batch_paths = files[i:i + BATCH_SIZE]

        # Load images (convert JP2 to numpy arrays)
        images = []
        for p in batch_paths:
            img = np.array(Image.open(p))
            # Convert grayscale to BGR, or RGB to BGR for OpenCV
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)

        # Run inference on batch
        results = model(images, verbose=False)

        # Save images with bounding boxes
        for path, result in zip(batch_paths, results):
            out_path = OUTPUT_DIR / path.with_suffix(".jpg").name
            annotated = result.plot()  # BGR numpy array with boxes drawn
            cv2.imwrite(str(out_path), annotated)
            print(f"OK: {path.name}")

    print("Done")


if __name__ == "__main__":
    process_images()


### 1909 III