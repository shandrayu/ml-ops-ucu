import argparse
import os
import sys
import cv2
import base64
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from object_detection.yolo import YoloInference


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the YOLO model"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )

    args = parser.parse_args()

    yolo_wrapper = YoloInference(model_path=args.model)
    results = yolo_wrapper.run(args.image)
    output_idx = 0
    for result in results:
        # Retrieve image and save it
        decoded_image = base64.b64decode(result.visualization_image)
        nparr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(f"output_{output_idx}.jpg", image)


if __name__ == "__main__":
    main()
