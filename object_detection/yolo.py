from ultralytics import YOLOv10
from ultralytics import settings
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")

settings.update({"mlflow": True, 'wandb': False})
print(settings)

from typing import List, Tuple
from dataclasses import dataclass
import cv2
import base64


@dataclass
class ObjectDetectionResult:
    xyxy: List[Tuple[int, int, int, int]]  # Coordinates of detected objects
    confidence: List[float]  # Probabilities of detected objects
    # TODO: only for debug, in real system it will not be needed
    visualization_image: bytes  # Image in binary form


class YoloTrainer:
    def __init__(self, model_path, data_config_path, project):
        """
        Initialize the YOLO API with the model and data configuration.

        :param model_path: Path to the YOLO model.
        :param data_config_path: Path to the dataset YAML configuration file.
        """
        # TODO: model name is not used now. Only the smallest (nano) used for now
        # self.model_path = model_path
        self.data_config_path = data_config_path
        self.model = YOLOv10.from_pretrained("jameslahm/yolov10n")  # YOLO(model_path)
        self.project = project

    # TODO: read training parameters from the file. If we will add augmentation, there will be too much parameters
    def train(self, run_name, epochs=25, batch=32, plots=True):
        """
        Train the YOLO model.

        :run_name: custom name for a run.
        :param epochs: Number of epochs to train for.
        :param batch_size: Batch size for training.
        :param plots: Whether to plot training results.
        """
        self.model.train(
            data=self.data_config_path,
            epochs=epochs,
            batch=batch,
            plots=plots,
            project=self.project,
            name=run_name,
        )


class YoloInference:
    def __init__(self, model_path) -> List[ObjectDetectionResult]:
        """
        Initialize the YOLO model.

        :param model_path: Path to the YOLO model.
        """
        # https://github.com/THU-MIG/yolov10/issues/46
        model_name = model_path.split("/")[-1]
        assert (
            "yolov10" in model_name
        ), f"Rename model name '{model_name}' to contain 'yolov10'. Otherwise, it will fail...."
        self.model = YOLOv10(model_path)

    def run(self, image_path):
        """
        Run inference on an image using the YOLO model.

        :param image_path: Path to the input image.
        :return: Inference results.
        """
        results = self.model.predict(image_path)
        result_objects = []
        for result in results:
            boxes = result.boxes
            visualization_img = result.plot()
            _, buffer = cv2.imencode(".jpg", visualization_img)
            encoded_image = base64.b64encode(buffer)
            r = ObjectDetectionResult(
                xyxy=boxes.xyxy.cpu().numpy().astype(int),
                confidence=boxes.conf.cpu().numpy(),
                visualization_image=encoded_image,
            )
            result_objects.append(r)

        return result_objects
