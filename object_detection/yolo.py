from ultralytics import YOLO


class YoloWrapper:
    def __init__(self, model_path, data_config_path, project):
        """
        Initialize the YOLO API with the model and data configuration.

        :param model_path: Path to the YOLO model.
        :param data_config_path: Path to the dataset YAML configuration file.
        """
        self.model_path = model_path
        self.data_config_path = data_config_path
        self.model = YOLO(model_path)
        self.project = project

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

    def inference(self, image_path):
        """
        Run inference on an image using the YOLO model.

        :param image_path: Path to the input image.
        :return: Inference results.
        """
        results = self.model.predict(image_path)
        return results


if __name__ == "__main__":
    yolo_api = YoloWrapper(
        model_path="data/yolo_weights/yolov10n.pt",
        data_config_path="data/zod/yolo_mini/dataset.yaml",
    )

    # Train the model
    yolo_api.train(epochs=25, batch=32, plots=True)

    # Run inference on an image
    inference_results = yolo_api.inference("path/to/image.jpg")
    print(inference_results)
