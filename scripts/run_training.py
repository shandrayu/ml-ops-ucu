import os
import sys
from ultralytics import settings

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from object_detection.yolo import YoloTrainer


def run_yolo_training(models, batches, epochs, datasets, project_name):
    """
    Start YOLO training for a set of datasets.

    :param models: List of model paths.
    :param batches: List of batch sizes.
    :param epochs: List of epoch counts.
    :param datasets: List of dataset names.
    """

    for model in models:
        for batch in batches:
            for epoch in epochs:
                for dataset in datasets:
                    run_name = (
                        f"{dataset}_{model.split('/')[-1]}_batch{batch}_epochs{epoch}"
                    )
                    data_config_path = f"data/zod/{dataset}/dataset.yaml"
                    yolo = YoloTrainer(
                        model_path=model,
                        data_config_path=data_config_path,
                        project=project_name,
                    )
                    yolo.run_training(epochs=epoch, batch=batch, run_name=run_name)


if __name__ == "__main__":
    # TODO: can 
    # Update a setting
    # settings.update({"runs_dir": "/path/to/runs"})

    # # Update multiple settings
    # settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

    models = ["data/yolo_weights/yolov10n.pt"]
    batches = [16]
    epochs = [2]
    datasets = [
        "yolo_full_FR",
        "yolo_full_FR,NO",
        "yolo_full_FR,NO,GB",
        "yolo_full_FR,NO,GB,IE",
        "yolo_full_FR,NO,GB,IE,LU",
    ]

    project_name = "YOLOv8"

    run_yolo_training(models, batches, epochs, datasets, project_name)
