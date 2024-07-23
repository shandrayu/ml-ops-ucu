import os
import sys
# from ultralytics import settings
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from object_detection.dataset_formation import yolo_dataset_formation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from object_detection.yolo import YoloTrainer


def generate_nested_list(base_list):
    nested_list = []
    
    for i in range(len(base_list)):
        nested_list.append(base_list[:i+1])
    
    return nested_list

def get_current_date_time():
    current_time_date = datetime.now()
    formatted_time_date = current_time_date.strftime("%H%M%S%y%m%d")
    return formatted_time_date


def run_yolo_training(models, batches, epochs, countries, project_name):
    """
    Start YOLO training for a set of countries.
    """

    for country_set in countries:
        # Prepare dataset here
        dataset_file_path = yolo_dataset_formation(country_list=country_set,
                                                   classes=CLASSES,
                                                   version=VERSION, 
                                                   data_path=DATASET_ROOT, 
                                                   copy_files=COPY_FILES)
        dataset_name = dataset_file_path.parent.name
        for model in models:
            for batch in batches:
                for epoch in epochs:
                    run_name = (
                        f"{dataset_name}_{model.split('/')[-1]}_batch{batch}_epochs{epoch}_{get_current_date_time()}"
                    )
                    # TODO: Add data augmentation here
                    data_augmentation_config = {
                        ""
                    }
                    # Start SageMaker Estimator job here
                    yolo = YoloTrainer(
                        model_path=model,
                        data_config_path=dataset_file_path,
                        project=project_name,
                    )
                    yolo.train(epochs=epoch, batch=batch, run_name=run_name)


if __name__ == "__main__":    
    # Training parameters
    models = ["yolov8n.pt"]
    batches = [32]
    epochs = [1]

    # Data parameters
    # TODO: create dataset config file
    # dataset_config = {
    #     "dataset_path": "s3://some-path", 
    #     "dataset_version": "1.2",
    #     "countries": ["IE", "LU"], 
    #     "classes": ["Vehicle", "Pedestrian", "VulnerableVehicle"],
    #     }
    # TODO: log dataset_config to mlflow
    VERSION = "full"
    DATASET_ROOT = "/Users/yshand/repos/ml-ops-ucu/data/zod"
    CLASSES = ["Vehicle", "Pedestrian", "VulnerableVehicle"]
    COUNTRIES = ["IE", "LU"]

    # Technical parameters
    COPY_FILES = True
    country_list = generate_nested_list(COUNTRIES)
    
    # Tracking parameters
    project_name = "YOLOv8"

    run_yolo_training(models, batches, epochs, country_list, project_name)
