import sagemaker
from sagemaker.pytorch import PyTorch
import os
import datetime

# TODO: copy-pasted
def generate_nested_list(base_list):
    nested_list = []
    
    for i in range(len(base_list)):
        nested_list.append(base_list[:i+1])
    
    return nested_list

# TODO: copy-pasted
def get_current_date_time():
    current_time_date = datetime.now()
    formatted_time_date = current_time_date.strftime("%H%M%S%y%m%d")
    return formatted_time_date

def run_yolo_training_on_sagemaker(models, batches, epochs, countries, project_name):
    """
    Start YOLO training for a set of countries on Amazon SageMaker.
    """
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    for country_set in countries:
        dataset_file_path = f"s3://your-bucket-name/path-to-dataset/{country_set}"
        dataset_name = country_set  # Simplify or refine as needed

        for model in models:
            for batch in batches:
                for epoch in epochs:
                    run_name = f"{dataset_name}_{model}_batch{batch}_epochs{epoch}_{get_current_date_time()}"

                    estimator = PyTorch(entry_point='yolo_train.py',  # Your script name
                                        role=role,
                                        framework_version='2.3.1',  # Specify your PyTorch version
                                        py_version='py3',
                                        instance_count=1,
                                        instance_type='ml.p2.xlarge',  # Specify the instance type
                                        hyperparameters={
                                            'model': model,
                                            'batch': batch,
                                            'epochs': epoch,
                                            'data_config_path': dataset_file_path,
                                            'project': project_name
                                        })

                    estimator.fit({'training': dataset_file_path})

if __name__ == "__main__":
    models = ["yolov8n.pt"]
    batches = [32]
    epochs = [1]
    countries = [["IE", "LU"]]

    project_name = "YOLOv8"
    run_yolo_training_on_sagemaker(models, batches, epochs, countries, project_name)
