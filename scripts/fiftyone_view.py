import os

import fiftyone as fo

"""
Copied from cv homework

Notes: if closed incorrectly, there may arise issue with fiftyone mondodb.
Only deleting entirely folder `.fiftyone` helped so far.
"""

def load_51_dataset(dataset_name, labels_path, data_path=None):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path
    )
    dataset.name = dataset_name
    return dataset


current_file_path = os.path.dirname(os.path.abspath(__file__))

# labels_path = os.path.abspath(os.path.join(current_file_path, "..", "data/coco/zod_full_Anonymization.BLUR_train.json"))
# dataset_name = "zod-dataset-train"
data_path = current_file_path

labels_path = os.path.abspath(os.path.join(current_file_path, "..", "data/coco/zod_full_Anonymization.BLUR_val.json"))
dataset_name = "zod-dataset-val"


reload_even_if_exists = True 


if fo.dataset_exists(dataset_name):
    if reload_even_if_exists:
        fo.delete_dataset(dataset_name)
        dataset = load_51_dataset(data_path=data_path, labels_path=labels_path, dataset_name=dataset_name)
        print(f"Dataset {dataset_name} re-loaded successfully.")
    else:
        dataset = fo.load_dataset(dataset_name)
        print(f"Dataset {dataset_name} loaded successfully.")
else:
    print(f"Dataset {dataset_name} does not exist. Loading...")
    dataset = load_51_dataset(data_path=data_path, labels_path=labels_path, dataset_name=dataset_name)

print(dataset.get_field_schema())
dataset = fo.load_dataset(dataset_name)
session = fo.launch_app(dataset)
session.wait()