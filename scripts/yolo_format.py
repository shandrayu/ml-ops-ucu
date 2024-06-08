import os
import json
import yaml
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm.contrib.concurrent import process_map
from functools import partial
from zod import ZodFrames
from zod.anno.object import OBJECT_CLASSES, ObjectAnnotation
from zod.constants import AnnotationProject, Anonymization
from zod.data_classes.frame import ZodFrame
from zod.utils.utils import str_from_datetime

def generate_nested_list(base_list):
    nested_list = []
    
    for i in range(len(base_list)):
        nested_list.append(base_list[:i+1])
    
    return nested_list

# TODO: make as command line arguments
# Parameters
dataset_root = "/home/yshand/repos/ml-ops-ucu/data/zod"
output_dir = dataset_root
classes = ["Vehicle", "Pedestrian", "VulnerableVehicle"]
countries_list = generate_nested_list(["FR", "NO", "GB", "IE", "LU"])

anonymization = Anonymization.BLUR
use_png = False
version = "full"

# Map classes to categories, starting from 1
CATEGORY_NAME_TO_ID = {cls: i + 1 for i, cls in enumerate(OBJECT_CLASSES)}
OPEN_DATASET_URL = "https://www.ai.se/en/data-factory/datasets/data-factory-datasets/zenseact-open-dataset"


def _convert_frame(
    frame: ZodFrame, classes: List[str], anonymization: Anonymization, use_png: bool
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # if allowed_country_codes and frame.metadata.country_code not in allowed_country_codes:
    #     return None, []  # Skip frames not in the allowed countries

    objs: List[ObjectAnnotation] = frame.get_annotation(
        AnnotationProject.OBJECT_DETECTION
    )
    camera_frame = frame.info.get_key_camera_frame(anonymization=anonymization)
    file_name = camera_frame.filepath

    if anonymization == Anonymization.ORIGINAL:
        file_name = file_name.replace(
            Anonymization.BLUR.value, Anonymization.ORIGINAL.value
        )
    if use_png:
        file_name = file_name.replace(".jpg", ".png")

    image_dict = {
        "id": int(frame.info.id),
        "license": 1,
        "file_name": file_name,
        "height": camera_frame.height,
        "width": camera_frame.width,
        "date_captured": str_from_datetime(frame.info.keyframe_time),
        "metadata": {
            "country_code": frame.metadata.country_code,
            "latitude": frame.metadata.latitude,
            "longitude": frame.metadata.longitude,
            "road_condition": frame.metadata.road_condition,
            "road_type": frame.metadata.road_type,
            "scraped_weather": frame.metadata.scraped_weather,
            "solar_angle_elevation": frame.metadata.solar_angle_elevation,
            "time_of_day": frame.metadata.time_of_day,
        },
    }
    anno_dicts = [
        {
            "id": int(frame.info.id) * 1000 + obj_idx,
            "image_id": int(frame.info.id),
            "category_id": CATEGORY_NAME_TO_ID[obj.name],
            # TODO: WARNING: changed to xyxy!!!!
            "bbox": [round(val, 2) for val in obj.box2d.xyxy.tolist()],
            "area": round(obj.box2d.area, 2),
            "iscrowd": obj.subclass == "Unclear",
            "occusion_level": obj.occlusion_level,
        }
        for obj_idx, obj in enumerate(objs)
        if obj.name in classes
    ]
    return image_dict, anno_dicts

# TODO: get rig of coco json generation, it can be generated directly to YOLO
def generate_coco_json(
    dataset: ZodFrames,
    split: str,
    classes: List[str],
    allowed_country_codes: List[str],
    anonymization: Anonymization,
    use_png: bool,
) -> dict:
    frame_infos = [dataset[frame_id] for frame_id in dataset.get_split(split)]

    if not frame_infos:
        return {}

    # Filter frame_infos to include only valid frames
    filtered_frame_infos = [
        frame_info
        for frame_info in frame_infos
        if not allowed_country_codes
        or frame_info.metadata.country_code in allowed_country_codes
    ]

    _convert_frame_w_classes = partial(
        _convert_frame, classes=classes, anonymization=anonymization, use_png=use_png
    )
    results = process_map(
        _convert_frame_w_classes,
        filtered_frame_infos,
        desc=f"Converting {split} frames",
        chunksize=50 if dataset._version == "full" else 1,
    )

    image_dicts, all_annos = zip(*results)
    anno_dicts = [anno for annos in all_annos for anno in annos]  # flatten
    coco_json = {
        "images": image_dicts,
        "annotations": anno_dicts,
        "info": {
            "description": "Zenseact Open Dataset",
            "url": OPEN_DATASET_URL,
            "version": dataset._version,
            "year": 2022,
            "contributor": "ZOD team",
            "date_created": "2022/12/15",
        },
        "licenses": [
            {
                "url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "name": "Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
                "id": 1,
            },
        ],
        "categories": [
            {"supercategory": "object", "id": category_id, "name": category_name}
            for category_name, category_id in CATEGORY_NAME_TO_ID.items()
            if category_name in classes
        ],
    }
    return coco_json

def create_symlink_and_label(args):
    image_info, annotations, images_dir, labels_dir, dataset_root = args
    image_id = image_info["id"]
    image_width = image_info["width"]
    image_height = image_info["height"]
    image_path = Path(image_info["file_name"])
    symlink_path = images_dir / image_path.name
    create_symlink(Path(dataset_root) / image_path, symlink_path)

    label_path = labels_dir / f"{image_path.stem}.txt"
    with open(label_path, "w") as f:
        for anno in annotations:
            category_id = anno["category_id"] - 1  # YOLO category id starts from 0
            bbox = anno["bbox"]
            x_min = max(0, bbox[0])
            y_min = max(0, bbox[1])
            x_max = min(image_width, bbox[2])
            y_max = min(image_height, bbox[3])

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            x_center = (x_min + bbox_width / 2)
            y_center = (y_min + bbox_height / 2)

            bbox_width_normalized = bbox_width / image_width
            bbox_height_normalized = bbox_height / image_height
            x_center_normalized = x_center / image_width
            y_center_normalized = y_center / image_height

            assert x_center_normalized <= 1 and x_center_normalized >= 0, f"image_id={image_id}"
            assert y_center_normalized <= 1 and y_center_normalized >= 0, f"image_id={image_id}"
            assert bbox_height_normalized <= 1 and bbox_height_normalized >= 0, f"image_id={image_id}"
            assert bbox_width_normalized <= 1 and bbox_width_normalized >= 0, f"image_id={image_id}"

            f.write(f"{category_id} {x_center_normalized} {y_center_normalized} {bbox_width_normalized} {bbox_height_normalized}\n")


def create_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src, dst)


def save_yaml_file(yolo_output_dir, classes, train_images_dir, val_images_dir):
    yaml_content = {
        "train": str(train_images_dir),
        "val": str(val_images_dir),
        "nc": len(classes),
        "names": classes,
    }

    yaml_path = yolo_output_dir / "dataset.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)


def convert_to_yolo_format(
    dataset_root: str, version: str, output_dir: str, countries: List[str]
):
    zod_frames = ZodFrames(dataset_root, version)
    yolo_folder_name = (
        f"yolo_{version}" if not countries else f"yolo_{version}_{','.join(countries)}"
    )
    yolo_output_dir = Path(output_dir) / yolo_folder_name

    # Generate COCO JSON
    for split in ["train", "val"]:
        coco_json = generate_coco_json(
            zod_frames,
            split=split,
            classes=classes,
            allowed_country_codes=countries,
            anonymization=anonymization,
            use_png=use_png,
        )

        coco_json_path = yolo_output_dir / f"coco_{split}.json"
        coco_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(coco_json_path, "w") as f:
            json.dump(coco_json, f)

        # Create YOLO formatted dataset
        images_dir = yolo_output_dir / "images" / split
        labels_dir = yolo_output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        image_infos = coco_json["images"]
        annotations_dict = {image_info["id"]: [] for image_info in image_infos}
        for anno in coco_json["annotations"]:
            annotations_dict[anno["image_id"]].append(anno)

        args_list = [
            (
                image_info,
                annotations_dict[image_info["id"]],
                images_dir,
                labels_dir,
                dataset_root,
            )
            for image_info in image_infos
        ]

        process_map(
            create_symlink_and_label,
            args_list,
            desc=f"Creating symlinks and labels for {yolo_folder_name}/{split}",
            chunksize=50 if version == "full" else 1,
        )

    save_yaml_file(
        yolo_output_dir,
        classes,
        yolo_output_dir / "images/train",
        yolo_output_dir / "images/val",
    )


# Run the conversion for each set of countries
for countries in countries_list:
    convert_to_yolo_format(dataset_root, version, output_dir, countries)
