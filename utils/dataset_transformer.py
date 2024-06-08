import pandas as pd
import os
from tqdm import tqdm


def convert_zod_to_pandas(
    ids,
    zod_dataset,
    name,
    load_buffer_if_available=True,
    classes=["Vehicle", "Pedestrian", "VulnerableVehicle"],
):
    frames_buffer_name = f"frames_{name}.csv"
    object_buffer_name = f"objects_{name}.csv"
    if (
        load_buffer_if_available
        and os.path.exists(frames_buffer_name)
        and os.path.exists(object_buffer_name)
    ):
        frames_df = pd.read_csv(frames_buffer_name)
        objects_df = pd.read_csv(object_buffer_name)

    else:
        data_list = []
        object_list = []
        for frame_id in tqdm(ids, desc=f"Processing frames for {name}"):
            frame = zod_dataset[frame_id]
            data_list.append(
                {
                    "frame_id": frame.metadata.frame_id,
                    "country_code": frame.metadata.country_code,
                    "road_condition": frame.metadata.road_condition,
                    "road_type": frame.metadata.road_type,
                    "scraped_weather": frame.metadata.scraped_weather,
                    "time_of_day": frame.metadata.time_of_day,
                }
            )

            object_detections = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
            for detection in object_detections:
                if detection.name in classes:
                    object_list.append(
                        {
                            "uuid": detection.uuid,
                            "frame_id": frame_id,
                            "name": detection.name,
                            "object_type": detection.object_type,
                            "occlusion_level": detection.occlusion_level,
                            "unclear": detection.unclear,
                            "traffic_content_visible": detection.traffic_content_visible,
                            "with_rider": detection.with_rider,
                        }
                    )
        frames_df = pd.DataFrame(data_list)
        frames_df.to_csv(frames_buffer_name, index=False)
        objects_df = pd.DataFrame(object_list)
        objects_df.to_csv(object_buffer_name, index=False)
    return frames_df, objects_df
