# TODO: move to a separate file
###
import base64
import cv2
import numpy as np


def decode_visualization_image(encoded_image):
    """
    Decodes a base64-encoded image from the gRPC response.

    :param encoded_image: Base64-encoded image bytes from the gRPC response.
    :return: Decoded image as a numpy array.
    """
    decoded_bytes = base64.b64decode(encoded_image)
    nparr = np.frombuffer(decoded_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


###

import grpc
import argparse

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from services import object_detection_service_pb2
from services import object_detection_service_pb2_grpc


def run(train):
    channel = grpc.insecure_channel("localhost:50051")
    stub = object_detection_service_pb2_grpc.ObjectDetectionServiceStub(channel)

    # Training request
    if train:
        train_request = object_detection_service_pb2.TrainRequest(
            run_name="test_run", epochs=1, batch_size=16, plots=True
        )
        train_response = stub.Train(train_request)
        print("Train response:", train_response.message)

    # Inference request
    inference_request = object_detection_service_pb2.InferenceRequest(
        image_path="data/zod/yolo_full_DE,FR,NO,HU,GB/images/val/002707_india_2020-11-20T09:14:16.553489Z.jpg"
        # image_path="data/zod/yolo_full_DE,FR,NO,HU,GB/images/val/004800_golf_2021-02-24T08:58:18.891532Z.jpg"
        # image_path="data/zod/yolo_full_FR/images/train/002699_golf_2021-02-23T12:18:08.105011Z.jpg"
    )
    inference_response = stub.Inference(inference_request)
    print("Inference response results:")
    for result in inference_response.results:
        print("Bounding Boxes:\n", result.xyxy)
        print("Confidence Scores:", result.confidence)
        visualization_image = decode_visualization_image(result.visualization_image)
        scale_percent = 50
        width = int(visualization_image.shape[1] * scale_percent / 100)
        height = int(visualization_image.shape[0] * scale_percent / 100)
        new_dimensions = (width, height)
        downscaled_image = cv2.resize(
            visualization_image, new_dimensions, interpolation=cv2.INTER_AREA
        )
        cv2.imshow("Visualization Image", downscaled_image)
        cv2.waitKey(0)

    # Deploying best model
    deploy_best_model_request = object_detection_service_pb2.DeployBestModelRequest()
    deploy_best_model_response = stub.DeployBestModel(deploy_best_model_request)
    print("Deploy Best Model response:", deploy_best_model_response.message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()
    run(args.train)
