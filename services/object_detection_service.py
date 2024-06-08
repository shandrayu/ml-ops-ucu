from object_detection.yolo import ObjectDetectionResult, YoloTrainer, YoloInference
from services import object_detection_service_pb2_grpc


class ObjectDetectionService(
    object_detection_service_pb2_grpc.ObjectDetectionServiceServicer
):
    def __init__(self, model_path):
        self.model_path = model_path
        self.yolo_trainer = YoloTrainer(
            model_path=model_path,
            # TODO: put valid data
            data_config_path="path/to/config",
            project="project_name",
        )
        self.yolo_inference = YoloInference(model_path=model_path)

    def train(self, request, context):
        self.yolo_trainer.train(
            run_name=request.run_name,
            epochs=request.epochs,
            batch=request.batch_size,
            plots=request.plots,
        )
        return object_detection_service_pb2.TrainResponse(message="Training started")

    def inference(self, request, context):
        results = self.yolo_inference.run(image_path=request.image_path)
        response = object_detection_service_pb2.InferenceResponse()
        for result in results:
            detection_result = object_detection_service_pb2.ObjectDetectionResult(
                xyxy=[
                    object_detection_service_pb2.BBox(
                        x1=box[0], y1=box[1], x2=box[2], y2=box[3]
                    )
                    for box in result.xyxy
                ],
                confidence=result.confidence,
                visualization_image=result.visualization_image,
            )
            response.results.append(detection_result)
        return response

    def deploy_best_model(self, request, context):
        # Placeholder for deploying the best model.
        # Now it does not do anything.
        # This command will recreate YoloInference with best model from training.
        return object_detection_service_pb2.DeployBestModelResponse(
            message="Best model deployed"
        )
