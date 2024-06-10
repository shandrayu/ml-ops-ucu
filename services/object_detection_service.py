from object_detection.yolo import ObjectDetectionResult, YoloTrainer, YoloInference
from services import object_detection_service_pb2_grpc
from services import object_detection_service_pb2


class ObjectDetectionService(
    object_detection_service_pb2_grpc.ObjectDetectionServiceServicer
):
    def __init__(self, model_path):
        self.model_path = model_path
        self.yolo_trainer = YoloTrainer(
            model_path=model_path,
            # TODO: pass dataset file as parameter
            # TODO: is there a better way to handle dataset other that mounting volume?
            data_config_path="/app/data/zod/yolo_mini/dataset.yaml",
            project="YOLOv10",
        )
        self.yolo_inference = YoloInference(model_path=model_path)

    def Train(self, request, context):
        self.yolo_trainer.train(
            run_name=request.run_name,
            epochs=request.epochs,
            batch=request.batch_size,
            plots=request.plots,
        )
        return object_detection_service_pb2.TrainResponse(message="Training started")

    def Inference(self, request, context):
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

    def DeployBestModel(self, request, context):
        # Placeholder for deploying the best model.
        # Now it does not do anything.
        # This command will recreate YoloInference with best model from training.
        print("Deploy best model called. No implementation yet")
        return object_detection_service_pb2.DeployBestModelResponse(
            message="Best model deployed"
        )
