import grpc
from concurrent import futures

from services.object_detection_service import ObjectDetectionService
from services.object_detection_service_pb2_grpc import add_ObjectDetectionServiceServicer_to_server

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # TODO: make meaningful model name
    add_ObjectDetectionServiceServicer_to_server(ObjectDetectionService(model_path='YOLOv8/yolov10_best.pt'), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()