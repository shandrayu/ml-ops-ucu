import argparse
import os

from object_detection.yolo import YoloTrainer

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset_file_path", type=str, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    # Instantiate the YoloTrainer with the parameters passed from SageMaker
    yolo = YoloTrainer(
        model_path=args.model,
        data_config_path=args.dataset_file_path,
        project=args.project_name,
    )

    # Run the training
    yolo.train(epochs=args.epochs, batch=args.batch, run_name=args.run_name)

if __name__ == "__main__":
    main()
