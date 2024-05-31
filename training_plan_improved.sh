#!/bin/bash

# TODO: create local vitrual environment
source ../cont-learning/.cont-lear-venv/bin/activate

cleanup() {
  echo "An error occurred. Cleaning up..."
  # TODO: deactivate only if all processes have finished
  # deactivate
  # Terminate any background processes
  kill 0
  if command -v nvidia-smi &> /dev/null; then
    echo "Releasing GPU memory..."
    nvidia-smi --gpu-reset
  fi
  exit 1
}

trap cleanup ERR

run_yolo_training() {
  local model=$1
  local batch=$2
  local epochs=$3
  local dataset=$4
  local run_name=$5

  yolo task=detect mode=train epochs=$epochs batch=$batch plots=True model=$model data=data/zod/$dataset/dataset.yaml project=YOLOv8 name=$run_name
}

declare -a models=("data/yolo_weights/yolov10n.pt" "data/yolo_weights/yolov10s.pt" "data/yolo_weights/yolov10m.pt")
declare -a batches=(16)
declare -a epochs=(25)
declare -a datasets=("yolo_full_DE" "yolo_full_DE,FR,NO")

for model in "${models[@]}"; do
  for batch in "${batches[@]}"; do
    for epoch in "${epochs[@]}"; do
      for dataset in "${datasets[@]}"; do
        run_name="${dataset}_${model##*/}_batch${batch}_epochs${epoch}"
        if ! run_yolo_training $model $batch $epoch $dataset $run_name; then
          cleanup
        fi
      done
    done
  done
done

deactivate

echo "All YOLO training commands executed successfully."