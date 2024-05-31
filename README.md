# ML Ops UCU course project

## Fine-tune YOLO 10

1. `pip install -q git+https://github.com/THU-MIG/yolov10.git`
2. Get weights

   ```bash
   wget -P data/yolo_weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt &&
   wget -P data/yolo_weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt &&
   wget -P data/yolo_weights -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt
   ```

3. Train

   ```bash
   yolo task=detect mode=train epochs=25 batch=32 plots=True \
   model=data/yolo_weights/yolov10n.pt \
   data=data/zod/yolo_mini/dataset.yaml
   ```
