import torch
from ultralytics import YOLO
import sys

if __name__ == "__main__":
    model_name = sys.argv[1]
    imgsz = int(sys.argv[2])
    
    # Load the model with weights_only=False
    ckpt = torch.load(model_name, map_location="cpu")
    model = YOLO(ckpt['model'])
    
    model.export(format='tflite', imgsz=imgsz)