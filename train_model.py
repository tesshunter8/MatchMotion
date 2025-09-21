from ultralytics import YOLO
import os
import torch
def train():
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    model=YOLO("yolov8s.pt")
    model.train(
        data="C:/Users/lenovo/Desktop/badminton/data.yaml",
        epochs=50, 
        imgsz=640,
        device=0
        )
if __name__=="__main__":
    train()