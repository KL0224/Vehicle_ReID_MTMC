from ultralytics import YOLO
import os
import torch

def load_model_detector(model_path):
    if not os.path.exists(model_path):
        print(f"Không tồn tại model tại đường dẫn {model_path}")
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Đã load model thành công lên {device}")
    return model