from ultralytics import YOLO
import torch
print(torch.cuda.is_available())


if __name__ == "__main__":
    # model = YOLO("best.pt")
    # model = YOLO("yolo11m.pt")
    # model = YOLO("yolo26s.pt")
    model = YOLO(r"C:\py_projects\albic\fish\main_folder\runs\detect\train_float_with_false_positive_e1000_b16\weights\best.pt")

    results = model.train(data="data.yaml", epochs=50, imgsz=640, name="train_float_with_false_positive_e1000_b16_fine_tuning", batch=16, mosaic=0.0)
    # results = model("for_test.png", save=True, show=True, project="runs/detect/predict")
