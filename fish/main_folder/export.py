from ultralytics import YOLO

# ONNX
# model = YOLO(r"C:\py_projects\albic\fish\main_folder\runs\detect\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best.pt")
# model.export(format="onnx", nms=False)

# TensorRT
model = YOLO(r"C:\py_projects\albic\fish\main_folder\runs\detect\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best.pt")
model.export(format="engine")