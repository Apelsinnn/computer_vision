import cv2
import mss
import time
import random
import torch
import dxcam
import json
import numpy as np
import tensorrt as trt

from typing import Optional
from ultralytics import YOLO

from fish.main_folder.image_converter import ImageConverter

from pathlib import Path


print("="*200)

def onnx_live_detection(engine_path):
    # надо подумать над переписыванием в 2 stream видеокарты, возможно сделать чтение изображения и инференс в разных приложениях
    camera = dxcam.create(output_idx=0)
    camera.start(target_fps=120)
    prev_time = 1
    current_time = 2

    with open(engine_path, "rb") as f:
        meta_len = int.from_bytes(
            f.read(4),
            byteorder="little"
        )

        metadata_bytes = f.read(meta_len) # не читать, а просто подвинуть указатель?
        engine_bytes = f.read()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(
        engine_bytes
    )
    context = engine.create_execution_context()
    input_tensor_name = "images"
    output_tensor_name = "output0"

    for tensor_settings in range(engine.num_io_tensors):
        name = engine.get_tensor_name(tensor_settings)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_tensor_name = name

        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_tensor_name = name

    input_tensor = torch.empty(
        (1, 3, 640, 640),
        dtype=torch.float32,
        device="cuda"
    )

    input_ptr = input_tensor.data_ptr()

    context.set_tensor_address(
        input_tensor_name,
        input_ptr,
    )

    output_tensor = torch.empty(
        (1, 300, 6),
        dtype=torch.float32,
        device="cuda"
    )

    output_ptr = output_tensor.data_ptr()

    context.set_tensor_address(
        output_tensor_name,
        output_ptr,
    )

    stream = torch.cuda.Stream()

    while True:
        t0 = time.perf_counter()
        frame = camera.grab()
        # frame = camera.get_latest_frame()

        if frame is None:
            continue

        t1 = time.perf_counter()
        converter = ImageConverter(original_img=frame, converted_shape=(640, 640))
        uint8cv2 = converter.letterbox()
        float32 = uint8cv2.astype(np.float32)
        normalization = float32 / 255.0
        hwc_to_chw = np.transpose(normalization, (2, 0, 1))
        batch = np.expand_dims(hwc_to_chw, axis=0)
        t2 = time.perf_counter()

        # img_tensor = torch.from_numpy(batch).to("cuda")
        # input_tensor.copy_(img_tensor)

        img_tensor = torch.from_numpy(batch).pin_memory()
        input_tensor.copy_(
            img_tensor,
            non_blocking=True
        )

        with torch.cuda.stream(stream):
            context.execute_async_v3(
                stream.cuda_stream
            )
        stream.synchronize()
        # outputs = output_tensor.cpu().numpy()
        outputs = output_tensor.to("cpu").numpy()

        predictions = outputs[0]
        predictions = predictions[predictions[:, 4] > 0.7]

        t3 = time.perf_counter()
        if len(predictions) > 0:
            for predict in predictions:
                x1, y1, x2, y2 = converter.calculate_reverse_coordinates(predict)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (244,244,10), 2)

        t4 = time.perf_counter()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        current_time = time.time()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = cv2.resize(frame, (1280, 720))
        cv2.imshow("Detection", display)
        t5 = time.perf_counter()

        print(
            f"dxcam={(t1 - t0) * 1000:.2f} ms, "
            f"preprocess={(t2 - t1) * 1000:.2f} ms, "
            f"inference={(t3 - t2) * 1000:.2f} ms, "
            f"boxes={(t4 - t3) * 1000:.2f} ms, "
            f"fps_counter={(t5 - t4) * 1000:.2f} ms, "
            f"all={(t5 - t0) * 1000:.2f} ms"
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            camera.stop()
            break

    cv2.destroyAllWindows()

engine_path = r"C:\py_projects\albic\fish\main_folder\runs\detect\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best_fp32.engine"

onnx_live_detection(engine_path)


# def tensorrt_live_detection(engine_path):
#     camera = dxcam.create(output_idx=0)
#     camera.start(target_fps=120)
#     prev_time = time.time()
#
#     with open(engine_path, "rb") as f:
#         meta_len = int.from_bytes(
#             f.read(4),
#             byteorder="little"
#         )
#
#         metadata_bytes = f.read(meta_len)
#         engine_bytes = f.read()
#
#     logger = trt.Logger(trt.Logger.WARNING)
#     runtime = trt.Runtime(logger)
#     engine = runtime.deserialize_cuda_engine(
#         engine_bytes
#     )
#     context = engine.create_execution_context()
#
#     input_tensor_name = None
#     output_tensor_name = None
#
#     for i in range(engine.num_io_tensors):
#         name = engine.get_tensor_name(i)
#         mode = engine.get_tensor_mode(name)
#
#         if mode == trt.TensorIOMode.INPUT:
#             input_tensor_name = name
#
#         elif mode == trt.TensorIOMode.OUTPUT:
#             output_tensor_name = name
#
#     input_tensor = torch.empty(
#         (1, 3, 640, 640),
#         dtype=torch.float32,
#         device="cuda"
#     )
#
#     output_tensor = torch.empty(
#         (1, 300, 6),
#         dtype=torch.float32,
#         device="cuda"
#     )
#
#     context.set_tensor_address(
#         input_tensor_name,
#         input_tensor.data_ptr()
#     )
#
#     context.set_tensor_address(
#         output_tensor_name,
#         output_tensor.data_ptr()
#     )
#
#     stream = torch.cuda.Stream()
#
#     while True:
#         t0 = time.perf_counter()
#         frame = camera.grab()
#
#         if frame is None:
#             continue
#
#         t1 = time.perf_counter()
#         converter = ImageConverter(
#             original_img=frame,
#             converted_shape=(640, 640)
#         )
#         uint8cv2 = converter.letterbox()
#         t2 = time.perf_counter()
#         float32 = uint8cv2.astype(np.float32)
#         normalization = float32 / 255.0
#         hwc_to_chw = np.transpose(normalization,(2,0,1))
#         batch = np.expand_dims(hwc_to_chw, axis=0)
#
#         img_tensor = torch.from_numpy(batch).to("cuda")
#         input_tensor.copy_(img_tensor)
#
#         t3 = time.perf_counter()
#         with torch.cuda.stream(stream):
#             context.execute_async_v3(
#                 stream.cuda_stream
#             )
#         stream.synchronize()
#         outputs = output_tensor.cpu().numpy()
#         t4 = time.perf_counter()
#         predictions = outputs[0]
#
#         predictions = predictions[predictions[:,4] > 0.7]
#
#         if len(predictions):
#             for pred in predictions:
#                 x1,y1,x2,y2 = converter.calculate_reverse_coordinates(
#                     pred
#                 )
#
#                 cv2.rectangle(
#                     frame,
#                     (x1,y1),
#                     (x2,y2),
#                     (244,244,10),
#                     2
#                 )
#
#         t5 = time.perf_counter()
#         fps = 1 / (time.time() - prev_time)
#         prev_time = time.time()
#         cv2.putText(
#             frame,
#             f"FPS: {fps:.1f}",
#             (10,30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (255,255,0),
#             2
#         )
#         display = cv2.resize(
#             frame,
#             (1280,720)
#         )
#         cv2.imshow(
#             "Detection",
#             display
#         )
#
#         print(
#             f"dxcam={(t1-t0)*1000:.2f} ms | "
#             f"letterbox={(t2-t1)*1000:.2f} ms | "
#             f"preprocess={(t3-t2)*1000:.2f} ms | "
#             f"inference={(t4-t3)*1000:.2f} ms | "
#             f"postprocess={(t5-t4)*1000:.2f} ms | "
#             f"total={(t5-t0)*1000:.2f} ms"
#         )
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             camera.stop()
#             break
#
#     cv2.destroyAllWindows()
#
# # engine_path = r"C:\py_projects\albic\fish\main_folder\runs\detect\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best_fp32.engine"
# #
# # tensorrt_live_detection(engine_path)