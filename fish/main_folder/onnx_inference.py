import cv2
import time
import dxcam
import onnxruntime as ort
import numpy as np
from typing import Optional
from fish.main_folder.image_converter import ImageConverter


def onnx_live_detection(session):
    camera = dxcam.create(output_idx=0)
    camera.start(target_fps=120)
    prev_time = 1
    current_time = 2
    input_name = session.get_inputs()[0].name

    while True:
        t0 = time.perf_counter()
        frame = camera.grab()
        # frame = camera.get_latest_frame()

        if frame is None:
            continue

        t1 = time.perf_counter()
        converter = ImageConverter(original_img=frame, converted_shape=(640, 640))
        uint8cv2 = converter.letterbox()
        t2 = time.perf_counter()
        float32 = uint8cv2.astype(np.float32)
        normalization = float32 / 255.0
        hwc_to_chw = np.transpose(normalization, (2, 0, 1))
        batch = np.expand_dims(hwc_to_chw, axis=0)
        t3 = time.perf_counter()

        outputs = session.run(
            None,
            {input_name: batch}
        )

        predictions = outputs[0][0]
        predictions = predictions[predictions[:, 4] > 0.7]

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
            f"letterbox={(t2 - t1) * 1000:.2f} ms, "
            f"preprocess={(t3 - t2) * 1000:.2f} ms, "
            f"inference={(t4 - t3) * 1000:.2f} ms, "
            f"fps_counter={(t5 - t4) * 1000:.2f} ms"
            f"summ={(t5 - t0) * 1000:.2f} ms"
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            camera.stop()
            break

    cv2.destroyAllWindows()

session = ort.InferenceSession(
    path_or_bytes=r"C:\py_projects\albic\fish\main_folder\runs\detect\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best.onnx",
    providers=["CUDAExecutionProvider"],
)

onnx_live_detection(session)
