import cv2
import time
import torch
import dxcam
import numpy as np
import tensorrt as trt

from dataclasses import dataclass
from queue import Queue, Empty, Full
from threading import Thread

from fish.main_folder.image_converter import ImageConverter

print("="*200)

@dataclass(slots=True)
class FrameData:
    frame: np.ndarray
    converter: ImageConverter | None = None
    batch: np.ndarray | None = None

class TensorrtDetector:
    """Пайплайн детекции изображений в реальном времени."""
    def __init__(self, engine_path: str):
        self._engine_path = engine_path
        self._running = False
        self._camera = None
        self._capture_queue = None
        self._preprocess_queue = None
        self._inference_queue = None
        self._context = None
        self._input_tensor = None
        self._output_tensor = None
        self._stream = None

    def _capture_loop(self):
        """Запускает бесконечный цикл захвата изображений."""
        while self._running:
            frame = self._camera.grab()
            # frame = self.camera.get_latest_frame()

            if frame is None:
                continue

            frame_data = FrameData(frame)

            try:
                self._capture_queue.put_nowait(frame_data)
            except Full:
                self._capture_queue.get_nowait()
                self._capture_queue.put_nowait(frame_data)

    def _preprocess_loop(self):
        """Запускает бесконечный цикл преобразования изображений."""
        while self._running:
            frame_data = self._capture_queue.get()
            converter = ImageConverter(original_img=frame_data.frame, converted_shape=(640, 640))
            img = converter.letterbox()
            img = img.astype(np.float32)
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            frame_data.converter = converter
            frame_data.batch = img

            try:
                self._preprocess_queue.put_nowait(frame_data)
            except Full:
                self._preprocess_queue.get_nowait()
                self._preprocess_queue.put_nowait(frame_data)

    def _inference_loop(self):
        """Запускает бесконечный цикл инференса изображений."""
        current_time = 2

        while self._running:
            frame_data = self._preprocess_queue.get()
            img_tensor = torch.from_numpy(frame_data.batch).pin_memory()
            self._input_tensor.copy_(img_tensor, non_blocking=True)

            with torch.cuda.stream(self._stream):
                self._context.execute_async_v3(
                    self._stream.cuda_stream
                )

            self._stream.synchronize()
            outputs = self._output_tensor.cpu().numpy()
            predictions = outputs[0]
            predictions = predictions[
                predictions[:, 4] > 0.7
                ]

            prev_time = time.time()
            fps = 1 / (prev_time - current_time)

            try:
                self._inference_queue.put_nowait((predictions, frame_data, fps))
            except Full:
                self._inference_queue.get_nowait()
                self._inference_queue.put_nowait((predictions, frame_data, fps))

            current_time = time.time()

    def _display_loop(self):
        """Запускает бесконечный цикл отображения найденных объектов."""

        while self._running:
            predictions, frame_data, fps = self._inference_queue.get()

            # t0 = time.perf_counter()
            frame = frame_data.frame

            for predict in predictions:
                x1, y1, x2, y2 = (
                    frame_data.converter.calculate_reverse_coordinates(predict)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2
                )

            frame = cv2.resize(frame, (1280, 720))
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.imshow("Detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                self._running = False
                self._camera.stop()
                break

            # t1 = time.perf_counter()
            # print(f"display: {(t1 - t0) * 1000:.2f}ms")

    def _prepare_settings(self):
        """Задаёт начальные настройки."""
        self._running = True
        self._camera = dxcam.create(output_idx=0)
        self._camera.start(target_fps=120)
        self._capture_queue = Queue(maxsize=1)
        self._preprocess_queue = Queue(maxsize=1)
        self._inference_queue = Queue(maxsize=1)

        with open(self._engine_path, "rb") as f:
            meta_len = int.from_bytes(
                f.read(4),
                byteorder="little"
            )

            metadata_bytes = f.read(meta_len)
            engine_bytes = f.read()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(
            engine_bytes
        )
        self._context = engine.create_execution_context()
        input_tensor_name = "images"
        output_tensor_name = "output0"

        for tensor_settings in range(engine.num_io_tensors):
            name = engine.get_tensor_name(tensor_settings)

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_tensor_name = name

            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_tensor_name = name

        self._input_tensor = torch.empty(
            (1, 3, 640, 640),
            dtype=torch.float32,
            device="cuda"
        )

        input_ptr = self._input_tensor.data_ptr()

        self._context.set_tensor_address(
            input_tensor_name,
            input_ptr,
        )

        self._output_tensor = torch.empty(
            (1, 300, 6),
            dtype=torch.float32,
            device="cuda"
        )

        output_ptr = self._output_tensor.data_ptr()

        self._context.set_tensor_address(
            output_tensor_name,
            output_ptr,
        )

        self._stream = torch.cuda.Stream()

    def _enable_threads(self):
        """Запускает потоки."""
        capture_thread = Thread(
            target=self._capture_loop,
            daemon=True
        )

        preprocess_thread = Thread(
            target=self._preprocess_loop,
            daemon=True
        )

        inference_thread = Thread(
            target=self._inference_loop,
            daemon=True
        )

        display_thread = Thread(
            target=self._display_loop,
            daemon=True
        )

        capture_thread.start()
        preprocess_thread.start()
        inference_thread.start()
        display_thread.start()

        capture_thread.join()
        preprocess_thread.join()
        inference_thread.join()
        display_thread.join()

    def live_detection(self):
        """Запускает детекцию на экране."""
        self._prepare_settings()
        self._enable_threads()

model_engine_path = (r"C:\py_projects\albic\fish\main_folder\runs\detect"
               r"\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best_fp32.engine")

detector = TensorrtDetector(engine_path=model_engine_path)
detector.live_detection()

