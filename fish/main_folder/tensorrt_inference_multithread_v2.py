import cv2
import time
import torch
import dxcam
import numpy as np
import tensorrt as trt

from dataclasses import dataclass
from threading import Thread

from fish.main_folder.image_converter_v2 import ImageConverter

print("=" * 200)


@dataclass(slots=True)
class FrameData:
    frame: np.ndarray = None
    batch: np.ndarray = None

@dataclass(slots=True)
class InferenceData:
    predictions: list = None
    frame: np.ndarray = None


class TensorrtDetector:
    """Пайплайн детекции изображений для модели TensorRT в реальном времени."""

    def __init__(self, engine_path: str):
        self._engine_path = engine_path
        self._running = False
        self._camera = None
        self._context = None
        self._input_tensor = None
        self._output_tensor = None
        self._stream = None
        self._capture_frame = None
        self._preprocess_frame = None
        self._inference_data = InferenceData()
        self._fps = None
        self._converter = None
        self._host_input = None

    def _capture_loop(self):
        """Запускает бесконечный цикл захвата изображений."""
        while self._running:
            frame = self._camera.grab()
            # frame = self.camera.get_latest_frame()

            if frame is None:
                continue

            self._capture_frame = frame

    def _preprocess_loop(self):
        """Запускает бесконечный цикл преобразования изображений."""
        while self._capture_frame is None:
            print("Ожидание первого кадра!")
            time.sleep(0.2)

        while self._running:
            # frame = self._capture_frame.copy() # type: ignore
            frame = self._capture_frame # type: ignore
            img = self._converter.letterbox(frame)
            img = img.astype(np.float32)
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            self._preprocess_frame = FrameData(frame=frame, batch=img)

    def _inference_loop(self):
        """Запускает бесконечный цикл инференса изображений."""
        prev_time = 2

        while self._preprocess_frame is None:
            print("Ожидание обработанного кадра!")
            time.sleep(0.4)

        while self._running:
            t0 = time.perf_counter()

            frame_data = self._preprocess_frame
            frame, batch = frame_data.frame, frame_data.batch # type: ignore

            np.copyto(self._host_input.numpy(), batch)
            self._input_tensor.copy_(self._host_input, non_blocking=True)

            with torch.cuda.stream(self._stream):
                self._context.execute_async_v3(
                    self._stream.cuda_stream
                )

            self._stream.synchronize()
            outputs = self._output_tensor.cpu().numpy()
            predictions = outputs[0]
            predictions = predictions[predictions[:, 4] > 0.7]

            current_time = time.perf_counter()
            self._fps = 1 / (current_time - prev_time)
            self._inference_data = InferenceData(predictions=predictions, frame=frame)
            prev_time = current_time
            t1 = time.perf_counter()
            print(f"inference: {(t1 - t0) * 1000:.2f}ms")

    def _display_loop(self):
        """Запускает бесконечный цикл отображения найденных объектов."""
        while self._inference_data.predictions is None:
            print("Ожидание предсказания модели!")
            time.sleep(0.6)

        while self._running:
            # print(id(self._inference_data))
            inference_data = self._inference_data
            predictions, frame = inference_data.predictions, inference_data.frame
            # t0 = time.perf_counter()

            for predict in predictions:
                x1, y1, x2, y2 = (
                    self._converter.calculate_reverse_coordinates(predict)
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
            cv2.putText(
                img=display,
                text=f"FPS: {self._fps:.1f}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(0, 255, 255),
                thickness=2,
            )
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
        self._camera.start(target_fps=240)

        full_hd_monitor = (1920, 1080)
        model_training_size = (640, 640)
        gray = (114, 114, 114)
        self._converter = ImageConverter(
            original_shape=full_hd_monitor, converted_shape=model_training_size, padding_color=gray
        )

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

        model_training_batch = (1, 3, 640, 640)
        self._input_tensor = torch.empty(
            size=model_training_batch,
            dtype=torch.float32,
            device="cuda"
        )

        self._host_input = torch.empty(
            size=model_training_batch,
            dtype=torch.float32,
            pin_memory=True,
        )

        input_ptr = self._input_tensor.data_ptr()

        self._context.set_tensor_address(
            input_tensor_name,
            input_ptr,
        )

        model_output_tensor_shape = (1, 300, 6)
        self._output_tensor = torch.empty(
            size=model_output_tensor_shape,
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
        capture_thread = Thread(target=self._capture_loop)
        preprocess_thread = Thread(target=self._preprocess_loop)
        inference_thread = Thread(target=self._inference_loop)
        display_thread = Thread(target=self._display_loop)

        capture_thread.start()
        preprocess_thread.start()
        inference_thread.start()
        display_thread.start()

        capture_thread.join()
        preprocess_thread.join()
        inference_thread.join()
        display_thread.join()

    def live_detection(self):
        """Запускает детекцию объектов на экране."""
        self._prepare_settings()
        self._enable_threads()


model_engine_path = (r"C:\py_projects\albic\fish\main_folder\runs\detect"
                     r"\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best_fp32.engine")

detector = TensorrtDetector(engine_path=model_engine_path)
detector.live_detection()
