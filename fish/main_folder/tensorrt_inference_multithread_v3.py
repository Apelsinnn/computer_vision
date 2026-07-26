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
class PreprocessedFrame:
    frame: np.ndarray = None
    batch: np.ndarray = None

@dataclass(slots=True)
class PredictionResult:
    predictions: list = None
    frame: np.ndarray = None

# ping pong buffering
class TensorRTDetector:
    """Пайплайн детекции изображений для модели TensorRT в реальном времени."""

    def __init__(self, engine_path: str):
        self._engine_path = engine_path
        self._running = False
        self._camera = None
        self._execution_contexts = None
        self._device_input_buffers = None
        self._device_output_buffers = None
        self._cuda_streams = None
        self._captured_frame = None
        self._preprocessed_frame = None
        self._inference_data = PredictionResult()
        self._fps = None
        self._converter = None
        self._host_input_buffers = None

    def _capture_loop(self):
        """Получает кадры с экрана."""
        while self._running:
            frame = self._camera.grab()
            # frame = self.camera.get_latest_frame()

            if frame is None:
                continue

            self._captured_frame = frame

    def _preprocess_loop(self):
        """Подготавливает кадры к входному формату модели."""
        while self._captured_frame is None:
            print("Ожидание первого кадра!")
            time.sleep(0.2)

        while self._running:
            frame = self._captured_frame.copy() # type: ignore
            img = self._converter.letterbox(frame)
            img = img.astype(np.float32)
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            self._preprocessed_frame = PreprocessedFrame(frame=frame, batch=img)

    def _inference_loop(self):
        """Находит объекты на кадре."""
        prev_time = 2
        buffer_index = 0

        while self._preprocessed_frame is None:
            print("Ожидание обработанного кадра!")
            time.sleep(0.4)

        while self._running:
            t0 = time.perf_counter()

            buffer_index = (buffer_index + 1) % 2
            previous_index = 1 - buffer_index
            frame, batch = self._preprocessed_frame.frame, self._preprocessed_frame.batch # type: ignore

            self._cuda_events[buffer_index].synchronize()
            np.copyto(self._host_input_buffers[buffer_index].numpy(), batch)
            self._device_input_buffers[buffer_index].copy_(
                self._host_input_buffers[buffer_index],
                non_blocking=True
            )

            with torch.cuda.stream(self._cuda_streams[buffer_index]):
                self._execution_contexts[buffer_index].execute_async_v3(
                    self._cuda_streams[buffer_index].cuda_stream
                )
                self._cuda_events[buffer_index].record(self._cuda_streams[buffer_index])

            self._cuda_events[previous_index].synchronize()
            outputs = self._device_output_buffers[previous_index].cpu().numpy()
            predictions = outputs[0]
            predictions = predictions[predictions[:, 4] > 0.7]
            current_time = time.perf_counter()
            self._fps = 1 / (current_time - prev_time)
            self._inference_data = PredictionResult(predictions=predictions, frame=frame) # тут надо передавать предыдущий кадр
            prev_time = current_time
            t1 = time.perf_counter()
            # print(f"inference: {(t1 - t0) * 1000:.2f}ms")

    def _display_loop(self):
        """Отображает найденные объекты."""
        while self._inference_data.predictions is None:
            print("Ожидание предсказания модели!")
            time.sleep(0.6)

        while self._running:
            # print(id(self._inference_data))
            predictions, frame = self._inference_data.predictions, self._inference_data.frame
            # t0 = time.perf_counter()

            for predict in predictions: # type: ignore
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

    def _initialize_pipline(self):
        """Задаёт начальные настройки перед запуском основного пайплайна."""
        self._running = True
        self._camera = dxcam.create(output_idx=0)
        self._camera.start(target_fps=240)

        monitor_height = 1920
        monitor_width = 1080
        full_hd_monitor = (monitor_height, monitor_width)
        input_height = 640
        input_width = 640
        model_input_size = (input_height, input_width)
        gray = (114, 114, 114)
        self._converter = ImageConverter(
            original_shape=full_hd_monitor, converted_shape=model_input_size, padding_color=gray
        )

        yolo26_byte_length = 4
        with open(self._engine_path, "rb") as f:
            meta_len = int.from_bytes(
                f.read(yolo26_byte_length),
                byteorder="little"
            )

            metadata_bytes = f.read(meta_len)
            engine_bytes = f.read()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(
            engine_bytes
        )

        self._execution_contexts = [
            engine.create_execution_context(),
            engine.create_execution_context()
        ]

        input_tensor_name = "images"
        output_tensor_name = "output0"

        for tensor_settings in range(engine.num_io_tensors):
            name = engine.get_tensor_name(tensor_settings)

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_tensor_name = name

            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_tensor_name = name

        batch_size = 1
        channels = 3
        model_input_shape = (batch_size, channels, input_height, input_width)
        max_detections_count = 300
        prediction_data_count = 6
        model_output_shape = (batch_size, max_detections_count, prediction_data_count)

        self._device_input_buffers = [
            torch.empty(
                size=model_input_shape,
                dtype=torch.float32,
                device="cuda"
            ),
            torch.empty(
                size=model_input_shape,
                dtype=torch.float32,
                device="cuda"
            )
        ]

        input_ptr_0 = self._device_input_buffers[0].data_ptr()
        input_ptr_1 = self._device_input_buffers[1].data_ptr()

        self._execution_contexts[0].set_tensor_address(
            input_tensor_name,
            input_ptr_0,
        )

        self._execution_contexts[1].set_tensor_address(
            input_tensor_name,
            input_ptr_1,
        )

        self._host_input_buffers = [
            torch.empty(
                size=model_input_shape,
                dtype=torch.float32,
                pin_memory=True,
            ),
            torch.empty(
                size=model_input_shape,
                dtype=torch.float32,
                pin_memory=True,
            )
        ]

        self._device_output_buffers = [
            torch.empty(
                size=model_output_shape,
                dtype=torch.float32,
                device="cuda"
            ),
            torch.empty(
                size=model_output_shape,
                dtype=torch.float32,
                device="cuda"
            )
        ]

        output_ptr_0 = self._device_output_buffers[0].data_ptr()
        output_ptr_1 = self._device_output_buffers[1].data_ptr()

        self._execution_contexts[0].set_tensor_address(
            output_tensor_name,
            output_ptr_0,
        )

        self._execution_contexts[1].set_tensor_address(
            output_tensor_name,
            output_ptr_1,
        )

        self._cuda_streams = [
            torch.cuda.Stream(),
            torch.cuda.Stream(),
        ]

        self._cuda_events = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

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
        self._initialize_pipline()
        self._enable_threads()


model_engine_path = (r"C:\py_projects\albic\fish\main_folder\runs\detect"
                     r"\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best_fp32.engine")

detector = TensorRTDetector(engine_path=model_engine_path)
detector.live_detection()
