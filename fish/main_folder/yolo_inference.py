from ultralytics import YOLO
import cv2
import time
import dxcam


def live_detection_optimized(inference_model):
    camera = dxcam.create(output_idx=0)
    camera.start(target_fps=120)
    prev_time = 1
    current_time = 2

    while True:
        t0 = time.perf_counter()
        frame = camera.grab()
        # frame = camera.get_latest_frame()

        if frame is None:
            continue

        t1 = time.perf_counter()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = inference_model(frame, imgsz=640, iou=0.4, conf=0.7, verbose=False)
        t2 = time.perf_counter()

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            if len(boxes) >= 2:
                print(len(boxes))

            for box in boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (10,244,244), 2)
        t3 = time.perf_counter()

        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        current_time = time.time()

        display = cv2.resize(frame, (1280, 720))
        cv2.imshow("Detection", display)
        t4 = time.perf_counter()

        print(
            f"dxcam={(t1 - t0) * 1000:.2f} ms, "
            f"inference={(t2 - t1) * 1000:.2f} ms, "
            f"draw_boxes={(t3 - t2) * 1000:.2f} ms, "
            f"fps_counter={(t4 - t3) * 1000:.2f} ms, "
            f"summ={(t4 - t0) * 1000:.2f} ms, "
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            camera.stop()
            break

    cv2.destroyAllWindows()

model = YOLO(r"C:\py_projects\albic\fish\main_folder\runs\detect\train_float_with_false_positive_e1000_b16_fine_tuning\weights\best.pt")
model.fuse()
live_detection_optimized(model)