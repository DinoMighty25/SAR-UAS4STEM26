#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
import cv2
import threading
import queue
import time
import numpy as np

from qr_decode import safe_crop, try_decode_crop

CONFIDENCE_THRESHOLD = 0.3
MODEL_PATH = "/home/drone/SAR-UAS4STEM26/models/qr_seg_imx_export/qr_seg_imx_output/network.rpk"

MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 512

# how long a decoded result stays on screen after detection disappears (seconds)
RESULT_PERSIST_SEC = 1.0


def initialize_imx500(model_path):
    print("loading model...")
    imx = IMX500(model_path)
    print("model loaded")
    return imx


def setup_camera(imx):
    print("starting camera...")
    picam2 = Picamera2(imx.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    imx.show_network_fw_progress_bar()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("camera ready")
    return picam2


def parse_detections(imx, picam2, metadata, img_width, img_height, conf_threshold):
    outputs = imx.get_outputs(metadata, add_batch=False)
    if outputs is None or len(outputs) < 3:
        return []

    bboxes = outputs[0]
    scores = outputs[1]

    try:
        if len(outputs) > 3:
            num_dets = outputs[3]
            num_detections = int(num_dets.item() if hasattr(num_dets, 'item') else num_dets.flatten()[0])
        else:
            num_detections = len(scores)
    except Exception:
        num_detections = len(scores)

    detections = []
    for i in range(min(num_detections, len(bboxes))):
        conf = float(scores[i])
        if conf < conf_threshold:
            continue

        raw_x1, raw_y1, raw_x2, raw_y2 = [float(v) for v in bboxes[i]]
        rel_x1 = raw_x1 / MODEL_INPUT_WIDTH
        rel_y1 = raw_y1 / MODEL_INPUT_HEIGHT
        rel_x2 = raw_x2 / MODEL_INPUT_WIDTH
        rel_y2 = raw_y2 / MODEL_INPUT_HEIGHT

        x, y, w, h = imx.convert_inference_coords(
            (rel_y1, rel_x1, rel_y2, rel_x2), metadata, picam2
        )

        x1 = max(0, min(x, img_width))
        y1 = max(0, min(y, img_height))
        x2 = max(0, min(x + w, img_width))
        y2 = max(0, min(y + h, img_height))

        detections.append({
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'confidence': conf
        })

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


def draw_detection(frame, detection, decoded_text=None):
    x1, y1, x2, y2 = detection['bbox']
    color = (0, 255, 0) if decoded_text else (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"conf: {detection['confidence']:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if decoded_text:
        cv2.putText(frame, f"qr: {decoded_text[:60]}",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def decode_worker(decode_q, result_holder, result_lock):
    """Background thread that decodes QR crops from the queue."""
    while True:
        try:
            crop = decode_q.get(timeout=1)
            if crop is None:
                break
            text = try_decode_crop(crop)
            with result_lock:
                if text is not None:
                    print(f"decoded: {text}")
                    result_holder['text'] = text
                    result_holder['last_decode_time'] = time.monotonic()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"decode error: {e}")


def main():
    imx = initialize_imx500(MODEL_PATH)
    picam2 = setup_camera(imx)

    decode_q = queue.Queue(maxsize=1)
    result_holder = {'text': None, 'last_decode_time': 0.0}
    result_lock = threading.Lock()
    t = threading.Thread(target=decode_worker, args=(decode_q, result_holder, result_lock), daemon=True)
    t.start()

    frame_count = 0
    detection_count = 0
    print("q to quit\n")

    try:
        while True:
            request = picam2.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
                img_h, img_w = frame.shape[:2]

                detections = parse_detections(imx, picam2, metadata, img_w, img_h, CONFIDENCE_THRESHOLD)

                # FIX: read the persisted result once per frame
                with result_lock:
                    decoded_text = result_holder['text']
                    last_time = result_holder['last_decode_time']

                if detections:
                    detection_count += 1

                    # FIX: try ALL detections, not just the first one.
                    # Submit the highest-confidence crop that isn't already queued.
                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        crop = safe_crop(frame, x1, y1, x2, y2)
                        if crop is not None and crop.size > 0:
                            try:
                                decode_q.put_nowait(crop.copy())
                                break  # submitted one crop, move on
                            except queue.Full:
                                break  # worker is busy, don't block

                    # draw all detections
                    for detection in detections:
                        draw_detection(frame, detection, decoded_text)

                    cv2.putText(frame, f"detections: {detection_count}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    if decoded_text and (time.monotonic() - last_time > RESULT_PERSIST_SEC):
                        with result_lock:
                            result_holder['text'] = None
                        decoded_text = None

                    if decoded_text:
                        cv2.putText(frame, f"last qr: {decoded_text[:60]}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "searching...",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('QR decode test', frame)
                frame_count += 1

            finally:
                request.release()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        decode_q.put(None)
        print(f"\n{frame_count} frames | {detection_count} detections")
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
