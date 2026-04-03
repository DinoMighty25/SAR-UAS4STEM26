#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
import cv2
import threading
import queue
import time
import math
import numpy as np

from qr_decode import safe_crop, try_decode_crop
from precision_land import (
    connect_pixhawk, send_heartbeat, calculate_landing_target,
    send_landing_target, MESSAGE_RATE_HZ, PIXHAWK_PORT
)

CONFIDENCE_THRESHOLD = 0.3
MODEL_PATH = "/home/drone/SAR-UAS4STEM26/models/qr_seg_imx_export/qr_seg_imx_output/network.rpk"

MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 512


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
    except:
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

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            'bbox': (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)),
            'center': (cx, cy),
            'size': (x2 - x1, y2 - y1),
            'confidence': conf
        })

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


def draw_detection(frame, detection, angle_x, angle_y, distance, decoded_text=None):
    x1, y1, x2, y2 = detection['bbox']
    cx, cy = int(detection['center'][0]), int(detection['center'][1])
    color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

    img_cx, img_cy = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.drawMarker(frame, (img_cx, img_cy), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
    cv2.line(frame, (img_cx, img_cy), (cx, cy), (255, 0, 0), 1)

    cv2.putText(frame, f"conf: {detection['confidence']:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"dist: {distance:.2f}m",
                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"angle: {math.degrees(angle_x):.1f}, {math.degrees(angle_y):.1f}",
                (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if decoded_text:
        cv2.putText(frame, f"qr: {decoded_text}",
                    (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


def decode_worker(decode_q, result_holder, result_lock):
    while True:
        try:
            crop = decode_q.get(timeout=1)
            if crop is None:
                break
            text = try_decode_crop(crop)
            if text is not None:
                print(f"decoded: {text}")
                with result_lock:
                    result_holder['text'] = text
            else:
                print("decode failed")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"decode error: {e}")


def main():
    imx = initialize_imx500(MODEL_PATH)
    picam2 = setup_camera(imx)
    master = connect_pixhawk()

    decode_q = queue.Queue(maxsize=1)
    result_holder = {'text': None}
    result_lock = threading.Lock()
    t = threading.Thread(target=decode_worker, args=(decode_q, result_holder, result_lock), daemon=True)
    t.start()

    min_msg_interval = 1.0 / MESSAGE_RATE_HZ
    last_msg_time = 0
    last_heartbeat_time = 0

    frame_count = 0
    detection_count = 0
    msg_count = 0

    from precision_land import CAMERA_HFOV, CAMERA_VFOV, QR_SIZE_METERS
    print(f"fov: {math.degrees(CAMERA_HFOV):.1f} x {math.degrees(CAMERA_VFOV):.1f} deg")
    print(f"qr size: {QR_SIZE_METERS}m | mavlink: {MESSAGE_RATE_HZ}hz")
    print("q to quit\n")

    try:
        while True:
            request = picam2.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
                img_h, img_w = frame.shape[:2]

                detections = parse_detections(imx, picam2, metadata, img_w, img_h, CONFIDENCE_THRESHOLD)
                current_time = time.time()

                if master and (current_time - last_heartbeat_time) >= 1.0:
                    send_heartbeat(master)
                    last_heartbeat_time = current_time

                if detections:
                    detection = detections[0]
                    detection_count += 1

                    x1, y1, x2, y2 = detection['bbox']
                    crop = safe_crop(frame, x1, y1, x2, y2)
                    if crop is not None and crop.size > 0:
                        try:
                            decode_q.put_nowait(crop.copy())
                        except queue.Full:
                            pass

                    with result_lock:
                        decoded_text = result_holder['text']

                    result = calculate_landing_target(detection, img_w, img_h)
                    if result:
                        angle_x, angle_y, distance, size_x, size_y = result

                        if master and (current_time - last_msg_time) >= min_msg_interval:
                            try:
                                send_landing_target(master, angle_x, angle_y, distance, size_x, size_y)
                                last_msg_time = current_time
                                msg_count += 1
                            except Exception:
                                pass

                        draw_detection(frame, detection, angle_x, angle_y, distance, decoded_text)

                    cv2.putText(frame, f"detections: {detection_count}  msgs: {msg_count}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    with result_lock:
                        result_holder['text'] = None
                    cv2.putText(frame, "searching...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('QR landing target', frame)
                frame_count += 1

            finally:
                request.release()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        decode_q.put(None)
        print(f"\n{frame_count} frames | {detection_count} detections | {msg_count} messages")
        picam2.stop()
        cv2.destroyAllWindows()
        if master:
            master.close()


if __name__ == '__main__':
    main()
