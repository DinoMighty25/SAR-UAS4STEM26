#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
import cv2
import threading
import queue
import time
import math
import numpy as np

from qr_decode_with_warp import decode_qr, bbox_to_polygon, send_qr
from precision_land import *

CONFIDENCE_THRESHOLD = 0.3
MODEL_PATH = "/home/drone/SAR-UAS4STEM26/models/qr_seg_imx_export/qr_seg_imx_output/network.rpk"


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


def parse_detections(imx, picam2, metadata, img_w, img_h):
    outputs = imx.get_outputs(metadata, add_batch=False)
    if outputs is None:
        return []

    if not hasattr(parse_detections, '_printed'):
        parse_detections._printed = True
        for i, o in enumerate(outputs):
            print(f"  output[{i}]: shape={o.shape} dtype={o.dtype}")

    if len(outputs) < 2:
        return []

    boxes = outputs[0]
    scores = outputs[1]

    mask_coeffs = None
    mask_protos = None
    if len(outputs) >= 5:
        if outputs[3].ndim == 2 and outputs[3].shape[-1] == 32:
            mask_coeffs = outputs[3]
        if outputs[4].ndim == 3 and outputs[4].shape[0] == 32:
            mask_protos = outputs[4]
    elif len(outputs) >= 4:
        for idx in range(2, len(outputs)):
            t = outputs[idx]
            if t.ndim == 2 and t.shape[-1] == 32:
                mask_coeffs = t
            elif t.ndim == 3 and t.shape[0] == 32:
                mask_protos = t

    has_masks = mask_coeffs is not None and mask_protos is not None
    input_w, input_h = imx.get_input_size()

    detections = []
    for i in range(len(scores)):
        conf = float(scores[i])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x0, y0, x1, y1 = [float(v) for v in boxes[i]]
        coords = (y0 / input_h, x0 / input_w, y1 / input_h, x1 / input_w)
        ix, iy, iw, ih = imx.convert_inference_coords(coords, metadata, picam2)

        bx1 = max(0, min(int(ix), img_w))
        by1 = max(0, min(int(iy), img_h))
        bx2 = max(0, min(int(ix + iw), img_w))
        by2 = max(0, min(int(iy + ih), img_h))

        if bx2 - bx1 < 10 or by2 - by1 < 10:
            continue

        polygon = None
        if has_masks and i < len(mask_coeffs):
            polygon = _extract_polygon(
                mask_coeffs[i], mask_protos,
                (x0, y0, x1, y1),
                (bx1, by1, bx2, by2),
                input_w, input_h
            )

        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2

        detections.append({
            'bbox': (bx1, by1, bx2, by2),
            'center': (cx, cy),
            'size': (bx2 - bx1, by2 - by1),
            'confidence': conf,
            'polygon': polygon,
        })

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


def _extract_polygon(coefficients, protos, raw_bbox, img_bbox, input_w, input_h):
    try:
        c, proto_h, proto_w = protos.shape
        raw = coefficients.astype(np.float32) @ protos.reshape(c, -1).astype(np.float32)
        raw = raw.reshape(proto_h, proto_w)
        mask = 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))

        sx = proto_w / input_w
        sy = proto_h / input_h
        rx1, ry1, rx2, ry2 = raw_bbox
        px1 = max(0, int(rx1 * sx))
        py1 = max(0, int(ry1 * sy))
        px2 = min(proto_w, int(rx2 * sx))
        py2 = min(proto_h, int(ry2 * sy))

        if px2 - px1 < 2 or py2 - py1 < 2:
            return None

        patch = mask[py1:py2, px1:px2]
        P = 200
        patch_big = cv2.resize(patch, (P, P), interpolation=cv2.INTER_LINEAR)
        binary = (patch_big > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 50:
            return None

        epsilon = 0.012 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        pts = cnt.reshape(-1, 2).astype(np.float32)

        ix1, iy1, ix2, iy2 = img_bbox
        out = np.zeros_like(pts)
        out[:, 0] = ix1 + (pts[:, 0] / P) * (ix2 - ix1)
        out[:, 1] = iy1 + (pts[:, 1] / P) * (iy2 - iy1)
        return out

    except Exception:
        return None


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

    if detection.get('polygon') is not None:
        pts = detection['polygon'].astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (255, 255, 0), 2)


def decode_worker(decode_q, result_holder, result_lock):
    while True:
        try:
            item = decode_q.get(timeout=1)
            if item is None:
                break
            frame, poly, bbox = item
            text = decode_qr(frame, poly, bbox)
            with result_lock:
                result_holder['text'] = text
            if text:
                print(f"decoded: {text}")
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

    # last known landing target, shared with mavlink sender thread
    target_lock = threading.Lock()
    target_holder = {'data': None}  # (angle_x, angle_y, distance, size_x, size_y)
    mavlink_running = threading.Event()
    mavlink_running.set()

    def mavlink_sender():
        """Send landing_target + heartbeat at 30Hz, independent of camera FPS."""
        last_hb = 0
        while mavlink_running.is_set():
            now = time.time()

            if master and (now - last_hb) >= 1.0:
                try:
                    send_heartbeat(master)
                except Exception:
                    pass
                last_hb = now

            with target_lock:
                data = target_holder['data']

            if master and data is not None:
                angle_x, angle_y, distance, size_x, size_y = data
                try:
                    send_landing_target(master, angle_x, angle_y, distance, size_x, size_y)
                except Exception:
                    pass

            time.sleep(1.0 / 30)  # 30 Hz

    mav_thread = threading.Thread(target=mavlink_sender, daemon=True)
    mav_thread.start()

    last_qr_send_time = 0
    frame_count = 0
    detection_count = 0

    print(f"fov: {math.degrees(CAMERA_HFOV):.1f} x {math.degrees(CAMERA_VFOV):.1f} deg")
    print(f"qr size: {QR_SIZE_METERS}m | mavlink: 30hz (independent)")
    print("q to quit\n")

    try:
        while True:
            request = picam2.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
                img_h, img_w = frame.shape[:2]

                detections = parse_detections(imx, picam2, metadata, img_w, img_h)
                current_time = time.time()

                if detections:
                    detection = detections[0]
                    detection_count += 1

                    bbox = detection['bbox']
                    poly = detection['polygon'] if detection['polygon'] is not None else bbox_to_polygon(bbox)

                    try:
                        decode_q.put_nowait((frame.copy(), poly.copy(), bbox))
                    except queue.Full:
                        pass

                    with result_lock:
                        decoded_text = result_holder['text']

                    if decoded_text and master and (current_time - last_qr_send_time) >= 2.0:
                        send_qr(decoded_text, master)
                        last_qr_send_time = current_time

                    result = calculate_landing_target(detection, img_w, img_h)
                    if result:
                        angle_x, angle_y, distance, size_x, size_y = result

                        with target_lock:
                            target_holder['data'] = (angle_x, angle_y, distance, size_x, size_y)

                        draw_detection(frame, detection, angle_x, angle_y, distance, decoded_text)

                    cv2.putText(frame, f"detections: {detection_count}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    with result_lock:
                        result_holder['text'] = None
                    with target_lock:
                        target_holder['data'] = None
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
        mavlink_running.clear()
        decode_q.put(None)
        print(f"\n{frame_count} frames | {detection_count} detections")
        picam2.stop()
        cv2.destroyAllWindows()
        if master:
            master.close()


if __name__ == '__main__':
    main()
