#!/usr/bin/env python3

from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import cv2
import threading
import queue
import time
import numpy as np

from qr_decode import safe_crop, try_decode_crop

CONFIDENCE_THRESHOLD = 0.3
MODEL_PATH = "/home/drone/SAR-UAS4STEM26/models/qr_seg_imx_export/qr_seg_imx_output/network.rpk"

RESULT_PERSIST_SEC = 3.0


def initialize_imx500(model_path):
    print("loading model...")
    imx = IMX500(model_path)
    intrinsics = imx.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    intrinsics.update_with_defaults()
    print("model loaded")
    print(f"  input size: {imx.get_input_size()}")
    print(f"  intrinsics task: {intrinsics.task}")
    if hasattr(intrinsics, 'bbox_normalization'):
        print(f"  bbox_normalization: {intrinsics.bbox_normalization}")
    if hasattr(intrinsics, 'bbox_order'):
        print(f"  bbox_order: {intrinsics.bbox_order}")
    return imx, intrinsics


def setup_camera(imx, intrinsics):
    print("starting camera...")
    picam2 = Picamera2(imx.camera_num)
    controls = {}
    if intrinsics.inference_rate is not None:
        controls["FrameRate"] = intrinsics.inference_rate

    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls=controls,
        buffer_count=12
    )
    imx.show_network_fw_progress_bar()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("camera ready")
    return picam2


def parse_detections(imx, picam2, metadata, conf_threshold):
    """Parse model outputs into detections with correct coordinate conversion.

    convert_inference_coords expects:
        coords: (y0, x0, y1, x1) — all values NORMALIZED to 0.0-1.0 range
    It returns:
        (x, y, w, h) — in ISP output (main stream) pixel coordinates

    The function internally:
        1. Unpacks as: y0, x0, y1, x1 = coords
        2. Scales to full sensor resolution (4056x3040)
        3. Maps through ScalerCrop to ISP output coordinates
    """
    outputs = imx.get_outputs(metadata, add_batch=True)
    if outputs is None:
        return []

    # get the actual model input dimensions from the firmware
    input_w, input_h = imx.get_input_size()

    # check intrinsics for bbox format info
    intrinsics = imx.network_intrinsics
    bbox_order = getattr(intrinsics, 'bbox_order', 'yx') if intrinsics else 'yx'
    bbox_normalization = getattr(intrinsics, 'bbox_normalization', False) if intrinsics else False

    # With add_batch=True, outputs[0] has shape (1, N, 4), outputs[1] has shape (1, N)
    # Index [0] to remove the batch dimension
    try:
        bboxes = outputs[0][0]   # shape: (N, 4)
        scores = outputs[1][0]   # shape: (N,)
    except (IndexError, TypeError):
        bboxes = outputs[0]
        scores = outputs[1]

    # Determine number of valid detections
    try:
        if len(outputs) > 3:
            num_dets = outputs[3]
            num_detections = int(num_dets.item() if hasattr(num_dets, 'item') else num_dets.flatten()[0])
        else:
            num_detections = len(scores)
    except Exception:
        num_detections = len(scores)

    # Normalize bboxes to 0.0-1.0 range if the intrinsics say so
    # Official demo: if bbox_normalization flag is set, divides by input_h
    if bbox_normalization:
        bboxes = bboxes / input_h

    detections = []
    for i in range(min(num_detections, len(bboxes))):
        conf = float(scores[i])
        if conf < conf_threshold:
            continue

        box = bboxes[i]

        # Handle bbox ordering:
        # Default IMX500 order is "yx": (y0, x0, y1, x1)
        # Some models use "xy": (x0, y0, x1, y1) — need to swap to yx
        if bbox_order == "xy":
            # Model gives (x0, y0, x1, y1), convert to (y0, x0, y1, x1)
            coords = (float(box[1]), float(box[0]), float(box[3]), float(box[2]))
        else:
            # Model already gives (y0, x0, y1, x1) — pass through
            coords = tuple(float(v) for v in box)

        max_coord = max(abs(c) for c in coords)
        if max_coord > 1.5:
            coords = (
                coords[0] / input_h,   # y0
                coords[1] / input_w,   # x0
                coords[2] / input_h,   # y1
                coords[3] / input_w,   # x1
            )

        # convert_inference_coords maps normalized (y0,x0,y1,x1) -> ISP pixel (x,y,w,h)
        try:
            x, y, w, h = imx.convert_inference_coords(coords, metadata, picam2)
        except Exception as e:
            print(f"coord conversion error: {e}")
            continue

        if w <= 0 or h <= 0:
            continue

        detections.append({
            'bbox': (int(x), int(y), int(x + w), int(y + h)),
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
    imx, intrinsics = initialize_imx500(MODEL_PATH)
    picam2 = setup_camera(imx, intrinsics)

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

                detections = parse_detections(imx, picam2, metadata, CONFIDENCE_THRESHOLD)

                # read the persisted decode result once per frame
                with result_lock:
                    decoded_text = result_holder['text']
                    last_time = result_holder['last_decode_time']

                if detections:
                    detection_count += 1

                    # submit highest-confidence crop for decoding
                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        crop = safe_crop(frame, x1, y1, x2, y2)
                        if crop is not None and crop.size > 0:
                            try:
                                decode_q.put_nowait(crop.copy())
                                break
                            except queue.Full:
                                break

                    # draw all detections
                    for detection in detections:
                        draw_detection(frame, detection, decoded_text)

                    cv2.putText(frame, f"detections: {detection_count}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    # let decoded text persist for a few seconds after detection disappears
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
