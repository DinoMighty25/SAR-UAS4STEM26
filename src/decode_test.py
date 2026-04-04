
import cv2
import math
import numpy as np
import time
import threading
import queue

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

from qr_decode_simple import decode_qr, bbox_to_polygon

MODEL_PATH = "/home/drone/SAR-UAS4STEM26/models/qr_seg_imx_export/qr_seg_imx_output/network.rpk"
CONF_THRESHOLD = 0.3


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
    input_w, input_h = imx.get_input_size()

    detections = []
    for i in range(len(scores)):
        conf = float(scores[i])
        if conf < CONF_THRESHOLD:
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

        detections.append({
            'bbox': (bx1, by1, bx2, by2),
            'confidence': conf,
        })

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


def decode_worker(decode_q, result_lock, result_holder):
    while True:
        try:
            item = decode_q.get(timeout=1)
        except queue.Empty:
            continue
        if item is None:
            break

        frame, poly, bbox = item
        try:
            text = decode_qr(frame, poly, bbox)
        except Exception as e:
            print(f"decode error: {e}")
            text = None

        with result_lock:
            result_holder['text'] = text


def main():
    imx = initialize_imx500(MODEL_PATH)
    picam2 = setup_camera(imx)

    decode_q = queue.Queue(maxsize=1)
    result_holder = {'text': None}
    result_lock = threading.Lock()
    last_printed = None

    t = threading.Thread(target=decode_worker,
                         args=(decode_q, result_lock, result_holder),
                         daemon=True)
    t.start()

    print("QR Reader running. Press 'q' to quit.\n")

    while True:
        request = picam2.capture_request()
        try:
            frame = request.make_array("main")
            metadata = request.get_metadata()
            img_h, img_w = frame.shape[:2]

            detections = parse_detections(imx, picam2, metadata, img_w, img_h)

            with result_lock:
                decoded_text = result_holder['text']

            if decoded_text and decoded_text != last_printed:
                print(f"\n>>> {decoded_text}\n")
                last_printed = decoded_text

            for i, det in enumerate(detections):
                bbox = det['bbox']
                conf = det['confidence']
                poly = bbox_to_polygon(bbox)

                try:
                    decode_q.put_nowait((frame.copy(), poly.copy(), bbox))
                except queue.Full:
                    pass

                x1, y1, x2, y2 = bbox
                if decoded_text:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, decoded_text[:60], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"QR ({conf:.0%})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if not detections:
                with result_lock:
                    result_holder['text'] = None
                    last_printed = None

            n = len(detections)
            status = f"Detected: {n}"
            if decoded_text:
                status += f" | {decoded_text[:40]}"
            cv2.putText(frame, status, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("QR Reader", frame)

        finally:
            request.release()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    decode_q.put(None)
    t.join(timeout=2)
    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
