
import cv2
import math
import numpy as np
import time
import threading
import queue

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

from qr_decode_with_warp import decode_qr, bbox_to_polygon

MODEL_PATH = "/home/drone/SAR-UAS4STEM26/models/qr_seg_imx_export/qr_seg_imx_output/network.rpk"
MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.15


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


def parse_detections(imx, picam2, metadata, img_width, img_height):
    outputs = imx.get_outputs(metadata, add_batch=False)
    if outputs is None or len(outputs) < 2:
        return []

    bboxes = outputs[0]
    scores = outputs[1]

    mask_coeffs = None
    mask_protos = None

    if len(outputs) >= 4:
        for idx in range(2, len(outputs)):
            tensor = outputs[idx]
            if tensor.ndim == 2 and tensor.shape[1] == 32:
                mask_coeffs = tensor
            elif tensor.ndim == 3 and tensor.shape[0] == 32:
                mask_protos = tensor
            elif tensor.ndim == 4 and tensor.shape[1] == 32:
                mask_protos = tensor[0]

    has_masks = mask_coeffs is not None and mask_protos is not None

    try:
        num_detections = len(scores)
        for idx in range(2, len(outputs)):
            t = outputs[idx]
            if t.ndim <= 1 or (t.ndim == 1 and t.shape[0] == 1):
                num_detections = int(t.item() if hasattr(t, 'item') else t.flatten()[0])
                break
    except:
        num_detections = len(scores)

    detections = []
    for i in range(min(num_detections, len(bboxes))):
        conf = float(scores[i])
        if conf < CONF_THRESHOLD:
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

        polygon = None
        if has_masks and i < len(mask_coeffs):
            polygon = _extract_polygon(
                mask_coeffs[i], mask_protos,
                (x1, y1, x2, y2), img_width, img_height
            )

        detections.append({
            'bbox': (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)),
            'confidence': conf,
            'polygon': polygon,
        })

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


def _extract_polygon(coefficients, protos, bbox, img_w, img_h):
    try:
        proto_h, proto_w = protos.shape[1], protos.shape[2]

        mask = coefficients @ protos.reshape(32, -1)
        mask = mask.reshape(proto_h, proto_w)
        mask = 1.0 / (1.0 + np.exp(-mask))

        x1, y1, x2, y2 = bbox
        scale_x = proto_w / img_w
        scale_y = proto_h / img_h
        px1 = max(0, int(x1 * scale_x))
        py1 = max(0, int(y1 * scale_y))
        px2 = min(proto_w, int(x2 * scale_x))
        py2 = min(proto_h, int(y2 * scale_y))

        cropped = np.zeros_like(mask)
        cropped[py1:py2, px1:px2] = mask[py1:py2, px1:px2]

        binary = (cropped > 0.5).astype(np.uint8) * 255
        binary = cv2.resize(binary, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 100:
            return None

        epsilon = 0.015 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)

        return cnt.reshape(-1, 2).astype(np.float32)

    except Exception:
        return None


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
            result_holder['bbox'] = bbox


def main():
    imx = initialize_imx500(MODEL_PATH)
    picam2 = setup_camera(imx)

    decode_q = queue.Queue(maxsize=1)
    result_holder = {'text': None, 'bbox': None}
    result_lock = threading.Lock()

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

            for i, det in enumerate(detections):
                bbox = det['bbox']
                conf = det['confidence']
                poly = det['polygon'] if det['polygon'] is not None else bbox_to_polygon(bbox)

                try:
                    decode_q.put_nowait((frame.copy(), poly.copy(), bbox))
                except queue.Full:
                    pass

                if det['polygon'] is not None:
                    pts = det['polygon'].astype(int).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

                x1, y1, x2, y2 = bbox
                if decoded_text:
                    print(f"[QR {i}] {decoded_text}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, decoded_text[:50], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"QR ({conf:.0%})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if not detections:
                with result_lock:
                    result_holder['text'] = None

            n = len(detections)
            status = f"Detected: {n}"
            if decoded_text:
                status += f" | Decoded: {decoded_text[:30]}"
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
