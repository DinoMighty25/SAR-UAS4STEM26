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

        polygon = None
        if has_masks and i < len(mask_coeffs):
            polygon = _extract_polygon(
                mask_coeffs[i], mask_protos,
                (x0, y0, x1, y1),
                (bx1, by1, bx2, by2),
                input_w, input_h
            )

        detections.append({
            'bbox': (bx1, by1, bx2, by2),
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
