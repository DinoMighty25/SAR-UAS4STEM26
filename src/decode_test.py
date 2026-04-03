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
QR_CLASS_ID = 1       # 0=background, 1=QR — adjust to match your model
MIN_MASK_AREA = 100   # minimum mask pixels for a valid detection
RESULT_PERSIST_SEC = 3.0


def initialize_imx500(model_path):
    print("loading model...")
    imx = IMX500(model_path)
    intrinsics = imx.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "segmentation"
    intrinsics.update_with_defaults()

    input_w, input_h = imx.get_input_size()
    print(f"model loaded  input={input_w}x{input_h}  task={getattr(intrinsics, 'task', '?')}")
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


_diagnosed = False

def diagnose_outputs(outputs):
    """Print tensor info once on first frame — helps debug custom models."""
    global _diagnosed
    if _diagnosed:
        return
    _diagnosed = True

    print(f"\n=== MODEL OUTPUTS ({len(outputs)} tensors) ===")
    for i, out in enumerate(outputs):
        arr = np.array(out)
        unique = np.unique(arr)
        extra = f"unique={unique}" if len(unique) <= 20 else f"{len(unique)} unique values"
        print(f"  [{i}] shape={arr.shape} dtype={arr.dtype} range=[{arr.min():.4f}, {arr.max():.4f}] {extra}")
    print("===\n")


def mask_to_detections(imx, picam2, metadata, qr_class_id=QR_CLASS_ID):
    """Extract bounding boxes from the segmentation mask output."""
    outputs = imx.get_outputs(metadata, add_batch=False)
    if outputs is None:
        return []

    diagnose_outputs(outputs)
    input_w, input_h = imx.get_input_size()
    mask = np.array(outputs[0])

    # collapse to 2D class map regardless of output layout
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[0] <= 4:
            mask = np.argmax(mask, axis=0)
        else:
            mask = np.argmax(mask, axis=2)
    elif mask.ndim == 4:
        mask = mask[0]
        if mask.shape[0] <= 4:
            mask = np.argmax(mask, axis=0)
        else:
            mask = np.argmax(mask, axis=2)

    # build binary mask for the QR class
    if mask.dtype in (np.float32, np.float64):
        if mask.max() <= 1.0:
            binary = (mask > 0.5).astype(np.uint8)
        else:
            binary = (mask.astype(int) == qr_class_id).astype(np.uint8)
    else:
        binary = (mask == qr_class_id).astype(np.uint8)

    # fallback: any non-background pixel (in case class id is wrong)
    if binary.sum() == 0:
        binary = (mask > 0).astype(np.uint8)
        if binary.sum() == 0:
            return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    mask_h, mask_w = binary.shape
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_MASK_AREA:
            continue

        mx, my, mw, mh = cv2.boundingRect(contour)

        # normalize to 0-1 then convert to ISP output coords
        # convert_inference_coords expects (y0, x0, y1, x1) normalized
        coords = (my / mask_h, mx / mask_w, (my + mh) / mask_h, (mx + mw) / mask_w)

        try:
            x, y, w, h = imx.convert_inference_coords(coords, metadata, picam2)
        except Exception as e:
            print(f"coord conversion error: {e}")
            continue

        if w <= 0 or h <= 0:
            continue

        bbox_area = mw * mh
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        confidence = min(fill_ratio + 0.3, 1.0)

        detections.append({
            'bbox': (int(x), int(y), int(x + w), int(y + h)),
            'confidence': round(confidence, 2),
            'mask_area': int(area)
        })

    detections.sort(key=lambda d: d['mask_area'], reverse=True)
    return detections


def draw_detection(frame, detection, decoded_text=None):
    x1, y1, x2, y2 = detection['bbox']
    color = (0, 255, 0) if decoded_text else (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"area: {detection.get('mask_area', '?')}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if decoded_text:
        cv2.putText(frame, f"qr: {decoded_text[:60]}",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def decode_worker(decode_q, result_holder, result_lock):
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

                detections = mask_to_detections(imx, picam2, metadata)

                with result_lock:
                    decoded_text = result_holder['text']
                    last_time = result_holder['last_decode_time']

                if detections:
                    detection_count += 1

                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        crop = safe_crop(frame, x1, y1, x2, y2)
                        if crop is not None and crop.size > 0:
                            try:
                                decode_q.put_nowait(crop.copy())
                                break
                            except queue.Full:
                                break

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
