#!/usr/bin/env python3
"""
QR Code Landing Target System
Detects QR codes using IMX500 and sends landing target info to Pixhawk
"""

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
import cv2
from pymavlink import mavutil
import time
import math
import numpy as np

# camera setup
CAMERA_HFOV = math.radians(78)
CAMERA_VFOV = math.radians(60)
QR_SIZE_METERS = 0.2  # qr code size in meters

# detection
CONFIDENCE_THRESHOLD = 0.3
MODEL_PATH = "/home/drone/SAR-rpk/qrdet_final2.rpk" # .rpk file contains the IMX500 model format, make it the right path

# MAVLink
PIXHAWK_PORT = '/dev/ttyAMA0'
PIXHAWK_BAUD = 57600
MESSAGE_RATE_HZ = 10

# RC switch control
RC_CHANNEL = 6
RC_ENABLE_VALUE = 2065
RC_TOLERANCE = 50

# model resolution - basically so that the bounding box maps onto the camera's resolution better
MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 512

DEBUG = False

# IMX500 model is loaded
def initialize_imx500(model_path):
    print("Loading QR detection model...")
    imx = IMX500(model_path)
    print("Model loaded")
    return imx

# picamera is launched
def setup_camera(imx):
    print("Setting up camera...")
    picam2 = Picamera2(imx.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    imx.show_network_fw_progress_bar()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("Camera ready")
    return picam2

# pixhawk is connected
def connect_pixhawk():
    print(f"Connecting to Pixhawk on {PIXHAWK_PORT}...")
    try:
        master = mavutil.mavlink_connection(PIXHAWK_PORT, baud=PIXHAWK_BAUD)
        master.wait_heartbeat()
        print("Connected to Pixhawk")
        
        # request RC channel data stream
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,
            10, 1
        )
        return master
    except Exception as e:
        print(f"Warning: Could not connect to Pixhawk: {e}")
        return None

# check if RC switch is in enabled position
# returns (enabled, rc_value) - returns (None, None) if no new data
def check_rc_switch(master):
    if master is None:
        return None, None
    
    # drain buffer to get most recent RC message
    latest_msg = None
    while True:
        msg = master.recv_match(type='RC_CHANNELS', blocking=False)
        if msg is None:
            break
        latest_msg = msg
    
    if latest_msg is None:
        return None, None
    
    rc_val = getattr(latest_msg, f'chan{RC_CHANNEL}_raw', None)
    if rc_val is None:
        return None, None
    
    enabled = abs(rc_val - RC_ENABLE_VALUE) <= RC_TOLERANCE
    return enabled, rc_val

# figure out QR code detections and scale to proper size
def parse_detections(imx, metadata, img_width, img_height, conf_threshold):
    outputs = imx.get_outputs(metadata, add_batch=False)
    
    if outputs is None or len(outputs) < 3:
        return []
    
    bboxes = outputs[0]
    scores = outputs[1]
    
    # get number of detections
    try:
        if len(outputs) > 3:
            num_dets = outputs[3]
            if hasattr(num_dets, 'item'):
                num_detections = int(num_dets.item())
            else:
                num_detections = int(num_dets.flatten()[0])
        else:
            num_detections = len(scores)
    except:
        num_detections = len(scores)
    
    # scale from model input size to actual image size
    scale_x = img_width / MODEL_INPUT_WIDTH
    scale_y = img_height / MODEL_INPUT_HEIGHT
    
    detections = []
    for i in range(min(num_detections, len(bboxes))):
        conf = float(scores[i])
        if conf < conf_threshold:
            continue
        
        # scale bbox from model space (512x512) to image space (1280x720)
        x1, y1, x2, y2 = [float(v) for v in bboxes[i]]
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        
        # clamp to image bounds
        x1 = max(0, min(x1, img_width))
        x2 = max(0, min(x2, img_width))
        y1 = max(0, min(y1, img_height))
        y2 = max(0, min(y2, img_height))
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        detections.append({
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'center': (cx, cy),
            'size': (w, h),
            'confidence': conf
        })
    
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections

# calculate all the landing target parameters
def calculate_landing_target(detection, img_width, img_height):
    cx, cy = detection['center']
    width, height = detection['size']
    
    if width <= 0 or height <= 0:
        return None
    
    # offset from center, normalized to [-1, 1]
    offset_x = (cx - img_width / 2) / (img_width / 2)
    offset_y = (cy - img_height / 2) / (img_height / 2)
    
    # convert to angles
    angle_x = offset_x * (CAMERA_HFOV / 2)
    angle_y = offset_y * (CAMERA_VFOV / 2)
    
    # angular size
    size_x = (width / img_width) * CAMERA_HFOV
    size_y = (height / img_height) * CAMERA_VFOV
    
    # distance estimate using pinhole camera model
    focal_length_px = (img_width / 2) / math.tan(CAMERA_HFOV / 2)
    distance = (QR_SIZE_METERS * focal_length_px) / width
    
    return angle_x, angle_y, distance, size_x, size_y

# send the landing target message from the parameters
def send_landing_target(master, angle_x, angle_y, distance, size_x, size_y):
    time_usec = int(time.time() * 1e6)
    
    try:
        master.mav.landing_target_send(
            time_usec,
            0,  # target_num
            int(mavutil.mavlink.MAV_FRAME_BODY_FRD),
            float(angle_x),
            float(angle_y),
            float(distance),
            float(size_x),
            float(size_y),
            int(mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL),
            0  # position_valid
        )
    except Exception as e:
        print(f"    MAVLink error: {e}")
        raise

# draw the bounding box in the camera frame
def draw_detection(frame, detection, angle_x, angle_y, distance, enabled):
    x1, y1, x2, y2 = detection['bbox']
    cx, cy = int(detection['center'][0]), int(detection['center'][1])
    
    color = (0, 255, 0) if enabled else (0, 255, 255)  # green if active, yellow if not
    
    # bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # center crosshair
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
    
    # image center reference
    img_cx = frame.shape[1] // 2
    img_cy = frame.shape[0] // 2
    cv2.drawMarker(frame, (img_cx, img_cy), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
    cv2.line(frame, (img_cx, img_cy), (cx, cy), (255, 0, 0), 1)
    
    # Info text
    cv2.putText(frame, f"Conf: {detection['confidence']:.2f}", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"Dist: {distance:.2f}m", 
                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"Angle: {math.degrees(angle_x):.1f}°, {math.degrees(angle_y):.1f}°", 
                (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# main loop
def main():
    imx = initialize_imx500(MODEL_PATH)
    picam2 = setup_camera(imx)
    master = connect_pixhawk()
    
    min_msg_interval = 1.0 / MESSAGE_RATE_HZ
    last_msg_time = 0
    last_rc_val = None
    rc_enabled = False  # cached state
    
    frame_count = 0
    detection_count = 0
    msg_count = 0
    
    print(f"Camera FOV: {math.degrees(CAMERA_HFOV):.1f}° x {math.degrees(CAMERA_VFOV):.1f}°")
    print(f"QR size: {QR_SIZE_METERS}m")
    print(f"MAVLink rate: {MESSAGE_RATE_HZ}Hz")
    print(f"RC{RC_CHANNEL} enable value: {RC_ENABLE_VALUE} (+/- {RC_TOLERANCE})")
    print("Press Ctrl+C or 'q' to quit\n")
    
    try:
        while True:
            request = picam2.capture_request()
            
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
                img_h, img_w = frame.shape[:2]
                
                # check RC switch - only update state when we get new data
                new_enabled, rc_val = check_rc_switch(master)
                if rc_val is not None:
                    last_rc_val = rc_val
                    rc_enabled = new_enabled
                
                detections = parse_detections(imx, metadata, img_w, img_h, CONFIDENCE_THRESHOLD)
                current_time = time.time()
                
                # show RC status on screen
                if last_rc_val is not None:
                    status_color = (0, 255, 0) if rc_enabled else (0, 0, 255)
                    status_txt = "ON" if rc_enabled else "OFF"
                    cv2.putText(frame, f"RC{RC_CHANNEL}: {last_rc_val} [{status_txt}]", 
                               (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                else:
                    cv2.putText(frame, f"RC{RC_CHANNEL}: --", 
                               (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if detections:
                    detection = detections[0]
                    detection_count += 1
                    
                    if DEBUG:
                        cx, cy = detection['center']
                        w, h = detection['size']
                        print(f"Frame {frame_count} - QR detected at ({cx:.0f}, {cy:.0f}), "
                              f"size {w:.0f}x{h:.0f}px, conf {detection['confidence']:.3f}")
                    
                    result = calculate_landing_target(detection, img_w, img_h)
                    
                    if result:
                        angle_x, angle_y, distance, size_x, size_y = result
                        
                        if DEBUG:
                            print(f"  Angle: {math.degrees(angle_x):+.1f}°, "
                                  f"{math.degrees(angle_y):+.1f}°, Dist: {distance:.2f}m")
                        
                        # send to Pixhawk at specified rate if rc is enabled
                        if rc_enabled and master and (current_time - last_msg_time) >= min_msg_interval:
                            try:
                                send_landing_target(master, angle_x, angle_y, distance, size_x, size_y)
                                last_msg_time = current_time
                                msg_count += 1
                                if DEBUG:
                                    print(f"  -> MAVLink sent ({msg_count} total)")
                            except Exception as e:
                                if DEBUG:
                                    print(f"  -> MAVLink failed: {e}")
                        elif not rc_enabled and DEBUG:
                            print(f"  -> Skipped, RC{RC_CHANNEL}={last_rc_val}")
                        
                        draw_detection(frame, detection, angle_x, angle_y, distance, rc_enabled)
                        
                        cv2.putText(frame, f"Detections: {detection_count}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Messages: {msg_count}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Searching...", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('QR Landing Target', frame)
                frame_count += 1
                
            finally:
                request.release()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        print(f"\nStats: {frame_count} frames, {detection_count} detections, {msg_count} messages")
        picam2.stop()
        cv2.destroyAllWindows()
        if master:
            master.close()


if __name__ == '__main__':
    main()
