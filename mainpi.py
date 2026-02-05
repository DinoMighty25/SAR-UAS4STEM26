#given image file path, detect and plot QR codes in the image

from qrdet import QRDetector, _plot_result
from picamera2 import Picamera2
import cv2
from pymavlink import mavutil
import time
import math


if __name__ == '__main__':
    #create detector object and process image
    detector = QRDetector()
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    frame_count = 0
    
    #fovs are variable - change to camera settings
    HFOV = math.radians(78)
    VFOV = math.radians(60)
    target_width_m = 0.2  #target size in meters
    
    #connec to pixhawk
    print("connecting to pixhawk")
    try:
        master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
        mavutil.wait_heartbeat(master)
        print("connected successfully")
    except Exception as e:
        print(f"failed to connect to Pixhawk: {e}")
        master = None #master is none just to continue the program
    
    #other constants for the landing target message
    target_num = 0
    frame_type = mavutil.mavlink.MAV_FRAME_BODY_FRD  #for ArduPilot
    target_type = mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL
    
    
    #send messages at 10hz
    min_time_between_messages = 0.1  # 100ms = 10Hz
    last_message_time = 0
    
    print("starting feed, press q to quit")
    
    while True:
        frame = picam2.capture_array()
        
        cv2.imshow('frame', frame)
        
        detections = detector.detect(image=frame, is_bgr=False, legacy=False)
        
        current_time = time.time()
        
        if len(detections) > 0:
            d = detections[0]
            
            #pixel center coords
            cx, cy = d["cxcy"]
            #pixel width and height
            w, h   = d["wh"]
            #image height and width
            img_h, img_w = d["image_shape"]

            offsetx = (cx - img_w / 2) / (img_w / 2)  # -1 (left) to +1 (right)
            offsety = (cy - img_h / 2) / (img_h / 2)

            angle_x = offsetx * (HFOV / 2)
            angle_y = offsety * (VFOV / 2)
            size_x = (w / img_w) * HFOV
            size_y = (h / img_h) * VFOV

            focal_length_px = (img_w / 2) / math.tan(HFOV / 2)
            distance = (target_width_m * focal_length_px) / w


            print(f"Frame {frame_count}: angle_x: {angle_x:.6f}, angle_y: {angle_y:.6f}, distance: {distance:.4f}, size_x: {size_x:.6f}, size_y: {size_y:.6f}")
            
            #send landing target message to pixhawk
            if master is not None and (current_time - last_message_time) >= min_time_between_messages:
                try:
                    time_usec = int(current_time * 1e6)
                    master.mav.landing_target_send(
                        time_usec,       #timestamp (microseconds)
                        target_num,      #target ID
                        frame_type,      #coordinate frame
                        angle_x,         #angle_x (radians)
                        angle_y,         #angle_y (radians)
                        distance,        #distance (meters)
                        size_x,          #size_x (radians)
                        size_y,          #size_y (radians)
                        [0,0,0,0],       #unused quaternion for this mode
                        target_type,     #target type
                        0                #position_valid = 0 (using angles)
                    )
                    last_message_time = current_time
                except Exception as e:
                    print(f"error raised while sending message: {e}")
        else:
            print(f"No QR code found")

        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #cleanup
    picam2.stop()
    cv2.destroyAllWindows()
    
    if master is not None:
        print("Closing MAVLink connection...")

#unused function, alternative to the inline code
def send_landing_target(x_angle, y_angle, size_x, size_y, distance, master):
    try:
        master.mav.landing_target_send(
            int(time.time() * 1e6),               #time_usec
            0,                                    #target_num
            mavutil.mavlink.MAV_FRAME_BODY_FRD,  #frame
            x_angle,                              #angle_x (in radians)
            y_angle,                              #angle_y (in radians)
            distance,                             #distance (in meters)
            size_x,                               #size_x (in radians)
            size_y,                               #size_y (in radians)
            [0,0,0,0],                            #unused quaternion
            mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL,  #target type
            0                                     #position_valid = 0
        )
    except Exception as e:
        print(f"error raised while sending message: {e}")
