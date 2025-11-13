#given image file path, detect and plot QR codes in the image

from qrdet import QRDetector, _plot_result
import cv2
from pymavlink import mavutil
import time
import math


if __name__ == '__main__':
    #create detector object and process image
    detector = QRDetector()
    # image = cv2.cvtColor(cv2.imread(filename='/Users/Troy/Downloads/SAR UAS/qrdet/resources/qreader_test_image.jpeg'), code=cv2.COLOR_BGR2RGB)
    # detections = detector.detect(image=image, is_bgr=False, legacy=False)
    # _plot_result(image=image, detections=detections)
    # d = detections[0]
    cap = cv2.VideoCapture(0)
    frame_count = 0
    #fovs are variable - change to camera settings
    HFOV = math.radians(78)
    VFOV = math.radians(60)
    target_width_m = 0.2  #target size in meters

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if not ret:
            print("Failed to grab frame")
            break
        detections = detector.detect(image=frame, is_bgr=True, legacy=False)
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    


    # master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)

    # mavutil.wait_heartbeat(master)
    # print("heartbeat received")

    # # Define parameters
    # time_usec = int(time.time() * 1e6)
    # target_num = 0
    # frame = mavutil.mavlink.MAV_FRAME_BODY_FRD  # for ArduPilot
    # target_type = mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL

    # master.mav.landing_target_send(
    #     time_usec,       # timestamp (microseconds)
    #     target_num,      # target ID
    #     frame,           # coordinate frame
    #     angle_x, angle_y,
    #     distance,
    #     size_x, size_y,
    #     [0,0,0,0],       # unused quaternion for this mode
    #     target_type,
    #     0                # position_valid = 0 (since we’re using angles)
    # )
    #print(f"angle_x: {angle_x}, angle_y: {angle_y}, distance: {distance}, size_x: {size_x}, size_y: {size_y}")




def send_landing_target(x_angle, y_angle, size_x, size_y, distance):
    try:
        master.mav.landing_target_send(
            int(time.time() * 1e6),               # time_usec
            0,                                    # target_num
            mavutil.mavlink.MAV_FRAME_BODY_FRD,  # frame
            x_angle,                              # angle_x (in radians)
            y_angle,                              # angle_y (in radians)
            distance,                             # distance (in meters)
            size_x,                                    # size_x (unused)
            size_y                                     # size_y (unused)
        )
    except Exception as e:
        print(f"Error sending landing target: {e}")
