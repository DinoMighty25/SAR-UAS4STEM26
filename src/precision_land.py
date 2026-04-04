#!/usr/bin/env python3

from pymavlink import mavutil
import time
import math

CAMERA_HFOV = math.radians(78)
CAMERA_VFOV = math.radians(60)
QR_SIZE_METERS = 0.2

PIXHAWK_PORT = 'udpout:192.168.1.193:14550'
MESSAGE_RATE_HZ = 10


def connect_pixhawk(port=PIXHAWK_PORT):
    print(f"connecting to pixhawk on {port}...")
    try:
        master = mavutil.mavlink_connection(
            port,
            source_system=2,
            source_component=191
        )
        master.wait_heartbeat()
        print(f"pixhawk connected (target sysid {master.target_system})")
        return master
    except Exception as e:
        print(f"warning: pixhawk not connected: {e}")
        return None


def send_heartbeat(master):
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0
    )


def calculate_landing_target(detection, img_width, img_height):
    cx, cy = detection['center']
    width, height = detection['size']

    if width <= 0 or height <= 0:
        return None

    offset_x = (cx - img_width / 2) / (img_width / 2)
    offset_y = (cy - img_height / 2) / (img_height / 2)

    angle_x = offset_x * (CAMERA_HFOV / 2)
    angle_y = offset_y * (CAMERA_VFOV / 2)

    size_x = (width / img_width) * CAMERA_HFOV
    size_y = (height / img_height) * CAMERA_VFOV

    focal_length_px = (img_width / 2) / math.tan(CAMERA_HFOV / 2)
    distance = (QR_SIZE_METERS * focal_length_px) / width

    return angle_x, angle_y, distance, size_x, size_y


def send_landing_target(master, angle_x, angle_y, distance, size_x, size_y):
    try:
        master.mav.landing_target_send(
            int(time.time() * 1e6),
            0,
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            float(angle_x),
            float(angle_y),
            float(distance),
            float(size_x),
            float(size_y)
        )
    except Exception as e:
        print(f"mavlink error: {e}")
        raise
