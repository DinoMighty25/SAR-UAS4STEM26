#!/usr/bin/env python3

# PLND_ENABLED   = 1
# PLND_TYPE      = 1       # companion computer
# PLND_EST_TYPE  = 1       # Kalman filter back on
# PLND_LAG       = 0.10    # estimate; measure later if wobble persists
# PLND_OPTIONS   = 1       # moving target support (hat delivery)
# PLND_XY_DIST_MAX = 5     # meters; allows reacquisition from further off
# LAND_SPEED     = 40      # cm/s final descent
# RCx_OPTION     = 39      # precision loiter on pilot's switch (x = your channel)

from pymavlink import mavutil
from collections import deque
import time
import math
import numpy as np

CAMERA_HFOV = math.radians(66)
CAMERA_VFOV = math.radians(40.4)
QR_SIZE_METERS = 0.914          # last-resort guess only; real size is learned in flight

# Below this altitude the baro/EKF alt is unreliable (ground effect),
# so we switch to pinhole using the learned QR size.
ALT_TRUST_MIN = 1.0

PIXHAWK_PORT = '/dev/serial0'
BAUD = 57600
MESSAGE_RATE_HZ = 10


def connect_pixhawk(port=PIXHAWK_PORT):
    print(f"connecting to pixhawk on {port}...")
    try:
        master = mavutil.mavlink_connection(
            port,
            baud=BAUD,
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


class QRSizeEstimator:
    """Learns the physical QR size (m) while altitude is trusted, so the
    low-altitude pinhole fallback works for any mat size."""

    MIN_SAMPLES = 10

    def __init__(self, maxlen=100):
        self.samples = deque(maxlen=maxlen)

    def update(self, size_px, distance_m, focal_px):
        est = size_px * distance_m / focal_px
        if 0.05 < est < 3.0:            # reject nonsense
            self.samples.append(est)

    def get(self):
        if len(self.samples) < self.MIN_SAMPLES:
            return None
        return float(np.median(self.samples))


def apparent_size_px(detection):
    """Rotation-invariant apparent size of the (square) QR in pixels.

    Uses sqrt(polygon area): the bounding box of a 45-degree-rotated square
    is ~40% too wide, which garbles pinhole distance every time the drone
    yaws. Falls back to bbox width if no polygon."""
    poly = detection.get('polygon')
    if poly is not None and len(poly) >= 3:
        x = poly[:, 0].astype(np.float64)
        y = poly[:, 1].astype(np.float64)
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area > 4.0:
            return math.sqrt(area)
    return float(detection['size'][0])


def _learning_ok(detection, img_width, img_height, margin=8):
    """Only learn QR size from clean detections: confident and not
    clipped at the frame edge (a clipped bbox reads too small)."""
    if detection['confidence'] < 0.5:
        return False
    x1, y1, x2, y2 = detection['bbox']
    return (x1 > margin and y1 > margin
            and x2 < img_width - margin and y2 < img_height - margin)


def calculate_landing_target(detection, img_width, img_height,
                             rel_alt=None, size_est=None):
    cx, cy = detection['center']
    width, height = detection['size']
    if width <= 0 or height <= 0:
        return None

    focal_px = (img_width / 2) / math.tan(CAMERA_HFOV / 2)

    angle_x = math.atan((cx - img_width / 2) / focal_px)
    angle_y = math.atan((cy - img_height / 2) / focal_px)

    size_x = (width / img_width) * CAMERA_HFOV
    size_y = (height / img_height) * CAMERA_VFOV

    size_px = apparent_size_px(detection)

    if rel_alt is not None and rel_alt > ALT_TRUST_MIN:
        # altitude-based slant range: independent of QR size
        distance = rel_alt * math.sqrt(1 + math.tan(angle_x)**2 + math.tan(angle_y)**2)
        # while distance is trusted, learn the QR's real size for later
        if size_est is not None and _learning_ok(detection, img_width, img_height):
            size_est.update(size_px, distance, focal_px)
    else:
        # low altitude (or no alt): pinhole with the learned size
        qr_size = size_est.get() if size_est is not None else None
        if qr_size is None:
            qr_size = QR_SIZE_METERS
        distance = (qr_size * focal_px) / size_px

    return angle_x, angle_y, distance, size_x, size_y


def send_landing_target(master, angle_x, angle_y, distance, size_x, size_y,
                        time_usec=None):
    # time_usec should be the frame CAPTURE time, not send time --
    # ArduPilot rewinds by PLND_LAG to line the measurement up with
    # the attitude the vehicle had when the frame was taken.
    if time_usec is None:
        time_usec = int(time.time() * 1e6)
    try:
        master.mav.landing_target_send(
            int(time_usec),
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
    


# for altitude

def request_altitude_stream(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        200000, 0, 0, 0, 0, 0)  # 5 Hz


def altitude_listener(master, alt_holder, alt_lock, running):
    while running.is_set():
        try:
            msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg:
                with alt_lock:
                    alt_holder['alt'] = msg.relative_alt / 1000.0
                    alt_holder['time'] = time.time()
        except Exception as e:
            print(f"alt listener error: {e}")
            time.sleep(0.5)
