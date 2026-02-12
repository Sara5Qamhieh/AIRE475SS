import time
import numpy as np
import cv2
import os
from dataclasses import dataclass
from ultralytics import YOLO
from gpiozero import Motor

# =========================================================
# 0) PARAMETERS
# =========================================================

# --- Hardware Pins (Your confirmed pins) ---
MOTOR_L_FWD = 23
MOTOR_L_BWD = 24
MOTOR_R_FWD = 26
MOTOR_R_BWD = 19

# --- Camera Settings ---
FRAME_W, FRAME_H = 640, 480
ROI_TOP = int(FRAME_H * 0.55)

# --- HSV Thresholds ---
YELLOW_LO = (15, 80, 80)
YELLOW_HI = (40, 255, 255)
WHITE_LO = (0, 0, 180)    
WHITE_HI = (179, 60, 255) 

# --- Edge/Hough ---
CANNY_LO, CANNY_HI = 30, 100
HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH = 1, np.pi/180, 10
HOUGH_MIN_LINE, HOUGH_MAX_GAP = 10, 30

# --- Control Logic ---
LANE_WIDTH_PX = 260
KP_LANE, KD_LANE = 0.0040, 0.0015
MAX_TURN = 0.50 

KP_AVOID, KD_AVOID = 0.0060, 0.0020
MAX_AVOID = 0.50

# --- Speed Settings ---
BASE_SPEED = 0.55   
MIN_SPEED = 0.40    
CURVE_SLOW_K = 0.001

# --- MIO Logic ---
SLOW_BOTTOMY = int(FRAME_H * 0.70)
STOP_BOTTOMY = int(FRAME_H * 0.88)
OBSTACLE_CLASSES = {"person", "bottle", "chair", "stop sign", "traffic cone", "cup"}

# --- Kalman ---
IOU_MATCH_THRESH = 0.25
MAX_TRACK_AGE = 5
KALMAN_Q = 1e-2
KALMAN_R = 2e-1


# =========================================================
# 1) HARDWARE & HELPER CLASSES
# =========================================================
motor_left = Motor(forward=MOTOR_L_FWD, backward=MOTOR_L_BWD)
motor_right = Motor(forward=MOTOR_R_FWD, backward=MOTOR_R_BWD)

def set_motor_speeds(left: float, right: float):
    left = max(min(left, 1.0), -1.0)
    right = max(min(right, 1.0), -1.0)
    if left >= 0: motor_left.forward(left)
    else: motor_left.backward(-left)
    if right >= 0: motor_right.forward(right)
    else: motor_right.backward(-right)

class PD:
    def __init__(self, kp, kd, out_clip): 
        self.kp, self.kd, self.out_clip = kp, kd, out_clip
        self.last_err, self.last_t = 0.0, time.time()
    def step(self, err):
        t = time.time(); dt = max(t - self.last_t, 1e-3)
        out = float(np.clip(self.kp * err + self.kd * (err - self.last_err)/dt, -self.out_clip, self.out_clip))
        self.last_err, self.last_t = err, t
        return out

class Kalman2D:
    def __init__(self, x, y): 
        self.x, self.P = np.array([x, y, 0, 0], dtype=np.float32), np.eye(4, dtype=np.float32)*10
    def predict(self, dt): 
        F=np.eye(4, dtype=np.float32); F[0,2]=dt; F[1,3]=dt
        self.x=F@self.x; self.P=F@self.P@F.T + np.eye(4, dtype=np.float32)*KALMAN_Q
    def update(self, mx, my): 
        H=np.zeros((2,4), dtype=np.float32); H[0,0]=1; H[1,1]=1
        y=np.array([mx, my], dtype=np.float32)-(H@self.x); S=H@self.P@H.T + np.eye(2, dtype=np.float32)*KALMAN_R
        K=self.P@H.T@np.linalg.inv(S); self.x+=K@y; self.P=(np.eye(4)-K@H)@self.P
    @property
    def pos(self): return float(self.x[0]), float(self.x[1])

@dataclass
class Track: tid: int; kf: Kalman2D; bbox: tuple; cls_name: str; age: int=0

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area = (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (area - inter + 1e-6) if inter > 0 else 0


# =========================================================
# 2) VISION PIPELINE
# =========================================================
def process_lanes(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask_y = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    mask_w = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    
    mask_y[:ROI_TOP, :] = 0
    mask_w[:ROI_TOP, :] = 0
    
    lines_y = cv2.HoughLinesP(cv2.Canny(mask_y, CANNY_LO, CANNY_HI), HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH, HOUGH_MIN_LINE, HOUGH_MAX_GAP)
    lines_w = cv2.HoughLinesP(cv2.Canny(mask_w, CANNY_LO, CANNY_HI), HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH, HOUGH_MIN_LINE, HOUGH_MAX_GAP)
    return lines_y, lines_w

def fit_lines(lines):
    if lines is None: return []
    pts = []
    for l in lines[:, 0]:
        x1, y1, x2, y2 = l
        if abs(y2 - y1) < 5: continue
        pts.append((x1, y1, x2, y2))
    
    candidates = []
    y_eval = FRAME_H - 5
    cx = FRAME_W * 0.5
    for (x1, y1, x2, y2) in pts:
        m = (x2 - x1) / (y2 - y1 + 1e-6)
        if abs(m) > 4.0: continue
        b = x1 - m * y1
        x_at_bottom = m * y_eval + b
        candidates.append((x_at_bottom, m, b))
    
    return [(c[1], c[2]) for c in candidates]

def get_right_lane_center(lines_y, lines_w, y_eval, cx):
    fits_y = fit_lines(lines_y)
    fits_w = fit_lines(lines_w)
    x_yellow = None
    x_white = None

    right_cands = [m*y_eval+b for m,b in fits_y if (m*y_eval+b) > cx - 50]
    if right_cands: x_yellow = min(right_cands, key=lambda x: abs(x - cx))

    limit = x_yellow if x_yellow else cx + 100
    left_cands = [m*y_eval+b for m,b in fits_w if (m*y_eval+b) < limit]
    if left_cands: x_white = max(left_cands)

    if x_yellow and x_white: return 0.5 * (x_yellow + x_white), x_white, x_yellow
    if x_yellow: return 0.5 * (x_yellow + (x_yellow - LANE_WIDTH_PX)), (x_yellow - LANE_WIDTH_PX), x_yellow
    if x_white: return 0.5 * (x_white + (x_white + LANE_WIDTH_PX)), x_white, (x_white + LANE_WIDTH_PX)

    return None, None, None

def get_mio_in_lane(tracks, xL, xR):
    if not tracks: return None
    if xL is None: xL = (FRAME_W/2) - (LANE_WIDTH_PX/2)
    if xR is None: xR = (FRAME_W/2) + (LANE_WIDTH_PX/2)

    candidates = []
    for tr in tracks:
        obj_x = tr.kf.pos[0]
        if (obj_x > xL - 20) and (obj_x < xR + 20):
            candidates.append(tr)
    
    if not candidates: return None
    return max(candidates, key=lambda tr: tr.bbox[3])


# =========================================================
# 4) MAIN LOOP
# =========================================================
def main():
    print("---------------------------------------")
    print("1. INITIALIZING CAMERA (OpenCV Mode)...")
    
    # *** CHANGE: USING OPENCV INSTEAD OF PICAMERA2 ***
    # This is
