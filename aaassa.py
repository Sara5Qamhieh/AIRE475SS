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

# --- Hardware Pins ---
MOTOR_L_FWD = 23
MOTOR_L_BWD = 24
MOTOR_R_FWD = 26
MOTOR_R_BWD = 19

# --- Camera ---
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
MAX_TURN = 0.60       # Increased for sharper curves

# PID Gains
KP_LANE = 0.0035      # Main steering (Reacts to current error)
KD_LANE = 0.0015      # Damping (Prevents wobbling)
K_CURVE = 0.0025      # NEW: Lookahead Gain (Follows the shape)

KP_AVOID, KD_AVOID = 0.0060, 0.0020
MAX_AVOID = 0.50

# --- Speed Settings ---
BASE_SPEED = 0.50
MIN_SPEED = 0.35    
CURVE_SLOW_DOWN = 0.15 # How much to slow down during a turn

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
    def pos(self): return float(self.x[0]), float(self.x
