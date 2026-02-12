import os
# --- 1. MOTOR DRIVER FIX (MUST BE AT THE TOP) ---
os.environ['GPIOZERO_PIN_FACTORY'] = 'pigpio'

import time
import numpy as np
import cv2
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
MAX_TURN = 0.60       

# PID Gains
KP_LANE = 0.0035      # Main steering (Position)
KD_LANE = 0.0015      # Damping
K_CURVE = 0.0025      # Lookahead steering (Curve Following)

KP_AVOID, KD_AVOID = 0.0060, 0.0020
MAX_AVOID = 0.50

# --- Speed Settings ---
BASE_SPEED = 0.50
MIN_SPEED = 0.35    
CURVE_SLOW_DOWN = 0.15 

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

# Function A: Get Position (Bottom)
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

# Function B: Get Shape (Top vs Bottom)
def get_lane_shape(lines_y, lines_w, cx):
    fits_y = fit_lines(lines_y)
    fits_w = fit_lines(lines_w)
    y_bot = FRAME_H - 5
    y_top = ROI_TOP + 20

    # Yellow
    x_y_bot, x_y_top = None, None
    right_cands = [(m, b) for m,b in fits_y if (m*y_bot+b) > cx - 50]
    if right_cands:
        best_y = min(right_cands, key=lambda l: abs((l[0]*y_bot + l[1]) - cx))
        x_y_bot = best_y[0] * y_bot + best_y[1]
        x_y_top = best_y[0] * y_top + best_y[1]

    # White
    limit = x_y_bot if x_y_bot else cx + 100
    x_w_bot, x_w_top = None, None
    left_cands = [(m, b) for m,b in fits_w if (m*y_bot+b) < limit]
    if left_cands:
        best_w = max(left_cands, key=lambda l: (l[0]*y_bot + l[1]))
        x_w_bot = best_w[0] * y_bot + best_w[1]
        x_w_top = best_w[0] * y_top + best_w[1]

    center_bot, center_top = None, None
    
    # Bottom Center
    if x_y_bot and x_w_bot: center_bot = 0.5 * (x_y_bot + x_w_bot)
    elif x_y_bot: center_bot = x_y_bot - (LANE_WIDTH_PX * 0.5)
    elif x_w_bot: center_bot = x_w_bot + (LANE_WIDTH_PX * 0.5)

    # Top Center
    if x_y_top and x_w_top: center_top = 0.5 * (x_y_top + x_w_top)
    elif x_y_top: center_top = x_y_top - (LANE_WIDTH_PX * 0.5)
    elif x_w_top: center_top = x_w_top + (LANE_WIDTH_PX * 0.5)

    curve_val = 0
    if center_bot is not None and center_top is not None:
        curve_val = center_top - center_bot

    return center_bot, curve_val

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
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    if not cap.isOpened():
        print("❌ ERROR: Camera could not open! Run 'sudo reboot' if stuck.")
        return

    print("2. LOADING YOLO...")
    try:
        if os.path.exists("yolo11n.pt"): model = YOLO("yolo11n.pt")
        else: model = YOLO("yolo11n.pt") 
    except:
        model = YOLO("yolov8n.pt")

    lane_pd = PD(KP_LANE, KD_LANE, MAX_TURN)
    avoid_pd = PD(KP_AVOID, KD_AVOID, MAX_AVOID)

    tracks, next_id = [], 1
    last_time = time.time()

    print("---------------------------------------")
    print("🚀 ROBOT READY. DETECTING TURNS.")
    print("---------------------------------------")

    try:
        while True:
            # 1. Capture
            ret, frame_bgr = cap.read()
            if not ret: continue
            dt = max(time.time() - last_time, 1e-3)
            last_time = time.time()

            # ---------------------------------------------------------
            # 2. VISION & LOGIC
            # ---------------------------------------------------------
            lines_y, lines_w = process_lanes(frame_bgr)
            y_eval = FRAME_H - 5
            
            # A. Get Position & Boundaries (Keep this for MIO!)
            lane_center, xL, xR = get_right_lane_center(lines_y, lines_w, y_eval, FRAME_W*0.5)
            
            # B. Get Curve Shape (Add this new line!)
            _, curve_val = get_lane_shape(lines_y, lines_w, FRAME_W*0.5)

            # ---------------------------------------------------------
            # 3. OBJECT DETECTION (YOLO)
            # ---------------------------------------------------------
            results = model.predict(frame_bgr, verbose=False, conf=0.4)[0]
            dets = []
            if results.boxes:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    name = results.names[cls_id]
                    if name in OBSTACLE_CLASSES:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        dets.append((name, 0.0, (x1, y1, x2, y2)))
            
            # Tracking
            for tr in tracks: tr.kf.predict(dt); tr.age += 1
            used = set()
            for tr in tracks:
                best_i, best_j = 0, -1
                for j, d in enumerate(dets):
                    if j not in used:
                        v = iou(tr.bbox, d[2])
                        if v > best_i: best_i, best_j = v, j
                if best_j >= 0 and best_i >= IOU_MATCH_THRESH:
                    tr.kf.update(0.5*(dets[best_j][2][0]+dets[best_j][2][2]), 0.5*(dets[best_j][2][1]+dets[best_j][2][3]))
                    tr.bbox, tr.cls_name, tr.age = dets[best_j][2], dets[best_j][0], 0
                    used.add(best_j)
            for j, d in enumerate(dets):
                if j not in used:
                    tracks.append(Track(next_id, Kalman2D(0.5*(d[2][0]+d[2][2]), 0.5*(d[2][1]+d[2][3])), d[2], d[0]))
                    next_id += 1
            tracks = [tr for tr in tracks if tr.age <= MAX_TRACK_AGE]

            # Get MIO
            mio = get_mio_in_lane(tracks, xL, xR)

            # ---------------------------------------------------------
            # 4. CONTROL LOGIC (UPDATED FOR CURVES)
            # ---------------------------------------------------------
            speed, turn = BASE_SPEED, 0.0

            if lane_center is not None:
                # A. Lane Position (Where am I now?)
                err_pos = (FRAME_W/2) - lane_center
                steer_pos = lane_pd.step(err_pos)

                # B. Curve Shape (Where is the road going?)
                steer_curve = curve_val * K_CURVE

                # C. Combine: Position Correction + Lookahead Turning
                total_steer = steer_pos + steer_curve
                
                # Turn Logic for Obstacles
                steer_avoid, prox = 0.0, 0.0
                if mio:
                    mx = mio.kf.pos[0]
                    lat_dist = mx - (FRAME_W/2)
                    bottom = mio.bbox[3]
                    if bottom > SLOW_BOTTOMY:
                        prox = min(1.0, (bottom - SLOW_BOTTOMY)/(STOP_BOTTOMY - SLOW_BOTTOMY))
                    steer_avoid = -avoid_pd.step(lat_dist * prox)
                
                # Final Calculation
                turn = np.clip(total_steer + steer_avoid, -MAX_TURN, MAX_TURN)
                
                # Slow down if turning hard
                if abs(curve_val) > 50:
                    speed -= 0.15

                speed = max(speed, MIN_SPEED)
                
                if mio and mio.bbox[3] >= STOP_BOTTOMY:
                    speed = 0.0
                    print("🛑 STOP: MIO Reached")
            
            # 5. Drive
            l_val = speed - turn
            r_val = speed + turn
            set_motor_speeds(l_val, r_val)

    except KeyboardInterrupt:
        print("\n🛑 STOPPING ROBOT...")
    finally:
        motor_left.stop()
        motor_right.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ SHUTDOWN COMPLETE.")

if __name__ == "__main__":
    main()
