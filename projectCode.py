import time
import cv2
import numpy as np

# ============================
# Project 2 - Option 2 (Differential)
# Lane centering + YOLO obstacle detection
# Most Important Object (MIO) selection (camera-only)
# ============================

# ----------------------------
# USER SETTINGS (TUNE THESE)
# ----------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# Differential drive base speed (0..1)
BASE_SPEED = 0.45
MAX_TURN = 0.35

# PID1 (lane centering) gains
KP_LANE = 0.0025
KD_LANE = 0.0010

# Camera-only "distance" proxy using bbox bottom y (pixels)
# Tune these based on your camera height/tilt:
SLOW_BOTTOMY = 330     # start slowing when MIO bottom_y >= this
STOP_BOTTOMY = 430     # stop when MIO bottom_y >= this

# YOLO model weights and confidence
YOLO_MODEL_PATH = "yolo11n.pt"   # change if using yolov8n.pt or best.pt
YOLO_CONF = 0.35

# If you want to restrict obstacle classes, put them here (or leave None)
ALLOWED_CLASSES = None  # e.g. {"person","chair","bottle","backpack"}


# ----------------------------
# Motor control (gpiozero)
# ----------------------------
USE_MOTORS = True
try:
    from gpiozero import Motor
    # CHANGE these pins to match your wiring (BCM numbering)
    LEFT_MOTOR = Motor(forward=17, backward=18)
    RIGHT_MOTOR = Motor(forward=22, backward=23)
except Exception as e:
    print(f"[WARN] Motor control disabled: {e}")
    USE_MOTORS = False
    LEFT_MOTOR = None
    RIGHT_MOTOR = None

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def set_diff_drive(left_speed, right_speed):
    """left_speed/right_speed are -1..1"""
    left_speed = clamp(left_speed, -1.0, 1.0)
    right_speed = clamp(right_speed, -1.0, 1.0)

    if not USE_MOTORS:
        print(f"[MOTORS OFF] L={left_speed:.2f} R={right_speed:.2f}")
        return

    if left_speed >= 0:
        LEFT_MOTOR.forward(left_speed)
    else:
        LEFT_MOTOR.backward(-left_speed)

    if right_speed >= 0:
        RIGHT_MOTOR.forward(right_speed)
    else:
        RIGHT_MOTOR.backward(-right_speed)

def stop_robot():
    set_diff_drive(0.0, 0.0)


# ----------------------------
# YOLO (Ultralytics)
# ----------------------------
from ultralytics import YOLO
yolo_model = YOLO(YOLO_MODEL_PATH)


# ----------------------------
# MIO Geometry (same concept as your PDF)
# ----------------------------
class Point:
    def __init__(self, x=0, y=0):
        self.x = float(x)
        self.y = float(y)

class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        self.x1, self.y1, self.x2, self.y2 = float(min_x), float(min_y), float(max_x), float(max_y)

    def corners(self):
        return [Point(self.x1, self.y1), Point(self.x2, self.y1),
                Point(self.x1, self.y2), Point(self.x2, self.y2)]

    def bottom_y(self):
        return self.y2

    def center(self):
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

class Line:
    """y = m x + b and x_at_y(y)"""
    def __init__(self, p1, p2):
        dx = (p2.x - p1.x)
        if abs(dx) < 1e-6:
            dx = 1e-6
        self.m = (p2.y - p1.y) / dx
        self.b = p1.y - self.m * p1.x

    def x_at_y(self, y):
        if abs(self.m) < 1e-6:
            return float("inf")
        return (y - self.b) / self.m

def most_important_object(detections, line_left, line_right):
    """
    In-lane test: ANY rectangle corner must lie between lane lines (x_left(y) <= x <= x_right(y)).
    MIO = in-lane detection with greatest bottom_y (closest in image).
    Returns index or -1.
    """
    in_lane = []
    for i, rect in enumerate(detections):
        for c in rect.corners():
            x_left = line_left.x_at_y(c.y)
            x_right = line_right.x_at_y(c.y)
            if x_left <= c.x <= x_right:
                in_lane.append(i)
                break

    if not in_lane:
        return -1

    best = in_lane[0]
    best_by = detections[best].bottom_y()
    for i in in_lane[1:]:
        by = detections[i].bottom_y()
        if by > best_by:
            best_by = by
            best = i
    return best


# ----------------------------
# Lane detection (ROI + edges + Hough + fit)
# ----------------------------
def roi_mask(edges):
    h, w = edges.shape[:2]
    mask = np.zeros_like(edges)

    # Trapezoid ROI - tune for your camera view
    poly = np.array([[
        (int(0.05 * w), h),
        (int(0.42 * w), int(0.62 * h)),
        (int(0.58 * w), int(0.62 * h)),
        (int(0.95 * w), h),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(edges, mask)

def fit_lane_lines(frame, lines):
    """
    Returns left_pts, right_pts each as (bottom_point, top_point) or None.
    """
    if lines is None:
        return None, None

    h, w = frame.shape[:2]
    left_fit = []
    right_fit = []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # reject near-horizontal
        if abs(slope) < 0.4:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    def make_points(fit):
        if len(fit) == 0:
            return None
        m, b = np.mean(fit, axis=0)
        y1 = h
        y2 = int(h * 0.62)
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        return (Point(x1, y1), Point(x2, y2))

    return make_points(left_fit), make_points(right_fit)

def lane_center_x(left_pts, right_pts):
    if left_pts is None or right_pts is None:
        return None
    return int((left_pts[0].x + right_pts[0].x) / 2)

def estimate_lane_type(frame, left_pts, right_pts):
    """
    Basic (non-ML) lane color classification for display requirement.
    """
    if left_pts is None or right_pts is None:
        return "Unknown"

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def sample(p):
        x = int(clamp(p.x, 2, frame.shape[1] - 3))
        y = int(clamp(p.y, 2, frame.shape[0] - 3))
        patch = hsv[y-2:y+3, x-2:x+3]
        return np.mean(patch.reshape(-1, 3), axis=0)

    lp_mid = Point((left_pts[0].x + left_pts[1].x) / 2.0, (left_pts[0].y + left_pts[1].y) / 2.0)
    rp_mid = Point((right_pts[0].x + right_pts[1].x) / 2.0, (right_pts[0].y + right_pts[1].y) / 2.0)

    lH, lS, lV = sample(lp_mid)
    rH, rS, rV = sample(rp_mid)

    def classify(H, S, V):
        is_white = (S < 40 and V > 160)
        is_yellow = (15 <= H <= 45 and S > 60 and V > 80)
        if is_yellow:
            return "Yellow"
        if is_white:
            return "White"
        return "Unknown"

    return f"L:{classify(lH,lS,lV)} R:{classify(rH,rS,rV)}"


# ----------------------------
# Main loop
# ----------------------------
def main():
    last_err = 0.0
    last_t = time.time()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print("[INFO] Running Project 2 Option 2 (Differential) - MIO - NO ultrasonic.")
    print("[INFO] Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            img_center = w // 2

            # ---- Lane detection ----
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 60, 160)
            masked = roi_mask(edges)

            lines = cv2.HoughLinesP(masked, 2, np.pi/180,
                                    threshold=50, minLineLength=35, maxLineGap=120)

            left_pts, right_pts = fit_lane_lines(frame, lines)

            # Fallback if lane not detected
            if left_pts is None or right_pts is None:
                set_diff_drive(0.20, 0.20)
                continue

            line_left = Line(left_pts[0], left_pts[1])
            line_right = Line(right_pts[0], right_pts[1])

            lc = lane_center_x(left_pts, right_pts)
            if lc is None:
                set_diff_drive(0.20, 0.20)
                continue

            # ---- PID1: lane centering ----
            err = float(lc - img_center)
            now = time.time()
            dt = max(1e-3, now - last_t)
            derr = (err - last_err) / dt
            turn_lane = KP_LANE * err + KD_LANE * derr
            turn_lane = clamp(turn_lane, -MAX_TURN, MAX_TURN)
            last_err = err
            last_t = now

            # ---- YOLO detections ----
            results = yolo_model.predict(frame, conf=YOLO_CONF, verbose=False)
            r = results[0]
            names = r.names

            rects, metas = [], []  # Rectangle + (label, conf)
            if r.boxes is not None:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    label = names.get(cls_id, str(cls_id))
                    conf = float(b.conf[0].item())

                    if ALLOWED_CLASSES is not None and label not in ALLOWED_CLASSES:
                        continue

                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    rects.append(Rectangle(x1, y1, x2, y2))
                    metas.append((label, conf))

            # ---- MIO ----
            mio_idx = most_important_object(rects, line_left, line_right)

            # ---- Obstacle behavior (camera-only distance proxy) ----
            speed_scale = 1.0
            turn_obj = 0.0
            stopped = False

            if mio_idx != -1:
                mio = rects[mio_idx]
                by = mio.bottom_y()

                if by >= STOP_BOTTOMY:
                    speed_scale = 0.0
                    stopped = True
                elif by >= SLOW_BOTTOMY:
                    speed_scale = clamp((STOP_BOTTOMY - by) / (STOP_BOTTOMY - SLOW_BOTTOMY), 0.2, 1.0)

                # mild centering on MIO when close (optional)
                cx = mio.center().x
                obj_err = float(cx - img_center)
                proximity = clamp((by - SLOW_BOTTOMY) / max(1.0, (STOP_BOTTOMY - SLOW_BOTTOMY)), 0.0, 1.0)
                turn_obj = clamp(0.0015 * obj_err * proximity, -MAX_TURN, MAX_TURN)

            # ---- Combine lane + obstacle control ----
            turn = clamp(turn_lane + turn_obj, -MAX_TURN, MAX_TURN)
            fwd = BASE_SPEED * speed_scale

            left_cmd = clamp(fwd - turn, -1.0, 1.0)
            right_cmd = clamp(fwd + turn, -1.0, 1.0)
            set_diff_drive(left_cmd, right_cmd)

            # ---- Visualization (project requirement) ----
            vis = frame.copy()

            # Lane lines: white
            cv2.line(vis, (int(left_pts[0].x), int(left_pts[0].y)),
                     (int(left_pts[1].x), int(left_pts[1].y)), (255, 255, 255), 3)
            cv2.line(vis, (int(right_pts[0].x), int(right_pts[0].y)),
                     (int(right_pts[1].x), int(right_pts[1].y)), (255, 255, 255), 3)

            # Obstacles: all white; MIO yellow
            for i, rect in enumerate(rects):
                x1, y1, x2, y2 = int(rect.x1), int(rect.y1), int(rect.x2), int(rect.y2)
                color = (255, 255, 255)
                if i == mio_idx:
                    color = (0, 255, 255)  # yellow
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                label, conf = metas[i]
                cv2.putText(vis, f"{label} {conf:.2f}", (x1, max(0, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            lane_type = estimate_lane_type(frame, left_pts, right_pts)
            cv2.putText(vis, f"LaneType: {lane_type}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if mio_idx != -1:
                cv2.putText(vis, "MIO: ACTIVE", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if stopped:
                cv2.putText(vis, "STOPPED at MIO", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("AIRE475 Project 2 - Option 2 (No Ultrasonic)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    finally:
        stop_robot()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
