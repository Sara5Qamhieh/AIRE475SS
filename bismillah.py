import time
import numpy as np
import cv2
from gpiozero import Motor
from picamera2 import Picamera2
from ultralytics import YOLO

# =========================================================
# CONFIGURATION & PROJECT CONSTRAINTS
# =========================================================
# Hardware: Raspberry Pi 5 with L298N Motor Driver [cite: 171, 365]
MOTOR_L_FWD, MOTOR_L_BWD = 23, 24  # [cite: 389]
MOTOR_R_FWD, MOTOR_R_BWD = 26, 19  # [cite: 389]

# Vision Parameters
FRAME_W, FRAME_H = 640, 480
ROI_TOP = int(FRAME_H * 0.60)  # [cite: 82, 105]

# PID Constants for Differential Control [cite: 41, 42, 64]
# PID 1: Lane Tracking (Steering)
KP_LANE, KD_LANE = 0.005, 0.002 
# PID 2: MIO Distance (Linear Velocity)
KP_DIST = 0.8  

# Detection Thresholds
STOP_DIST_PX = int(FRAME_H * 0.85) # Y-coordinate threshold for "Close" obstacle [cite: 21]
OBSTACLE_CLASSES = [0, 39, 41, 67] # YOLO IDs: person, bottle, cup, stop sign [cite: 86, 109]

# =========================================================
# HARDWARE SETUP
# =========================================================
motor_left = Motor(forward=MOTOR_L_FWD, backward=MOTOR_L_BWD)
motor_right = Motor(forward=MOTOR_R_FWD, backward=MOTOR_R_BWD)

def set_differential_drive(linear_v, angular_w):
    """
    Combines outputs based on obstacle proximity & lane alignment.
    """
    left_speed = linear_v - angular_w
    right_speed = linear_v + angular_w
    
    # Clip to normalized range [-1, 1] [cite: 391]
    left_speed = np.clip(left_speed, -1, 1)
    right_speed = np.clip(right_speed, -1, 1)
    
    if left_speed >= 0: motor_left.forward(left_speed)
    else: motor_left.backward(abs(left_speed))
    
    if right_speed >= 0: motor_right.forward(right_speed)
    else: motor_right.backward(abs(right_speed))

# =========================================================
# PERCEPTION: LANE & MIO ALGORITHMS
# =========================================================
class LaneAssist:
    def __init__(self):
        self.last_error = 0
        self.last_time = time.time()
        self.model = YOLO("yolo11n.pt") # [cite: 458, 477]

    def get_lane_center(self, frame):
        """Detects lane markings and finds target center[cite: 17, 88]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) [cite: 75, 98]
        blur = cv2.GaussianBlur(gray, (5, 5), 0) [cite: 78, 101]
        edges = cv2.Canny(blur, 50, 150) [cite: 80, 103]
        
        # Masking ROI [cite: 82, 105]
        mask = np.zeros_like(edges)
        polygon = np.array([[(0, FRAME_H), (FRAME_W, FRAME_H), (FRAME_W, ROI_TOP), (0, ROI_TOP)]])
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150) [cite: 89, 112]
        
        left_x, right_x = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < -0.5: left_x.append(x2) # White line/Left
                elif slope > 0.5: right_x.append(x1) # Yellow line/Right
        
        # Default to frame center if lines missing
        l_avg = np.mean(left_x) if left_x else 100
        r_avg = np.mean(right_x) if right_x else FRAME_W - 100
        return (l_avg + r_avg) / 2, l_avg, r_avg

    def find_mio(self, frame, l_bound, r_bound):
        """Identifies the Most Important Object (closest in-lane obstacle)[cite: 40, 114, 116]."""
        results = self.model.predict(frame, conf=0.5, verbose=False) [cite: 86, 109]
        mio = None
        max_y = -1
        
        for box in results[0].boxes:
            if int(box.cls) in OBSTACLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                obj_center_x = (x1 + x2) / 2
                bottom_y = y2
                
                # Check if object is between lane boundaries [cite: 135]
                if l_bound < obj_center_x < r_bound:
                    if bottom_y > max_y:
                        max_y = bottom_y
                        mio = (int(x1), int(y1), int(x2), int(y2))
        return mio, max_y

# =========================================================
# MAIN CONTROL LOOP
# =========================================================
def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    assist = LaneAssist()
    base_speed = 0.4
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 1. Lane Tracking (PID 1) [cite: 41, 118]
            center, l_edge, r_edge = assist.get_lane_center(frame_bgr)
            error = (FRAME_W / 2) - center
            
            dt = time.time() - assist.last_time
            derivative = (error - assist.last_error) / (dt + 1e-6)
            steer_angle = (KP_LANE * error) + (KD_LANE * derivative) # [cite: 118]
            
            # 2. Obstacle Management (MIO / PID 2) [cite: 42, 119]
            mio_box, mio_y = assist.find_mio(frame_bgr, l_edge, r_edge)
            
            current_speed = base_speed
            if mio_box:
                # Stop if MIO is close [cite: 14, 21, 121]
                if mio_y > STOP_DIST_PX:
                    current_speed = 0
                else:
                    # Slow down as we approach [cite: 119]
                    current_speed *= (1 - (mio_y / FRAME_H))
            
            # 3. Execution 
            set_differential_drive(current_speed, steer_angle)
            
            # 4. Display Requirements [cite: 23, 24, 25, 26, 70]
            # Draw lane center and lines in white [cite: 25]
            cv2.line(frame_bgr, (int(l_edge), ROI_TOP), (int(l_edge), FRAME_H), (255, 255, 255), 2)
            cv2.line(frame_bgr, (int(r_edge), ROI_TOP), (int(r_edge), FRAME_H), (255, 255, 255), 2)
            
            if mio_box:
                # Highlight MIO in yellow [cite: 24, 70]
                cv2.rectangle(frame_bgr, (mio_box[0], mio_box[1]), (mio_box[2], mio_box[3]), (0, 255, 255), 3)
                cv2.putText(frame_bgr, "MIO - TARGET", (mio_box[0], mio_box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow("Option 2: Differential Lane Assist", frame_bgr)
            
            assist.last_error = error
            assist.last_time = time.time()
            
            if cv2.waitKey(1) == ord('q'): break
            
    finally:
        motor_left.stop() # [cite: 393]
        motor_right.stop() # [cite: 394]
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
