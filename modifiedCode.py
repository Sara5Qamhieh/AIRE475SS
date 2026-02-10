import time
import cv2
import numpy as np
from ultralytics import YOLO
from gpiozero import Motor
from picamera2 import Picamera2

# ----------------------------
# 1. GEOMETRIC CLASSES (From your provided sample)
# ----------------------------
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.min_x, self.max_x = min(x1, x2), max(x1, x2)
        self.min_y, self.max_y = min(y1, y2), max(y1, y2)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        self.top_left = Point(self.min_x, self.min_y)

    def corners(self):
        x0, y0 = self.top_left.x, self.top_left.y
        x1, y1 = x0 + self.width, y0 + self.height
        return [Point(x0, y0), Point(x1, y0), Point(x0, y1), Point(x1, y1)]

class Line:
    def __init__(self, p1, p2):
        # Prevent division by zero for vertical lines
        if (p2.x - p1.x) == 0:
            self.m = 9999
        else:
            self.m = (p2.y - p1.y) / (p2.x - p1.x)
        self.b = p1.y - self.m * p1.x

    def x_at_y(self, y):
        return (y - self.b) / self.m

def most_important_object(detections, line_left, line_right):
    in_lane_indices = []
    for i, rect in enumerate(detections):
        corners = rect.corners()
        for c in corners:
            x_left = line_left.x_at_y(c.y)
            x_right = line_right.x_at_y(c.y)
            # If any corner is between the left and right boundaries
            if x_left <= c.x <= x_right:
                in_lane_indices.append(i)
                break
    
    if not in_lane_indices:
        return -1
    
    # Return the index of the object with the largest max_y (closest to robot)
    # This fulfills the "Closest In-Lane" requirement
    closest_idx = in_lane_indices[0]
    max_y = detections[closest_idx].max_y
    for idx in in_lane_indices:
        if detections[idx].max_y > max_y:
            max_y = detections[idx].max_y
            closest_idx = idx
    return closest_idx

# ----------------------------
# 2. SETTINGS & CALIBRATION
# ----------------------------
FRAME_W, FRAME_H = 640, 480
BASE_SPEED = 0.4
KP, KD = 0.0035, 0.0015 # PID Gains
STOP_Y = 430             # Stop when MIO bottom_y hits this

# ----------------------------
# 3. MAIN SYSTEM
# ----------------------------
motor_left = Motor(forward=4, backward=14)
motor_right = Motor(forward=17, backward=27)

def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    model = YOLO("yolo11n.pt")
    last_error = 0

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Define Lane Boundaries (This simulates the lines based on your BEV/Detection)
            # In a real test, you'd get these points from your lane detection logic
            # Here we define the left/right boundaries of the lane in front of the car
            line_L = Line(Point(100, 480), Point(250, 200)) 
            line_R = Line(Point(540, 480), Point(390, 200))

            # YOLO Detection
            results = model.predict(frame_bgr, conf=0.4, verbose=False)[0]
            rects = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                rects.append(Rectangle(x1, y1, x2, y2))

            # MIO Selection using your specific Class Logic
            mio_idx = most_important_object(rects, line_L, line_R)

            # Lane Centering (Error calculation)
            # We assume lane center is between the two lines at the bottom of the screen
            target_center = (line_L.x_at_y(480) + line_R.x_at_y(480)) / 2
            error = target_center - (FRAME_W / 2)
            turn = (error * KP) + ((error - last_error) * KD)
            last_error = error

            # Stopping Logic
            speed = BASE_SPEED
            if mio_idx != -1:
                if rects[mio_idx].max_y > STOP_Y:
                    speed = 0

            # Drive Motors
            l_val = np.clip(speed + turn, -1, 1)
            r_val = np.clip(speed - turn, -1, 1)
            motor_left.value = l_val if l_val >= 0 else 0 # Simple forward only for safety
            motor_right.value = r_val if r_val >= 0 else 0

            # Visualization (Requirement: MIO Yellow)
            for i, r in enumerate(rects):
                color = (0, 255, 255) if i == mio_idx else (255, 255, 255)
                cv2.rectangle(frame_bgr, (int(r.min_x), int(r.min_y)), 
                              (int(r.max_x), int(r.max_y)), color, 2)

            cv2.imshow("Robot AI View", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        motor_left.stop(); motor_right.stop(); picam2.stop()

if __name__ == "__main__":
    main()
