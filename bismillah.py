import time
import numpy as np
import cv2
from gpiozero import Motor
from picamera2 import Picamera2
from ultralytics import YOLO

# Hardware Pins - VERIFY THESE ON YOUR PI [cite: 389]
MOTOR_L_FWD, MOTOR_L_BWD = 23, 24 
MOTOR_R_FWD, MOTOR_R_BWD = 26, 19 

FRAME_W, FRAME_H = 640, 480
# Define the ROI to focus on the lower part [cite: 105]
ROI_TOP = int(FRAME_H * 0.60) 

# PID Constants [cite: 64]
KP_LANE = 0.005
BASE_SPEED = 0.45 # Increase this if it hums but doesn't move

motor_left = Motor(forward=MOTOR_L_FWD, backward=MOTOR_L_BWD)
motor_right = Motor(forward=MOTOR_R_FWD, backward=MOTOR_R_BWD)

def drive(v, w):
    # Differential control mixing [cite: 119, 121]
    left = np.clip(v - w, -1, 1)
    right = np.clip(v + w, -1, 1)
    
    if left >= 0: motor_left.forward(left)
    else: motor_left.backward(abs(left))
    
    if right >= 0: motor_right.forward(right)
    else: motor_right.backward(abs(right))

class Perception:
    def __init__(self):
        self.model = YOLO("yolo11n.pt") 

    def process(self, frame):
        # 1. Lane Detection [cite: 111, 112]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Mask sky/background [cite: 82]
        mask = np.zeros_like(edges)
        cv2.rectangle(mask, (0, ROI_TOP), (FRAME_W, FRAME_H), 255, -1)
        edges = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=100)
        l_edge, r_edge = 100, FRAME_W - 100 # Defaults
        
        lane_type = "Unknown"
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2-y1)/(x2-x1+1e-5)
                if slope < -0.5: l_edge = x2; lane_type = "White Solid" [cite: 26]
                if slope > 0.5: r_edge = x1; lane_type = "Yellow Solid" [cite: 26]

        center = (l_edge + r_edge) / 2
        
        # 2. MIO Detection (Closest in ego-lane) [cite: 59, 116]
        results = self.model.predict(frame, conf=0.4, verbose=False)
        mio = None
        max_y = -1
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            # Check if obstacle is in-lane 
            if l_edge - 20 < cx < r_edge + 20:
                if y2 > max_y:
                    max_y = y2
                    mio = (int(x1), int(y1), int(x2), int(y2))
                    
        return center, mio, max_y, lane_type

def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    sense = Perception()
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            center, mio, mio_y, l_type = sense.process(frame_bgr)
            
            # PID 1: Steering [cite: 41]
            error = (FRAME_W / 2) - center
            steer = error * KP_LANE
            
            # PID 2: Distance/Stopping [cite: 42, 137]
            speed = BASE_SPEED
            if mio is not None:
                # Stop if close [cite: 21]
                if mio_y > (FRAME_H * 0.80):
                    speed = 0
                    print("!!! STOPPING FOR MIO !!!")
            
            drive(speed, steer)
            
            # Display Requirements [cite: 24, 25, 26]
            cv2.putText(frame_bgr, f"Lane: {l_type}", (10, 30), 0, 0.7, (255, 255, 255), 2)
            if mio:
                # Highlight MIO in Yellow [cite: 24]
                cv2.rectangle(frame_bgr, (mio[0], mio[1]), (mio[2], mio[3]), (0, 255, 255), 3)
            
            cv2.imshow("MIO Algorithm - Option 2", frame_bgr)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        drive(0, 0)
        picam2.stop()

if __name__ == "__main__":
    main()

