import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# Project Constraints
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_TOP = int(FRAME_HEIGHT * 0.55) # focus on the lower part [cite: 82, 105]
LANE_WIDTH_ASSUMPTION = 350 # pixels; adjust this based on your track width

class LaneAssistPerception:
    def __init__(self):
        # Use YOLO to identify objects (MIO algorithm) [cite: 86, 109, 142]
        self.model = YOLO("yolo11n.pt") #CHNGE TO WHERE YOLO IS

    def get_perception_data(self, frame_bgr):
        # 1. Pre-processing [cite: 75, 98]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150) # Edge detection [cite: 80, 103]
        
        # 2. ROI Masking [cite: 82, 105]
        mask = np.zeros_like(edges)
        cv2.rectangle(mask, (0, ROI_TOP), (FRAME_WIDTH, FRAME_HEIGHT), 255, -1)
        edges = cv2.bitwise_and(edges, mask)
        
        # 3. Line Detection (Hough) [cite: 89, 112]
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=120)
        
        l_bound, r_bound = None, None
        lane_type = "Searching..."

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2-y1)/(x2-x1+1e-6)
                if slope < -0.4: l_bound = x2; lane_type = "White/Dashed" # Left [cite: 17, 26]
                elif slope > 0.4: r_bound = x1; lane_type = "Yellow/Solid" # Right [cite: 17, 26]

        # --- PATH CALCULATION LOGIC ---
        # This is what makes it move even if lines flicker
        if l_bound and r_bound:
            center = (l_bound + r_bound) / 2
        elif l_bound:
            center = l_bound + (LANE_WIDTH_ASSUMPTION / 2) # Estimate path from left
        elif r_bound:
            center = r_bound - (LANE_WIDTH_ASSUMPTION / 2) # Estimate path from right
        else:
            center = FRAME_WIDTH / 2 # If totally lost, go straight
            lane_type = "Lost - Straight"

        # 4. MIO Identification [cite: 40, 51, 116]
        results = self.model.predict(frame_bgr, conf=0.4, verbose=False)
        mio, max_y = None, -1
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            # Obstacle Identification: Only closest in-lane [cite: 13, 18, 135]
            if (center - 150) < cx < (center + 150):
                if y2 > max_y:
                    max_y = y2
                    mio = (int(x1), int(y1), int(x2), int(y2))
                    
        return center, mio, max_y, lane_type