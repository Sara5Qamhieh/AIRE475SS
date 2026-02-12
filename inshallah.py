import cv2
import numpy as np
from picamera2 import Picamera2 # Specific library for RPi Cam v3

def process_frame(frame):
    """
    Core logic converted from SimpleLaneDetectionV2.m
    """
    # Pre-processing: RGB to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge Enhancement (Morphological Gradient)
    kernel = np.ones((2, 2), np.uint8)
    min_img = cv2.erode(gray, kernel)
    max_img = cv2.dilate(gray, kernel)
    edge = cv2.subtract(max_img, min_img)
    
    # Thresholding
    _, bw = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ROI: Ignore upper half
    height, width = bw.shape
    bw[0:int(height/2), :] = 0
    
    # Hough Transform
    lines = cv2.HoughLinesP(bw, 1, np.pi/180, threshold=50, 
                            minLineLength=100, maxLineGap=150)

    left_lanes = []
    right_lanes = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # MATLAB Logic: Left lane (30 to 70), Right lane (-70 to -30)
            # Adjusting signs for OpenCV coordinate system
            if 30 <= angle <= 70:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                right_lanes.append(((x1, y1), (x2, y2)))
            elif -70 <= angle <= -30:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                left_lanes.append(((x1, y1), (x2, y2)))

    # Calculate Lane Center
    if len(left_lanes) > 0 and len(right_lanes) > 0:
        mid_left = np.mean(left_lanes[0], axis=0).astype(int)
        mid_right = np.mean(right_lanes[0], axis=0).astype(int)
        lane_center = ((mid_left + mid_right) / 2).astype(int)
        
        # Draw white 'X' at center
        cv2.drawMarker(frame, tuple(lane_center), (255, 255, 255), 
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
        return lane_center
    
    return None

# --- Camera Setup ---
picam2 = Picamera2()
# Configure for standard 640x480 for faster processing
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

try:
    print("Starting Lane Detection... Press Ctrl+C to stop.")
    while True:
        # Capture frame from RPi Cam v3
        frame = picam2.capture_array()
        
        # Process the frame
        position = process_frame(frame)
        
        # Display the result
        cv2.imshow('RPi Cam v3 - Lane Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
