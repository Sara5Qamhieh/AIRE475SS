import cv2
import numpy as np
import matplotlib.pyplot as plt

def simple_lane_detection_v2(image):
    # Initialization
    left_lane_boundary = []
    right_lane_boundary = []
    
    # Pre-processing
    # Convert RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge Enhancement (Morphological Gradient)
    # Equivalent to Max - Min (ordfilt2)
    kernel = np.ones((2, 2), np.uint8)
    min_img = cv2.erode(gray, kernel)
    max_img = cv2.dilate(gray, kernel)
    edge = cv2.subtract(max_img, min_img)
    
    # Thresholding
    _, bw = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Region of Interest (ROI)
    height, width = bw.shape
    bw[0:int(height/2), :] = 0  # Ignore upper half (sky, buildings)
    
    # Hough Transform
    # OpenCV's HoughLinesP is more efficient for detecting segments
    # Parameters tuned to match MATLAB's FillGap (200) and MinLength (150)
    lines = cv2.HoughLinesP(bw, rho=1, theta=np.pi/180, threshold=1, 
                            minLineLength=150, maxLineGap=200)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle in degrees to match MATLAB's theta
            # Note: OpenCV theta differs slightly in coordinate system
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Map Python angles to MATLAB's theta logic (~30 to 70 and -70 to -30)
            # In OpenCV, horizontal is 0. We adjust to find vertical-ish lanes.
            if 30 <= abs(angle) <= 70:
                if angle < 0: # Left Lane (Upward right-to-left)
                    plt.plot([x1, x2], [y1, y2], color='green', linewidth=2)
                    plt.text(x1, y1, f'{int(angle)}', color='red', fontsize=12)
                    left_lane_boundary.append(((x1, y1), (x2, y2)))
                else: # Right Lane (Upward left-to-right)
                    plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)
                    plt.text(x1, y1, f'{int(angle)}', color='red', fontsize=12)
                    right_lane_boundary.append(((x1, y1), (x2, y2)))

    # Lane Center Estimation
    position = None
    if len(left_lane_boundary) > 0 and len(right_lane_boundary) > 0:
        # Select first detected left and right lanes
        left = left_lane_boundary[0]
        right = right_lane_boundary[0]
        
        # Calculate Midpoints
        mid_left = np.mean(left, axis=0)
        mid_right = np.mean(right, axis=0)
        
        # Lane center point
        mid_lane = (mid_left + mid_right) / 2
        position = mid_lane
        
        # Visualization
        plt.plot(mid_lane[0], mid_lane[1], 'wx', markersize=10, markeredgewidth=3)

    plt.show()
    return position

# Usage Example:
# img = cv2.imread('road.jpg')
# pos = simple_lane_detection_v2(img)
