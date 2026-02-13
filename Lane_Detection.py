import cv2
import numpy as np
from picamera2 import Picamera2

# ==========================================
# GLOBAL VARIABLES
# ==========================================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_HEIGHT_PERCENT = 0.55
CANNY_LOW = 50
CANNY_HIGH = 150
SENSOR_COUNT = 8             
SENSOR_THRESHOLD = 10        
# ==========================================

def init_camera():
    """Initializes and starts the camera"""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    return picam2

def step1_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def step2_reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def step3_detect_edges(image):
    return cv2.Canny(image, CANNY_LOW, CANNY_HIGH)

def step4_region_of_interest(image):
    height, width = image.shape
    polygons = np.array([
        [(0, height), (0, int(height * (1 - ROI_HEIGHT_PERCENT))), 
         (width, int(height * (1 - ROI_HEIGHT_PERCENT))), (width, height)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def detect_lane_markings_color(image_bgr):
    """
    UPDATED: Detects lane markings using 3 sample points for robustness.
    """
    height, width, _ = image_bgr.shape
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- SAMPLING STRATEGY ---
    # We define 3 sample boxes at the bottom of the image to find the "Road Color"
    # This prevents errors if one sample hits a white line.
    
    box_size = 40
    y_start = height - 100
    y_end = height - 20 # Don't go to the very bottom (bumper)
    
    # X coordinates for Left, Center, Right samples
    # We keep them somewhat central to avoid sampling grass/off-road
    x_left = int(width * 0.3)
    x_center = int(width * 0.5)
    x_right = int(width * 0.7)
    
    # Extract ROI for the 3 samples
    roi_left = gray[y_start:y_end, x_left-box_size:x_left+box_size]
    roi_center = gray[y_start:y_end, x_center-box_size:x_center+box_size]
    roi_right = gray[y_start:y_end, x_right-box_size:x_right+box_size]
    
    # Calculate Mean Brightness for each
    # Handle edge cases where ROI might be empty
    val_l = np.mean(roi_left) if roi_left.size > 0 else 128
    val_c = np.mean(roi_center) if roi_center.size > 0 else 128
    val_r = np.mean(roi_right) if roi_right.size > 0 else 128
    
    # --- MEDIAN FILTERING ---
    # If road is Black (50) and one sample hits a White Line (220):
    # Mean would be (50+50+220)/3 = 106 (Might think road is light!)
    # Median would be 50 (Correctly thinks road is dark)
    road_brightness = np.median([val_l, val_c, val_r])
    
    # --- THRESHOLD LOGIC ---
    mask = np.zeros_like(gray)
    contrast_offset = 50 
    
    if road_brightness < 100:
        # DARK ROAD -> Look for LIGHT markings
        lower_bound = min(255, int(road_brightness + contrast_offset))
        _, mask = cv2.threshold(gray, lower_bound, 255, cv2.THRESH_BINARY)
    else:
        # LIGHT ROAD -> Look for DARK markings
        upper_bound = max(0, int(road_brightness - contrast_offset))
        _, mask = cv2.threshold(gray, upper_bound, 255, cv2.THRESH_BINARY_INV)
        
    # Clean up noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Return mask AND sample coords for visualization
    sample_coords = [(x_left, y_start), (x_center, y_start), (x_right, y_start)]
    return mask, sample_coords

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None
    y1 = image.shape[0]
    y2 = int(y1 * (1 - ROI_HEIGHT_PERCENT)) + 20
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < -0.5:
            left_fit.append((slope, intercept))
        elif slope > 0.5:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return left_line, right_line

def get_virtual_sensors(binary_image):
    sensor_readings = []
    height, width = binary_image.shape
    
    # Scan the bottom 100 pixels
    scan_area = binary_image[height-100:height, 0:width]
    section_width = width // SENSOR_COUNT
    
    for i in range(SENSOR_COUNT):
        start_x = i * section_width
        end_x = (i + 1) * section_width
        sensor_zone = scan_area[:, start_x:end_x]
        
        white_pixels = cv2.countNonZero(sensor_zone)
        
        if white_pixels > SENSOR_THRESHOLD:
            sensor_readings.append(1)
        else:
            sensor_readings.append(0)
            
    return sensor_readings

def get_lane_curve(picam2, display=False):
    # 0. Capture
    img = picam2.capture_array()
    img = cv2.flip(img, -1) 
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Pipeline
    gray = step1_grayscale(img_bgr)
    blur = step2_reduce_noise(gray)
    edges = step3_detect_edges(blur)
    
    # 2. Color Detection (Returns Mask AND Coords now)
    color_markings, sample_coords = detect_lane_markings_color(img_bgr)
    
    # 3. Combine
    combined_features = cv2.bitwise_or(edges, color_markings)
    roi_combined = step4_region_of_interest(combined_features)
    
    # 4. Sensors
    sensor_array = get_virtual_sensors(roi_combined)

    # 5. Hough Lines
    roi_edges = step4_region_of_interest(edges) 
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
    left_line, right_line = average_slope_intercept(img_bgr, lines)

    # 6. Visuals
    if display:
        line_image = np.zeros_like(img_bgr)
        if left_line is not None:
            cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 10)
        if right_line is not None:
            cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 10)
        
        combo_image = cv2.addWeighted(img_bgr, 0.8, line_image, 1, 1)
        
        # DRAW SAMPLE BOXES (Green) - Visualizing where the robot checks color
        for (sx, sy) in sample_coords:
            cv2.rectangle(combo_image, (sx-40, sy), (sx+40, sy+80), (0, 255, 0), 2)
    else:
        combo_image = None # Optimization if display is False

    return sensor_array, combo_image
