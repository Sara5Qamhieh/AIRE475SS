import cv2
import numpy as np
import time
from picamera2 import Picamera2

class LaneFollowerRobot:
    def __init__(self, kp=0.15, ki=0.01, kd=0.05, base_speed=40):
        # 1. Camera Initialization (RPi Cam v3)
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        
        # 2. PID Control Variables
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.base_speed = base_speed
        
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        
        # 3. Constants for processing
        self.width = 640
        self.height = 480
        self.frame_center = self.width // 2

  def get_lane_offset(self, frame):
        # 1. Change conversion from HLS to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Use your tuned HSV thresholds
        # Yellow Mask
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # White Mask
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([179, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 3. Combine them
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        # ROI: Focus on the bottom half of the image
        roi_mask = np.zeros_like(combined_mask)
        polygon = np.array([[(0, self.height), (self.width, self.height), 
                             (self.width, int(self.height * 0.5)), (0, int(self.height * 0.5))]], np.int32)
        cv2.fillPoly(roi_mask, polygon, 255)
        masked_img = cv2.bitwise_and(combined_mask, roi_mask)
        
        # Hough Transform (Tuned for dashed lines with high maxLineGap)
        lines = cv2.HoughLinesP(masked_img, 1, np.pi/180, 20, 
                                minLineLength=20, maxLineGap=300)

        left_x, right_x = [], []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                
                # Filter by slope (MATLAB logic adapted: Left < -0.3, Right > 0.3)
                if 0.3 < slope < 2.0:
                    right_x.extend([x1, x2])
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3) # Blue for Right
                elif -2.0 < slope < -0.3:
                    left_x.extend([x1, x2])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green for Left

        # Calculate Lane Center
        if left_x and right_x:
            lane_center_x = int((np.mean(left_x) + np.mean(right_x)) / 2)
            # Visualize detected center
            cv2.circle(frame, (lane_center_x, int(self.height*0.7)), 10, (0, 255, 255), -1)
            return lane_center_x - self.frame_center
        
        return None # Return None if lanes are lost

    def set_motors(self, steering):
        """
        Placeholder for motor hardware commands.
        """
        left_speed = self.base_speed + steering
        right_speed = self.base_speed - steering
        
        # Keep speeds within 0-100 range
        left_speed = max(0, min(100, left_speed))
        right_speed = max(0, min(100, right_speed))
        
        # print(f"L: {left_speed:.1f} | R: {right_speed:.1f}")
        # Insert your motor driver code here (e.g., pwm.set_duty_cycle(left_speed))

    def run(self):
        print("Starting Lane Follower... Press 'q' to quit.")
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Calculate timing
                current_time = time.time()
                dt = current_time - self.last_time
                if dt <= 0: dt = 0.001 # Prevent division by zero
                
                # 1. Image Processing
                offset = self.get_lane_offset(frame)
                
                # 2. PID Computation (Internalized in main loop)
                if offset is not None:
                    # Proportional
                    P = self.kp * offset
                    # Integral
                    self.integral += offset * dt
                    I = self.ki * self.integral
                    # Derivative
                    derivative = (offset - self.last_error) / dt
                    D = self.kd * derivative
                    
                    steering_correction = P + I + D
                    self.last_error = offset
                    
                    # 3. Motor Movement
                    self.set_motors(steering_correction)
                    
                    # Visualization Text
                    cv2.putText(frame, f"Offset: {offset}", (10, 30), 1, 1, (255,255,255), 2)
                else:
                    # Stop if lanes are lost for safety
                    self.set_motors(-self.base_speed) # Slow stop or search mode
                
                self.last_time = current_time
                
                # Show Feed
                cv2.imshow('Lane Robot View', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()

# --- Execution ---
if __name__ == "__main__":
    # You can tune Kp, Ki, Kd here
    robot = LaneFollowerRobot(kp=0.12, ki=0.005, kd=0.04, base_speed=35)
    robot.run()

