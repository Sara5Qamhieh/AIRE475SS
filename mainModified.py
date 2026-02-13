import cv2, time
from Lane_Detection import LaneAssistPerception, Picamera2
from PID_controller import init_motors, set_differential_drive, calculate_pids

def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    motors = init_motors()
    perception = LaneAssistPerception()
    
    # Constants: 'base_speed' must be high enough to overcome friction!
    state = {'kp_lane': 0.008, 'base_speed': 0.55, 'last_time': time.time()}

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Perception: Get the "Path" center and MIO data [cite: 116, 117]
            center, mio, mio_y, l_type = perception.get_perception_data(frame_bgr)
            
            # Control: Calculate Error relative to image center
            error = (640 / 2) - center 
            v, w = calculate_pids(error, mio_y, state)
            
            # Action: Move motors using Differential Control [cite: 121, 142]
            set_differential_drive(motors, v, w)
            
            # Visual Requirements [cite: 23, 24, 25, 26]
            cv2.putText(frame_bgr, f"Type: {l_type}", (10, 30), 0, 0.7, (255, 255, 255), 2)
            if mio:
                # Highlight the MIO in yellow 
                cv2.rectangle(frame_bgr, (mio[0], mio[1]), (mio[2], mio[3]), (0, 255, 255), 3)
            
            cv2.imshow("AIRE475 Option 2", frame_bgr)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        set_differential_drive(motors, 0, 0) # stop at end [cite: 14]
        picam2.stop()

if __name__ == "__main__":
    main()