import time
import cv2
from Lane_Detection import init_camera, get_lane_curve
from PID_controller import init_motors, stop_motors, execute_pid_step, set_motor_speed
from TX_image import ImageSender

LAPTOP_IP = "10.22.93.18" # <--- Update this

def main():
    picam2 = init_camera()
    motors = init_motors()
    
    print(f"Connecting to {LAPTOP_IP}...")
    sender = ImageSender(LAPTOP_IP)
    sender.connect()

    # PID State
    pid_state = {'kp': 2, 'ki': 0.00, 'kd': 0.0002, 
                 'base_speed': 0.5, 'prev_error': 0, 'integral': 0, 'last_time': time.time()}

    # MODE CONTROL
    mode = "MANUAL" # Start safe!
    print("Mode: MANUAL (Press 'm' to toggle)")

    try:
        while True:
            start_time = time.time()
            
            # 1. Get Image
            sensor_array, combo_image = get_lane_curve(picam2, display=True)

            # 2. Check for Commands from Laptop
            command = sender.receive_command()
            
            if command:
                print(f"Cmd: {command}")
                if command == 'm':
                    if mode == "MANUAL":
                        mode = "AUTO"
                        # Reset PID memory when switching to Auto
                        pid_state['integral'] = 0 
                        pid_state['prev_error'] = 0
                    else:
                        mode = "MANUAL"
                        stop_motors(motors) # Safety stop
                
                # Manual Controls
                if mode == "MANUAL":
                    if command == 'w':   # Forward
                        set_motor_speed(motors, 'left', 1)
                        set_motor_speed(motors, 'right', 1)
                    elif command == 's': # Backward
                        set_motor_speed(motors, 'left', -1)
                        set_motor_speed(motors, 'right', -1)
                    elif command == 'a': # Left
                        set_motor_speed(motors, 'left', -1)
                        set_motor_speed(motors, 'right', 1)
                    elif command == 'd': # Right
                        set_motor_speed(motors, 'left', 1)
                        set_motor_speed(motors, 'right', -1)
                    elif command == ' ': # Spacebar = Stop
                        stop_motors(motors)

            # 3. Logic based on Mode
            if mode == "AUTO":
                error, correction = execute_pid_step(sensor_array, motors, pid_state)
                # Visuals
                cv2.putText(combo_image, "AUTO", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # In Manual, we stay at the last command state (or stop if you prefer deadman switch)
                cv2.putText(combo_image, "MANUAL", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 4. Send Image
            small_frame = cv2.resize(combo_image, (320, 240))
            sender.send_image(small_frame)

            # 5. Timing
            processing_time = time.time() - start_time
            sleep_time = 0.033 - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_motors(motors)
        picam2.stop()
        sender.close()

if __name__ == "__main__":
    main()
