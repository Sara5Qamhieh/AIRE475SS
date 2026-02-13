import time
from gpiozero import PWMOutputDevice

def init_motors():
    # Motor A (Left) [cite: 389]
    in1 = PWMOutputDevice(27) # Backward
    in2 = PWMOutputDevice(4)  # Forward
    # Motor B (Right) [cite: 389]
    in3 = PWMOutputDevice(17) # Forward
    in4 = PWMOutputDevice(22) # Backward
    return {"in1": in1, "in2": in2, "in3": in3, "in4": in4}

def set_differential_drive(motors, linear_v, angular_w):
    # Combine outputs based on obstacle proximity & lane alignment [cite: 121]
    left_speed = linear_v - angular_w
    right_speed = linear_v + angular_w
    
    # Left Motor Control
    motors["in2"].value = max(0, min(left_speed, 1.0)) if left_speed > 0 else 0
    motors["in1"].value = max(0, min(abs(left_speed), 1.0)) if left_speed < 0 else 0
    
    # Right Motor Control
    motors["in3"].value = max(0, min(right_speed, 1.0)) if right_speed > 0 else 0
    motors["in4"].value = max(0, min(abs(right_speed), 1.0)) if right_speed < 0 else 0

def calculate_pids(center_error, mio_y, state):
    dt = time.time() - state['last_time']
    
    # PID 1: Lane tracking (centering) [cite: 41, 64, 118]
    steer_angle = center_error * state['kp_lane']
    
    # PID 2: Distance/MIO control [cite: 42, 64, 119]
    speed = state['base_speed']
    if mio_y > 400: # Close obstacle -> Stop [cite: 14, 21, 135]
        speed = 0
    elif mio_y > 200: # Approach -> Slow [cite: 119]
        speed *= 0.5
        
    state['last_time'] = time.time()
    return speed, steer_angle