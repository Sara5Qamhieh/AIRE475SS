import time
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# ==========================================
# HARDWARE INITIALIZATION
# ==========================================
def init_motors():
    """
    Initializes GPIO pins for the L298N H-Bridge.
    Returns a dictionary containing the motor objects.
    """
    # Motor A (Left)
   # ena = PWMOutputDevice()
    in2 = DigitalOutputDevice(23)  # Forward
    in1 = DigitalOutputDevice(24) # Backward
    
    # Motor B (Right)
    #enb = PWMOutputDevice(13)
    in3 = DigitalOutputDevice(26) # Forward
    in4 = DigitalOutputDevice(19) # Backward
    
    # Store in a standard dictionary
    motors = {
      
        "in1": in1,
        "in2": in2,
        "in3": in3,
        "in4": in4
    }
    return motors

def stop_motors(motors):
    """Stops all motors immediately."""
   # motors["ena"].off()
   # motors["enb"].off()
    motors["in1"].off()
    motors["in2"].off()
    motors["in3"].off()
    motors["in4"].off()

def set_motor_speed(motors, motor_side, speed):
    """
    Sets speed for a specific motor side.
    Args:
        motors (dict): The dictionary returned by init_motors()
        motor_side (str): 'left' or 'right'
        speed (float): -1.0 to 1.0
    """
    # Clamp speed between -1 and 1
    speed = max(min(speed, 1.0), -1.0)
    abs_speed = abs(speed)

    # Select the correct pins based on side
    if motor_side == 'left':
        pwm = motors["ena"]
        fwd = motors["in2"]
        bwd = motors["in1"]
    else:
        pwm = motors["enb"]
        fwd = motors["in3"]
        bwd = motors["in4"]
    
    # Apply PWM Speed
    pwm.value = abs_speed
    
    # Apply Direction
    if speed > 0:
        fwd.on()
        bwd.off()
    elif speed < 0:
        fwd.off()
        bwd.on()
    else:
        fwd.off()
        bwd.off()

# ==========================================
# PID LOGIC FUNCTIONS
# ==========================================
def calculate_error(sensor_array):
    """
    Calculates error from sensor array.
    Weights: -4 -3 -2 -1 | +1 +2 +3 +4
    """
    weights = [-4, -3, -2, -1, 1, 2, 3, 4]
    active_sensors = 0
    weighted_sum = 0
    
    for i, reading in enumerate(sensor_array):
        if reading == 1:
            weighted_sum += weights[i]
            active_sensors += 1
            
    if active_sensors == 0:
        return 0
        
    # Return average position of the line
    return weighted_sum / active_sensors

def execute_pid_step(sensor_array, motors, state):
    """
    Calculates PID and updates motor speeds.
    Args:
        sensor_array: List [0,1,0...]
        motors: Dict of motor pins
        state: Dict containing PID constants and history ('kp', 'prev_error', etc.)
    """
    current_time = time.time()
    dt = current_time - state['last_time']
    
    # 1. Calculate Error
    error = calculate_error(sensor_array)
    
    # 2. PID Math
    P = error
    state['integral'] += error * dt
    D = (error - state['prev_error']) / dt if dt > 0 else 0
    
    # Calculate Correction
    correction = (state['kp'] * P) + (state['ki'] * state['integral']) + (state['kd'] * D)
    
    # 3. Apply to Motors
    # If Error > 0 (Line on Right), we need to turn Left.
    # Left Motor Decreases, Right Motor Increases.
    base_speed = state['base_speed']
    left_speed = base_speed - correction
    right_speed = base_speed + correction
    
    # Apply values using the helper function
    set_motor_speed(motors, 'left', left_speed)
    set_motor_speed(motors, 'right', right_speed)
    
    # 4. Update State History for next loop
    state['prev_error'] = error
    state['last_time'] = current_time
    
    return error, correction
