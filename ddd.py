from gpiozero import Motor
from time import sleep

# Define pins (Check your wiring!)
# Left Motor
motor_left = Motor(forward=4, backward=14)
# Right Motor
motor_right = Motor(forward=17, backward=27)

print("Testing LEFT Motor...")
motor_left.forward(1.0) # Full speed
sleep(2)
motor_left.stop()

print("Testing RIGHT Motor...")
motor_right.forward(1.0) # Full speed
sleep(2)
motor_right.stop()

print("Testing BOTH Backward...")
motor_left.backward(1.0)
motor_right.backward(1.0)
sleep(2)
motor_left.stop()
motor_right.stop()

print("Test Complete.")
