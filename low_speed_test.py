#Test ESC / Motor at a very low speeimport adafruit_motor.servo
import math
import time
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685

def initialize_pca():
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 100  # Match ESC's expected PWM frequency
    return pca

def set_motor_speed(pca, percent):
    # percent should be between -1.0 and 1.0
    duty = ((percent) * 3277) + 65535 * 0.15
    duty = max(0, min(65535, duty))  # Clamp to valid range
    pca.channels[15].duty_cycle = math.floor(duty)
    print(f"Speed set to {percent:.2f}, duty cycle = {duty:.0f}")

def main():
    pca = initialize_pca()

    print("Running motor forward at low speed...")
    set_motor_speed(pca, 0.147)
    time.sleep(10)

    print("Stopping motor.")
    set_motor_speed(pca, 0)
    time.sleep(3)

if __name__ == "__main__":
    main()
