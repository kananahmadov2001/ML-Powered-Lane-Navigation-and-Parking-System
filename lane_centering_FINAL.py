import cv2
import numpy as np
import math
import time
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685

# ------------------ Configuration ------------------
LOOKAHEAD_DISTANCE = 240              # Approx. 3 meters in pixels
STEERING_GAIN_KP = 0.5
MAX_STEERING_ANGLE_DEG = 30           # For normalization

# ------------------ Motor + Steering Setup ------------------
def initialize_pca9685():
    i2c = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c)
    pca.frequency = 100
    return pca

def set_motor_speed(pca, speed_percent):
    duty = int(((speed_percent) * 3277) + (65535 * 0.15))
    duty = max(0, min(65535, duty))
    pca.channels[15].duty_cycle = duty

def set_steering(pca, steering_percent):
    neutral = 65535 * 0.15
    delta = steering_percent * 0.025 * 65535
    duty = int(neutral + delta)
    duty = max(0, min(65535, duty))
    pca.channels[14].duty_cycle = duty

# ------------------ Lane Detection + Control ------------------
def lane_centering():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    pca = initialize_pca9685()
    set_motor_speed(pca, 0.15)  # Start moving forward slowly

    print("[INFO] Starting lane-centering without ROI... Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            height, width = edges.shape

            # Hough line detection on full image
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                    minLineLength=50, maxLineGap=20)
            left_lines, right_lines = [], []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        if slope < -0.3:
                            left_lines.append((x1, y1, x2, y2))
                        elif slope > 0.3:
                            right_lines.append((x1, y1, x2, y2))

            if left_lines and right_lines:
                # Calculate average x-positions near bottom of frame
                left_x = np.mean([min(x1, x2) for x1, _, x2, _ in left_lines])
                right_x = np.mean([max(x1, x2) for x1, _, x2, _ in right_lines])
                lane_center = (left_x + right_x) / 2
                cam_center = width / 2
                offset = cam_center - lane_center  # Positive = lane is to the left

                # Compute steering angle
                theta_rad = math.atan2(offset, LOOKAHEAD_DISTANCE)
                theta_deg = math.degrees(theta_rad)

                steering_angle = STEERING_GAIN_KP * theta_deg
                steering_percent = np.clip(steering_angle / MAX_STEERING_ANGLE_DEG, -1.0, 1.0)

                # Apply steering
                set_steering(pca, steering_percent)

                # Print status
                print(f"[INFO] Offset: {offset:.2f} px | θ: {theta_deg:.2f}° | Steering: {steering_percent:.2f}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt. Stopping...")

    # Stop on exit
    set_motor_speed(pca, 0)
    set_steering(pca, 0)
    cap.release()
    print("[INFO] Car stopped.")

# ------------------ Entry ------------------
if __name__ == "__main__":
    lane_centering()

  
