import cv2
import numpy as np
import math
import time
import os
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685

# -------------------- Constants --------------------
lookahead_distance = 240    # In pixels
Kp = 0.5                    # Steering gain
max_steering_angle_deg = 30

# -------------------- PCA9685 Motor + Servo Setup --------------------
def initialize_pca():
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 100
    return pca

def set_motor_speed(pca, percent):
    duty = ((percent) * 3277) + 65535 * 0.15
    duty = max(0, min(65535, duty))
    pca.channels[15].duty_cycle = math.floor(duty)

def set_steering_angle(pca, angle_percent):
    neutral_duty = 0.15 * 65535
    delta = angle_percent * 0.025 * 65535
    duty = max(0, min(65535, neutral_duty + delta))
    pca.channels[14].duty_cycle = math.floor(duty)

def save_debug_frames(frame, roi_edges, output_image):
    os.makedirs("debug", exist_ok=True)
    timestamp = int(time.time() * 1000)
    cv2.imwrite(f"debug/frame_{timestamp}.jpg", frame)
    cv2.imwrite(f"debug/edges_{timestamp}.jpg", roi_edges)
    cv2.imwrite(f"debug/lanes_{timestamp}.jpg", output_image)

# -------------------- Image Processing --------------------
def cannyEdgeDetector(image):
    return cv2.Canny(image, 50, 150)

def getROI(image):
    height, width = image.shape[:2]
    triangle = np.array([[ 
        (100, int(height * 0.75)),
        (width, int(height * 0.75)),
        (int(width / 2 + 200), int(height / 2.1))
    ]], dtype=np.int32)

    black_image = np.zeros_like(image)
    mask = cv2.fillPoly(black_image, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# -------------------- Main Lane Classifier --------------------
def classify_lanes_live():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    pca = initialize_pca()
    set_motor_speed(pca, 0.12)

    print("\n[INFO] Press Ctrl+C in terminal to stop the program.\n")
    last_print_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flipped = cv2.flip(gray, 0)
            height, width = flipped.shape[:2]

            edges = cannyEdgeDetector(flipped)
            roi_edges = getROI(edges)

            lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

            output_image = cv2.cvtColor(flipped, cv2.COLOR_GRAY2BGR)
            left_lines, right_lines = [], []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        if slope < -0.3:
                            left_lines.append((x1, y1, x2, y2))
                            cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        elif slope > 0.3:
                            right_lines.append((x1, y1, x2, y2))
                            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if left_lines and right_lines:
                left_x_bottom = int(np.mean([min(line[0], line[2]) for line in left_lines]))
                right_x_bottom = int(np.mean([max(line[0], line[2]) for line in right_lines]))
                lane_center = (left_x_bottom + right_x_bottom) // 2
                cam_center = width // 2
                offset = lane_center - cam_center
                dist_to_left = cam_center - left_x_bottom
                dist_to_right = right_x_bottom - cam_center

                # Steering calculation
                camera_angle_rad = math.atan2(offset, lookahead_distance)
                camera_angle_deg = math.degrees(camera_angle_rad)
                steering_angle = Kp * camera_angle_deg
                steering_percent = np.clip(steering_angle / max_steering_angle_deg, -1.0, 1.0)

                set_steering_angle(pca, steering_percent)

                # Logging
                current_time = time.time()
                if current_time - last_print_time >= 0.5:
                    print(f"[INFO] Offset: {offset} px")
                    print(f"[INFO] ← Left Lane Distance: {dist_to_left} px")
                    print(f"[INFO] → Right Lane Distance: {dist_to_right} px")
                    print(f"[INFO] Camera Angle: {camera_angle_deg:.2f}°")
                    print(f"[INFO] Steering Angle (Kp*offset): {steering_angle:.2f}°")
                    print(f"[INFO] Steering Percent: {steering_percent:.2f}\n")

                    save_debug_frames(frame, roi_edges, output_image)
                    last_print_time = current_time

    except KeyboardInterrupt:
        print("\n[INFO] Stopping the car...")

    # Stop motors on exit
    set_motor_speed(pca, 0)
    set_steering_angle(pca, 0)
    cap.release()
    print("[INFO] Program terminated cleanly.")

# -------------------- Entry Point --------------------
def main():
    classify_lanes_live()

if __name__ == "__main__":
    main()
