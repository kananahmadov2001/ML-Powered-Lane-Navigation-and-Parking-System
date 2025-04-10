import cv2
import numpy as np
import time
import math
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
import os

# ---------- PCA9685 SETUP ----------
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
    neutral_duty = 0.15 * 65535  # center position
    delta = angle_percent * 0.025 * 65535  # map [-1.0, 1.0] to ±2.5% swing
    duty = max(0, min(65535, neutral_duty + delta))
    pca.channels[14].duty_cycle = math.floor(duty)

# ---------- OPTIONAL: SAVE DEBUG FRAMES ----------
def save_debug_frames(frame, edges, output_image):
    os.makedirs("debug", exist_ok=True)
    timestamp = int(time.time() * 1000)
    cv2.imwrite(f"debug/frame_{timestamp}.jpg", frame)
    cv2.imwrite(f"debug/edges_{timestamp}.jpg", edges)
    cv2.imwrite(f"debug/lanes_{timestamp}.jpg", output_image)

# ---------- LANE DETECTION + STEERING ----------
def classify_lanes_live(low_threshold, high_threshold):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    pca = initialize_pca()
    set_motor_speed(pca, 0.15)  # Start moving forward

    print("\nPress 'q' to quit, 'e' for emergency stop (in terminal).\n")
    last_print_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flipped = cv2.flip(gray, 0)
            height, width = flipped.shape[:2]

            edges = cv2.Canny(flipped, low_threshold, high_threshold)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

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
                left_x_bottom = int(np.mean([min(x1, x2) for x1, _, x2, _ in left_lines]))
                right_x_bottom = int(np.mean([max(x1, x2) for x1, _, x2, _ in right_lines]))
                lane_center = (left_x_bottom + right_x_bottom) // 2
                cam_center = width // 2
                offset = lane_center - cam_center

                # ---- Steering Based on Angle Offset ----
                lookahead_distance = height * 0.5  # pixels
                camera_angle_rad = math.atan2(offset, lookahead_distance)
                camera_angle_deg = math.degrees(camera_angle_rad)

                Kp = 0.02  # Tune this gain
                steering_angle = Kp * camera_angle_deg
                max_steering_angle_deg = 30
                steering_percent = np.clip(steering_angle / max_steering_angle_deg, -1.0, 1.0)

                set_steering_angle(pca, steering_percent)

                # Print status every 0.5 sec + save debug frames
                current_time = time.time()
                if current_time - last_print_time >= 0.5:
                    dist_to_left = cam_center - left_x_bottom
                    dist_to_right = right_x_bottom - cam_center

                    print(f"Offset: {offset} px")
                    print(f"Camera Angle: {camera_angle_deg:.2f} deg")
                    print(f"Steering Angle: {steering_angle:.2f} deg")
                    print(f"Steering Percent: {steering_percent:.2f}")
                    print(f"Left Lane Distance: {dist_to_left} px")
                    print(f"Right Lane Distance: {dist_to_right} px\n")

                    save_debug_frames(frame, edges, output_image)
                    last_print_time = current_time

            # Key listener only works with GUI — use Ctrl+C to quit
            # Or implement a GPIO button for emergency stop in real testing

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping...")

    # Stop everything
    set_motor_speed(pca, 0)
    set_steering_angle(pca, 0)
    cap.release()
    print("System shut down cleanly.")

# ---------- MAIN ----------
def main():
    try:
        low_threshold = int(input("Low Canny threshold (0-255): "))
        high_threshold = int(input("High Canny threshold (0-255): "))
    except ValueError:
        print("Invalid input. Please enter integers.")
        return

    classify_lanes_live(low_threshold, high_threshold)

if __name__ == "__main__":
    main()
