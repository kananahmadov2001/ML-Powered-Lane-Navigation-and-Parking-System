import cv2
import numpy as np
import math
import time
import os
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685

# ------------------ Config ------------------
LOOKAHEAD_DISTANCE = 240
STEERING_GAIN_KP = 0.5
MAX_STEERING_ANGLE_DEG = 30

# ------------------ PCA9685 Setup ------------------
def initialize_pca9685():
    i2c = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c)
    pca.frequency = 100
    return pca

def set_steering_degrees(pca, angle_deg):
    angle_deg = max(-MAX_STEERING_ANGLE_DEG, min(MAX_STEERING_ANGLE_DEG, angle_deg))
    steering_percent = angle_deg / MAX_STEERING_ANGLE_DEG
    neutral = 65535 * 0.15
    delta = steering_percent * 0.025 * 65535
    duty = int(neutral + delta)
    duty = max(0, min(65535, duty))
    pca.channels[14].duty_cycle = duty

# ------------------ ROI Mask ------------------
def apply_roi_mask(image):
    height, width = image.shape[:2]
    # Trapezoidal region of interest (bottom wide, top narrow)
    trapezoid = np.array([[
        (0, height),             # Bottom-left
        (width, height),             # Bottom-right
        (int(width * 0.7), int(height * 0.35)),  # Top-right
        (int(width * 0.3), int(height * 0.35))   # Top-left
    ]], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, trapezoid, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def lane_centering():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    pca = initialize_pca9685()
    os.makedirs("processed_images", exist_ok=True)

    print("[INFO] Capturing and saving 20 processed images (0.5s apart)...")

    try:
        for i in range(40):
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # Flip image upright if inverted
            flipped = cv2.flip(frame, 0)

            # Grayscale and ROI mask
            gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
            masked = apply_roi_mask(gray)

            # Edge detection
            edges = cv2.Canny(masked, 150, 160)
            height, width = edges.shape
            output = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                    minLineLength=50, maxLineGap=20)
            left_lines, right_lines = [], []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        if slope < -0.2:
                            left_lines.append((x1, y1, x2, y2))
                            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        elif slope > 0.2:
                            right_lines.append((x1, y1, x2, y2))
                            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if left_lines and right_lines:
                left_x = np.mean([min(x1, x2) for x1, _, x2, _ in left_lines])
                right_x = np.mean([max(x1, x2) for x1, _, x2, _ in right_lines])
                lane_center = (left_x + right_x) / 2
                cam_center = width / 2
                offset = cam_center - lane_center

                theta_rad = math.atan2(offset, LOOKAHEAD_DISTANCE)
                theta_deg = math.degrees(theta_rad)
                steering_angle = STEERING_GAIN_KP * theta_deg

                set_steering_degrees(pca, steering_angle)

                print(f"[{i+1}/20] Offset: {offset:.2f}px | θ: {theta_deg:.2f}° | Steering: {steering_angle:.2f}°")
            else:
                print(f"[{i+1}/20] No valid lanes detected.")

            # Save annotated image
            filename = f"processed_images/frame_{i+1:02}.jpg"
            cv2.imwrite(filename, output)

            time.sleep(0.25)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    set_steering_degrees(pca, 0)
    cap.release()
    print("[INFO] Capture complete. Images saved in /processed_images")

# ------------------ Entry ------------------
def main():
    lane_centering()

if __name__ == "__main__":
    main()
