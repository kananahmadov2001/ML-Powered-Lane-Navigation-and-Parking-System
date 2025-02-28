import cv2
import numpy as np
import math
import time
from board import SCL, SDA
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

# Initialize I2C and PWM
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 100

# Steering and motor channels
STEERING_CHANNEL = 14
MOTOR_CHANNEL = 15

# Proportional gain for steering control
Kp = 0.4  # Adjust to tune responsiveness

# Initialize steering and motor
steering_servo = servo.Servo(pca.channels[STEERING_CHANNEL], min_pulse=500, max_pulse=2400)
motor = pca.channels[MOTOR_CHANNEL]

def process_frame(frame):
    """
    Process the input frame to detect lane lines and compute the steering angle.
    """
    # Convert to grayscale and Apply Gaussian Blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest (ROI) mask
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Define the polygon mask for the bottom half of the image
    roi_vertices = np.array([[ 
        (0, height), 
        (width // 2 - 50, height // 2 + 50), 
        (width // 2 + 50, height // 2 + 50), 
        (width, height)
    ]], dtype=np.int32)

    # Apply the mask
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lane lines using Hough Transform
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

    return frame, lines

def compute_lane_center(lines, width):
    """
    Compute the midpoint between detected lane lines to determine the lane center.
    """
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero

            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    if not left_lines or not right_lines:
        return None  # No reliable lane lines detected

    # Compute the average position of left and right lanes
    left_x = np.mean([x1 for x1, _, x2, _ in left_lines] + [x2 for x1, _, x2, _ in left_lines])
    right_x = np.mean([x1 for x1, _, x2, _ in right_lines] + [x2 for x1, _, x2, _ in right_lines])

    # Compute the midpoint of the lane
    lane_center = (left_x + right_x) / 2

    return lane_center

def compute_steering_angle(lane_center, width):
    """
    Compute the required steering angle based on the lane center position.
    """
    # Compute error (difference between lane center and image center)
    image_center = width // 2
    error = lane_center - image_center

    # Compute steering angle (P-controller)
    steering_angle = Kp * error  # Proportional control

    return max(0, min(180, 90 + steering_angle))  # Keep within valid range

def set_motor_speed(percent):
    """
    Convert percentage (-1 to 1) to 16-bit PWM duty cycle.
    """
    speed = ((percent) * 3277) + (65535 * 0.15)
    motor.duty_cycle = math.floor(speed)

def main():
    """
    Main function to capture video feed, detect lanes, and compute steering control.
    """
    cap = cv2.VideoCapture(0)  # Use PiCamera or webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, lines = process_frame(frame)
        height, width = processed_frame.shape[:2]

        # Compute lane center
        lane_center = compute_lane_center(lines, width)

        if lane_center is not None:
            # Compute steering angle
            steering_angle = compute_steering_angle(lane_center, width)
            
            # Set steering angle
            steering_servo.angle = steering_angle
            
            # Move the car forward slowly
            set_motor_speed(0.10)  # 10% throttle for slow movement
            
            # Display the computed steering angle
            cv2.putText(processed_frame, f"Steering Angle: {steering_angle:.2f} degrees", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw lane center
            cv2.circle(processed_frame, (int(lane_center), height - 50), 10, (0, 0, 255), -1)

        else:
            # Stop the car if no lane detected
            set_motor_speed(0)

        # Show the processed frame
        cv2.imshow("Lane Detection", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop motors when exiting
    set_motor_speed(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
