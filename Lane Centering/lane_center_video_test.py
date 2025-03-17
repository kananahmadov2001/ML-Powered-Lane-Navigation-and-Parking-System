import cv2
import numpy as np
import math

# Proportional Gain for Steering Correction
Kp = 2  # Tune this value

def apply_region_of_interest(image):
    """Applies a mask to keep only the road section."""
    height, width = image.shape[:2]
    mask = np.zeros_like(image)

    # Define a polygon to isolate the road area (trapezoid)
    polygon = np.array([
        [(0, height), (width, height), (int(0.6 * width), int(0.6 * height)), (int(0.4 * width), int(0.6 * height))]
    ], np.int32)

    cv2.fillPoly(mask, polygon, 255)  # Fill polygon with white
    masked_image = cv2.bitwise_and(image, mask)  # Apply mask
    return masked_image

def extrapolate_lines(lines, height):
    """Extend detected lane lines to cover the full road"""
    if len(lines) == 0:
        return None
    x_coords = []
    y_coords = []
    for line in lines:
        x_coords.extend([line[0], line[2]])
        y_coords.extend([line[1], line[3]])
    
    poly = np.polyfit(y_coords, x_coords, 1)  # Fit a line y = mx + b
    y1 = height
    y2 = int(0.6 * height)  # Extend up to 60% of the image height
    x1 = int(np.polyval(poly, y1))
    x2 = int(np.polyval(poly, y2))
    return (x1, y1, x2, y2)

def process_frame(frame):
    """Processes a single frame to detect lanes and compute steering angle."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    height, width = image.shape[:2]

    # Apply ROI Mask
    masked_image = apply_region_of_interest(image)

    # DEBUG: Show the ROI Mask
    cv2.imshow("ROI Mask", masked_image)

    # Apply Canny Edge Detection (Lowered Thresholds for Better Sensitivity)
    edges = cv2.Canny(masked_image, 30, 100)

    # DEBUG: Show the Edge Detection Output
    cv2.imshow("Edges", edges)

    # Detect lines using Hough Transform (Less Strict Parameters)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=30)

    # Convert grayscale image to color for visualization
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf  # Avoid division by zero

            # Classify lane lines
            if slope < -0.3:  # Left lane (negative slope)
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:  # Right lane (positive slope)
                right_lines.append((x1, y1, x2, y2))

    # Extrapolate lane lines properly
    left_lane = extrapolate_lines(left_lines, height)
    right_lane = extrapolate_lines(right_lines, height)

    # Draw lanes on the output image
    if left_lane:
        cv2.line(output_image, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 3)  # Blue
    if right_lane:
        cv2.line(output_image, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 3)  # Green

    # Calculate lane center and steering angle
    if left_lane and right_lane:
        x_mid = (left_lane[0] + right_lane[0]) // 2
        y_fixed = left_lane[1] - 15  # Move the dot up slightly to avoid cutoff
        cv2.circle(output_image, (x_mid, y_fixed), 12, (0, 255, 255), -1)  # Yellow Dot at Lane Center

        # Compute steering angle
        x_cam = width // 2  
        offset = x_mid - x_cam
        y_lookahead = int(0.6 * height)  # Lookahead distance
        theta = math.degrees(math.atan2(offset, y_lookahead))

        # Compute Steering PWM using P-Controller
        steering_pwm = int(1500 + (Kp * theta))
        steering_pwm = max(1200, min(1800, steering_pwm))  # Clamp PWM to safe range

        # Print steering information
        print(f"\nSteering Calculation:")
        print(f" - Offset: {offset} pixels")
        print(f" - Steering Angle (theta): {theta:.2f} degrees")
        print(f" - Steering PWM: {steering_pwm}")

        # Compute steering arrow visualization
        arrow_length = 100
        x_arrow = int(x_cam + arrow_length * math.sin(math.radians(theta)))
        y_arrow = int(height - arrow_length * math.cos(math.radians(theta)))
        arrow_color = (0, 0, 255) if theta < 0 else (0, 255, 0)
        cv2.arrowedLine(output_image, (x_cam, height), (x_arrow, y_arrow), arrow_color, 5, tipLength=0.3)

        # Display calculated values on the image
        cv2.putText(output_image, f"Offset: {offset}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_image, f"Angle: {theta:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_image, f"PWM: {steering_pwm}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return output_image

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        output_frame = process_frame(frame)  # Process each frame
        cv2.imshow("Lane Detection & Steering", output_frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
