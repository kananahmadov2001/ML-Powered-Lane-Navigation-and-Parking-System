import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
import os  # OS module for file path handling
import math  # Math module for trigonometric calculations

# Proportional Gain for Steering Correction
Kp = 2  # Tune this value to adjust how aggressively the car corrects its position

def apply_region_of_interest(image):
    """Applies a mask to keep only the road section."""
    height, width = image.shape[:2]  # Get image dimensions
    mask = np.zeros_like(image)  # Create an empty black mask of the same size as the image

    # Define a polygon to isolate the road area
    polygon = np.array([
        [(0, height), (width, height), (int(0.6 * width), int(0.6 * height)), (int(0.4 * width), int(0.6 * height))]
    ], np.int32)  # The polygon defines a trapezoidal region of interest (the road section)

    cv2.fillPoly(mask, polygon, 255)  # Fill the defined polygon with white (255)
    masked_image = cv2.bitwise_and(image, mask)  # Apply the mask to the image
    return masked_image  # Return the masked image

def extrapolate_lines(lines, height):
    """Extend detected lane lines to cover the full road"""
    if len(lines) == 0:
        return None  # If no lines detected, return None
    x_coords = []
    y_coords = []
    for line in lines:
        x_coords.extend([line[0], line[2]])  # Collect all x-coordinates of detected lane lines
        y_coords.extend([line[1], line[3]])  # Collect all y-coordinates of detected lane lines
    poly = np.polyfit(y_coords, x_coords, 1)  # Fit a straight line (y = mx + b) to the points
    y1 = height  # Bottom of the image (road level)
    y2 = int(0.6 * height)  # Extend the line up to 60% of the image height
    x1 = int(np.polyval(poly, y1))  # Calculate x-coordinates of the line at y1
    x2 = int(np.polyval(poly, y2))  # Calculate x-coordinates of the line at y2
    return (x1, y1, x2, y2)  # Return both points representing the extrapolated lane line

def classify_and_draw_lanes(image_path):
    """Detects lane lines, computes lane center, applies steering correction, and saves the image."""
    # Read and convert the image to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
    if image is None:
        print(f"Error: Could not read image {image_path}")  # Error handling if file not found
        return None

    height, width = image.shape[:2]  # Get image dimensions

    # Apply region of interest (ROI) mask
    masked_image = apply_region_of_interest(image)

    # Apply Canny edge detection to detect edges in the image
    edges = cv2.Canny(masked_image, 50, 150)  

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

    # Convert grayscale image to color for visualization
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    left_lines = []  # Store detected left lane lines
    right_lines = []  # Store detected right lane lines

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Extract line coordinates
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf  # Compute slope, avoid division by zero

            # Classify lane lines based on slope
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

    # Calculate lane center
    if left_lane and right_lane:
        x_mid = (left_lane[0] + right_lane[0]) // 2  # Compute x-coordinate of the lane center
        y_fixed = left_lane[1] - 15  # Move the dot up slightly to avoid cutoff

        # Draw the lane center as a yellow dot
        cv2.circle(output_image, (x_mid, y_fixed), 12, (0, 255, 255), -1)  

        # Compute steering angle using trigonometry
        x_cam = width // 2  # Camera center (ideal position)
        offset = x_mid - x_cam  # Offset from the center of the lane
        y_lookahead = int(0.6 * height)  # Set a lookahead distance for steering angle calculation

        # Compute theta (steering angle)
        theta = math.degrees(math.atan2(offset, y_lookahead))

        # Compute Steering PWM using P-Controller
        steering_pwm = int(1500 + (Kp * theta))
        steering_pwm = max(1200, min(1800, steering_pwm))  # Clamp PWM value to prevent extreme turns

        # Print steering information
        print(f"\nSteering Calculation:")
        print(f" - Offset: {offset} pixels")
        print(f" - Steering Angle (theta): {theta:.2f} degrees")
        print(f" - Steering PWM: {steering_pwm}")

        # Compute steering arrow for visualization
        arrow_length = 100
        x_arrow = int(x_cam + arrow_length * math.sin(math.radians(theta)))
        y_arrow = int(height - arrow_length * math.cos(math.radians(theta)))

        # Choose color: Red for left turn, Green for right turn
        arrow_color = (0, 0, 255) if theta < 0 else (0, 255, 0)

        # Draw steering arrow
        cv2.arrowedLine(output_image, (x_cam, height), (x_arrow, y_arrow), arrow_color, 5, tipLength=0.3)

        # Display calculated values on the image
        cv2.putText(output_image, f"Offset: {offset}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_image, f"Angle: {theta:.2f}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_image, f"PWM: {steering_pwm}", (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return output_image  # Return the processed image with lane markings and steering information

def main():
    input_image = input("Enter the input image file path: ")  # Ask the user for an image file

    output_image = classify_and_draw_lanes(input_image)  # Process the image

    if output_image is not None:
        # Save the processed image with a new filename
        directory, filename = os.path.split(input_image)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_steering{ext}"
        output_path = os.path.join(directory, output_filename)

        cv2.imwrite(output_path, output_image)
        print(f"\nProcessed Image Saved at: {output_path}")

if __name__ == "__main__":
    main()  # Run the main function
