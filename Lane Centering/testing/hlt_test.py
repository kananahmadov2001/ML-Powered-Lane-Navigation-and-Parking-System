import cv2
import numpy as np
import os

def apply_region_of_interest(image):
    """Applies a mask to keep only the road section."""
    height, width = image.shape[:2]
    mask = np.zeros_like(image)

    # Define a polygon to isolate the road area
    polygon = np.array([
        [(0, height), (width, height), (int(0.6 * width), int(0.6 * height)), (int(0.4 * width), int(0.6 * height))]
    ], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
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
    y2 = int(0.6 * height)  # Extend to about 60% of the image
    x1 = int(np.polyval(poly, y1))
    x2 = int(np.polyval(poly, y2))
    return (x1, y1, x2, y2)

def classify_and_draw_lanes(image_path):
    """Detects lane lines and classifies them as left or right lanes."""
    # Read and convert the image to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # Apply region of interest (ROI) mask
    masked_image = apply_region_of_interest(image)

    # Apply Canny edge detection
    edges = cv2.Canny(masked_image, 50, 150)

    # Detect lines using Hough Transform with improved parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

    # Convert grayscale image to color for visualization
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf  # Avoid division by zero

            # Classify lane lines: discard nearly horizontal lines
            if slope < -0.3:  # Left lane (negative slope)
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:  # Right lane (positive slope)
                right_lines.append((x1, y1, x2, y2))

    # Apply extrapolation to extend lanes
    left_lane = extrapolate_lines(left_lines, image.shape[0])
    right_lane = extrapolate_lines(right_lines, image.shape[0])

    # Draw lanes on the output image
    if left_lane:
        cv2.line(output_image, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 3)  # Blue
    if right_lane:
        cv2.line(output_image, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 3)  # Green

    return output_image

def main():
    """Main function to run the lane detection pipeline."""
    input_image = input("Enter the input image file path: ")  # User-defined input image

    output_image = classify_and_draw_lanes(input_image)

    if output_image is not None:
        # Generate output filename
        directory, filename = os.path.split(input_image)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_lanes{ext}"
        output_path = os.path.join(directory, output_filename)

        # Save the processed image
        cv2.imwrite(output_path, output_image)
        print(f"Processed: {input_image} -> {output_path}")

if __name__ == "__main__":
    main()
