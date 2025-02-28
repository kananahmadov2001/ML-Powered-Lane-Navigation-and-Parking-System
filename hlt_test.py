import cv2
import numpy as np
import os

def detect_vertical_lines(image_path, angle_threshold):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Create an output image in color to draw lines
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Convert to degrees

            # Check if the line is within the angle threshold for verticality
            if abs(angle) >= (90 - angle_threshold) and abs(angle) <= (90 + angle_threshold):
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw in red

    return output_image

def main():
    input_image = input("Enter the input image file path: ")  # User-defined input image
    angle_threshold = float(input("Enter the vertical line angle threshold (in degrees): "))

    output_image = detect_vertical_lines(input_image, angle_threshold)

    if output_image is not None:
        directory, filename = os.path.split(input_image)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_lines{ext}"
        output_path = os.path.join(directory, output_filename)
        cv2.imwrite(output_path, output_image)
        print(f"Processed: {input_image} -> {output_path}")

if __name__ == "__main__":
    main()



