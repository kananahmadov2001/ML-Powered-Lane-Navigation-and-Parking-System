import cv2
import numpy as np
import os

def apply_canny_edge_detection(image_path, low_threshold, high_threshold):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Flip the image across the x-axis
    image = cv2.flip(image, 0)

    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Generate output filename by adding '_edge' to the input filename
    directory, filename = os.path.split(image_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_edge{ext}"
    output_path = os.path.join(directory, output_filename)

    # Save the edge-detected image
    cv2.imwrite(output_path, edges)
    print(f"Processed: {image_path} -> {output_path}")

def main():
    input_image = input("Enter the input image file path: ")  # User-defined input image

    # Ask user for threshold values within the valid range
    low_threshold = int(input("Enter low threshold for Canny edge detection (0-255): "))
    high_threshold = int(input("Enter high threshold for Canny edge detection (0-255): "))

    # Ensure valid threshold values
    if not (0 <= low_threshold <= 255 and 0 <= high_threshold <= 255):
        print("Error: Threshold values must be between 0 and 255.")
        return

    apply_canny_edge_detection(input_image, low_threshold, high_threshold)

if __name__ == "__main__":
    main()



