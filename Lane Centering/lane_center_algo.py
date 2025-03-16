import cv2
import numpy as np
import os
import math

# proportional gain for steering correction
Kp = 2

def apply_region_of_interest(image):
    """ This function takes an image (grayscale) input, then applies a mask to keep only the road section """
    height, width = image.shape[:2]  # getting height and width
    mask = np.zeros_like(image)      # creating black mask
    # creating a polygon (trapezoid) to isolate the road area (where lane markings are expected)
    polygon = np.array([[(0, height), (width, height), (int(0.6 * width), int(0.6 * height)), (int(0.4 * width), int(0.6 * height))]], np.int32)
    # (0, height) is bottom left
    # (width, height) is bottom right
    # (int(0.6 * width), int(0.6 * height)) is top right
    # (int(0.4 * width), int(0.6 * height)) is top left
    cv2.fillPoly(mask, polygon, 255)              # filling the polygon with white
    masked_image = cv2.bitwise_and(image, mask)   # applying the mask to the image
    return masked_image

def extrapolate_lines(lines, height):
    """ This function extends detected lane lines to cover the full road """
    if len(lines) == 0:  # error handling if no lines detected
        return None
    x_coords = []
    y_coords = []
    for line in lines:
        # line data format will be like line = (x1, y1, x2, y2)
        x_coords.extend([line[0], line[2]])      # x-coordinates of detected lane lines, x1 (start) and x2 (end)
        y_coords.extend([line[1], line[3]])      # y-coordinates of detected lane lines, y1 (start) and y2 (end)
    
    poly = np.polyfit(y_coords, x_coords, 1)     # fitting a line using least squares (x = ym + b) to the points because lanes are mostly vertical
    y1 = height                                  # bottom of the image
    y2 = int(0.6 * height)                       # extending the line up to 60% of the image height
    x1 = int(np.polyval(poly, y1))               # calculating x-coordinates of the line at y1
    x2 = int(np.polyval(poly, y2))               # calculating x-coordinates of the line at y2
    return (x1, y1, x2, y2)

def classify_and_draw_lanes(image_path):
    """ This function detects lane lines, computes lane center, applies steering correction, and saves the image """
    # detecting lane lines---------------------------------------------------------------------------------------------
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # read and convert the image to grayscale
    if image is None:                                     # error handling if image not found
        print(f"Error: Could NOT read image {image_path}")
        return None
    height, width = image.shape[:2]                       # image dimensions
    masked_image = apply_region_of_interest(image)        # applying the region of interest mask
    edges = cv2.Canny(masked_image, 50, 150)              # applying Canny edge detection to detect edges
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)   # detecting lines using Hough Transform
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)                                          # converting grayscale image to color for visualization
    left_lines = []   # detected left lane lines
    right_lines = []  # detected right lane lines

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]                               # extracting line coordinates
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf  # computing slope, avoiding division by zero
            # classifying lane lines based on slope
            if slope < -0.3:                                       # left lane (negative slope)
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:                                      # right lane (positive slope)
                right_lines.append((x1, y1, x2, y2))
    
    # extending lane lines---------------------------------------------------------------------------------------------
    left_lane = extrapolate_lines(left_lines, height)
    right_lane = extrapolate_lines(right_lines, height)
    # drawing lanes on the output image
    if left_lane:
        cv2.line(output_image, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 3)      # blue
    if right_lane:
        cv2.line(output_image, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 3)  # green

    # calculating lane center, steering angle and correction -----------------------------------------------------------
    if left_lane and right_lane:
        x_mid = (left_lane[0] + right_lane[0]) // 2                           # x-coordinate of the lane center
        y_fixed = left_lane[1] - 15                                           # moving the dot up to avoid cutoff
        cv2.circle(output_image, (x_mid, y_fixed), 12, (0, 255, 255), -1)     # drawing the lane center as a yellow dot

        # compute steering angle
        x_cam = width // 2                    # camera center (ideal position)
        offset = x_mid - x_cam                # offset from the center of the lane
        y_lookahead = int(0.6 * height)       # lookahead distance for steering angle calculation
        theta = math.degrees(math.atan2(offset, y_lookahead))

        # compute steering PWM using P-Controller
        steering_pwm = int(1500 + (Kp * theta))
        steering_pwm = max(1200, min(1800, steering_pwm))  # clamping PWM to prevent extreme turns

        print(f"\nSteering Calculations:")
        print(f" - Offset: {offset} pixels")
        print(f" - Steering Angle (theta): {theta:.2f} degrees")
        print(f" - Steering PWM: {steering_pwm}")

        # computing steering arrow for visualization
        arrow_length = 100
        x_arrow = int(x_cam + arrow_length * math.sin(math.radians(theta)))
        y_arrow = int(height - arrow_length * math.cos(math.radians(theta)))
        arrow_color = (0, 0, 255) if theta < 0 else (0, 255, 0)                                             # red for left turn, green for right turn
        cv2.arrowedLine(output_image, (x_cam, height), (x_arrow, y_arrow), arrow_color, 5, tipLength=0.3)   # drawing steering arrow
        
        # displaying calculated values on the image
        cv2.putText(output_image, f"Offset: {offset}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_image, f"Angle: {theta:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_image, f"PWM: {steering_pwm}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return output_image

def main():
    input_image = input("input image file path: ")
    output_image = classify_and_draw_lanes(input_image)
    if output_image is not None:
        directory, filename = os.path.split(input_image)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_steering{ext}"
        output_path = os.path.join(directory, output_filename)
        cv2.imwrite(output_path, output_image)
        print(f"\nOutput Image Saved at: {output_path}")

if __name__ == "__main__":
    main()
