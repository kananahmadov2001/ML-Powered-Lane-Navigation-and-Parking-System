import cv2
import numpy as np
import math
import time

# Parameters
lookahead_distance = 240  # Can be tuned (in pixels or real-world units if mapped)
Kp = 0.5  # Steering control gain

# Canny Edge Detector function
def canyEdgeDetector(image):
    edged = cv2.Canny(image, 50, 150)
    return edged

# Region of Interest (ROI) Masking function
def getROI(image):
    height, width = image.shape[:2]
    triangle = np.array([[
        (100, int(height*0.75)),
        (width, int(height*0.75)),
        (int(width / 2 + 200), int(height / 2.1))
    ]], dtype=np.int32)

    black_image = np.zeros_like(image)
    mask = cv2.fillPoly(black_image, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Lane classification and angle computation
def classify_lanes_live():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("\nPress 'q' to quit the video stream.\n")
    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flipped = cv2.flip(gray, 0)
        height, width = flipped.shape[:2]

        # Step 1: Edge detection
        edges = canyEdgeDetector(flipped)

        # Step 2: ROI mask applied after edge detection
        roi_edges = getROI(edges)

        # Step 3: Detect lines
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

            # Draw lines
            cv2.line(output_image, (lane_center, height - 10), (lane_center, height - 50), (0, 255, 255), 2)
            cv2.line(output_image, (cam_center, height - 10), (cam_center, height - 50), (255, 255, 255), 2)

            # Angle calculations
            camera_angle_rad = math.atan2(offset, lookahead_distance)
            camera_angle_deg = math.degrees(camera_angle_rad)
            steering_angle = Kp * camera_angle_deg

            # Print every 0.5 seconds
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                print(f"[INFO] Offset: {offset} px")
                print(f"[INFO] ← Left Lane Distance: {dist_to_left} px")
                print(f"[INFO] → Right Lane Distance: {dist_to_right} px")
                print(f"[INFO] Camera Angle: {camera_angle_deg:.2f}°")
                print(f"[INFO] Steering Angle (Kp*offset): {steering_angle:.2f}°\n")
                last_print_time = current_time

        # Show outputs
        cv2.imshow("Original", frame)
        cv2.imshow("ROI + Canny Edges", roi_edges)
        cv2.imshow("Lane Classification + Distances", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main entry point
def main():
    classify_lanes_live()

if __name__ == "__main__":
    main()
