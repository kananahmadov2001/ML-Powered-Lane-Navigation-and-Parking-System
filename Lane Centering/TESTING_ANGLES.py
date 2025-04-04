import cv2
import numpy as np
import time
import math

def classify_lanes_live(low_threshold, high_threshold, Kp=0.1, lookahead_distance=400):
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

        # Convert and flip
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flipped = cv2.flip(gray, 0)

        height, width = flipped.shape[:2]

        # Edge detection
        edges = cv2.Canny(flipped, low_threshold, high_threshold)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

        # Prepare for drawing
        output_image = cv2.cvtColor(flipped, cv2.COLOR_GRAY2BGR)
        left_lines = []
        right_lines = []

        # Line classification
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
            left_x_bottom = int(np.mean([min(x1, x2) for x1, y1, x2, y2 in left_lines]))
            right_x_bottom = int(np.mean([max(x1, x2) for x1, y1, x2, y2 in right_lines]))

            lane_center = (left_x_bottom + right_x_bottom) // 2
            cam_center = width // 2
            offset = lane_center - cam_center
            dist_to_left = cam_center - left_x_bottom
            dist_to_right = right_x_bottom - cam_center

            # Calculate angles
            camera_angle_rad = math.atan2(offset, lookahead_distance)
            camera_angle_deg = math.degrees(camera_angle_rad)
            steering_angle = Kp * offset  # Optional: convert to degrees if needed

            # Print every 0.5s
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                print(f"[INFO] Offset: {offset} px")
                print(f"[INFO] ← Left Lane Distance: {dist_to_left} px")
                print(f"[INFO] → Right Lane Distance: {dist_to_right} px")
                print(f"[INFO] Camera Angle: {camera_angle_deg:.2f}°")
                print(f"[INFO] Steering Angle (Kp*offset): {steering_angle:.2f}°\n")
                last_print_time = current_time

        # Show views
        cv2.imshow("Original", frame)
        cv2.imshow("Canny Edge Detection", edges)
        cv2.imshow("Lane Classification + Angles", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    try:
        low_threshold = int(input("Low Canny threshold (0-255): "))
        high_threshold = int(input("High Canny threshold (0-255): "))
        Kp = float(input("Enter Kp value (default is 0.1): "))
    except ValueError:
        print("Invalid input. Exiting.")
        return

    classify_lanes_live(low_threshold, high_threshold, Kp)

if __name__ == "__main__":
    main()
