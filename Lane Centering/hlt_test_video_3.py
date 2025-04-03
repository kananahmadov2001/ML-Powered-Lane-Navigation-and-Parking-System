import cv2
import numpy as np
import time

# Configuration
output_width = 1000                        # Width of bird’s-eye view in pixels
output_height = 600                        # Height of bird’s-eye view in pixels
lane_width_cm = 40                         # Measured real-world lane width
cm_per_px = lane_width_cm / output_width   # Updated scale

def classify_lanes_live(low_threshold, high_threshold):
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

        frame = cv2.flip(frame, 0)
        height, width = frame.shape[:2]

        # Define source points (tweak these if needed)
        src = np.float32([
            [width * 0.45, height * 0.65],   # top-left
            [width * 0.55, height * 0.65],   # top-right
            [width * 0.1, height * 0.95],    # bottom-left
            [width * 0.9, height * 0.95]     # bottom-right
        ])

        # Define destination points for warped top-down view
        dst = np.float32([
            [0, 0],
            [output_width, 0],
            [0, output_height],
            [output_width, output_height]
        ])

        # Perspective transform matrix + warp
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, M, (output_width, output_height))

        # Apply ROI mask to bottom half of the warped image
        roi_mask = np.zeros_like(warped)
        roi_points = np.array([
            [0, output_height],
            [0, output_height * 0.5],
            [output_width, output_height * 0.5],
            [output_width, output_height]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_points], (255, 255, 255))
        warped = cv2.bitwise_and(warped, roi_mask)

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

        output_image = warped.copy()
        left_lines = []
        right_lines = []

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
            left_x_bottom = int(np.mean([min(x1, x2) for (x1, y1, x2, y2) in left_lines]))
            right_x_bottom = int(np.mean([max(x1, x2) for (x1, y1, x2, y2) in right_lines]))

            lane_center = (left_x_bottom + right_x_bottom) // 2
            cam_center = output_width // 2
            offset = lane_center - cam_center
            dist_to_left = cam_center - left_x_bottom
            dist_to_right = right_x_bottom - cam_center

            # Convert pixel distances to centimeters
            offset_cm = offset * cm_per_px
            dist_to_left_cm = dist_to_left * cm_per_px
            dist_to_right_cm = dist_to_right * cm_per_px

            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                print(f"[INFO] Offset from Lane Center: {offset_cm:.2f} cm")
                print(f"[INFO] ← Left Lane Distance: {dist_to_left_cm:.2f} cm")
                print(f"[INFO] → Right Lane Distance: {dist_to_right_cm:.2f} cm\n")
                last_print_time = current_time

        # Display views
        cv2.imshow("Original View", frame)
        cv2.imshow("Bird’s-Eye View (With ROI)", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    try:
        low_threshold = int(input("Enter low Canny threshold (0-255): "))
        high_threshold = int(input("Enter high Canny threshold (0-255): "))
    except ValueError:
        print("Invalid input. Please enter integers.")
        return

    classify_lanes_live(low_threshold, high_threshold)

if __name__ == "__main__":
    main()
