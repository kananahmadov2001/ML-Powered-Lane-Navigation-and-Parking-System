import cv2
import numpy as np
import time

def classify_lanes_live(low_threshold, high_threshold):
    cap = cv2.VideoCapture(0)
    
    # make sure camera is open
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    print("\nPress 'q' to quit the video stream.\n")

    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if already grayscale, flip image vertically
        # otherwise, first conver to grayscale
        if len(frame.shape) == 2:
            flipped = cv2.flip(frame, 0)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flipped = cv2.flip(gray, 0)

        height, width = flipped.shape[:2]  # getting image dimensions

        # applying canny edge detection to find edges
        edges = cv2.Canny(flipped, low_threshold, high_threshold)
        # detecting lines using HLT
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

        # Convert grayscale to color image for visualizing results
        # output_image = cv2.cvtColor(flipped, cv2.COLOR_GRAY2BGR)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]  # Extract line segment endpoints
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)

                    if slope < -0.3:  # threshold to detect left lane
                        left_lines.append((x1, y1, x2, y2))
                        cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # draw left lane in blue
                    elif slope > 0.3:  # threshold to detect right lane
                        right_lines.append((x1, y1, x2, y2))
                        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # draw right lane in green

        if left_lines and right_lines:
            # average x-coordinate near the bottom of the frame for both lanes
            left_x_bottom = int(np.mean([min(line[0], line[2]) for line in left_lines]))
            right_x_bottom = int(np.mean([max(line[0], line[2]) for line in right_lines]))

            lane_center = (left_x_bottom + right_x_bottom) // 2
            cam_center = width // 2
            offset = lane_center - cam_center
            dist_to_left = cam_center - left_x_bottom
            dist_to_right = right_x_bottom - cam_center

            # PRINT offset and distances every 0.5 seconds
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                print(f"Offset: {offset} px")
                print(f"Left Lane Distance: {dist_to_left} px")
                print(f"Right Lane Distance: {dist_to_right} px\n")
                last_print_time = current_time

        cv2.imshow("Original", frame)
        cv2.imshow("Canny Edge Detection", edges)
        cv2.imshow("Lane Classification + Distances", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    try:
        low_threshold = int(input("low Canny threshold (0-255): "))
        high_threshold = int(input("high Canny threshold (0-255): "))
    except ValueError:
        return
        
    classify_lanes_live(low_threshold, high_threshold)

if __name__ == "__main__":
    main()
