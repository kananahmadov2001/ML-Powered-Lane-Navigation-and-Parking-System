import cv2
import numpy as np

def classify_lanes_live(low_threshold, high_threshold):
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        return

    print("\nPress 'q' to quit the video stream.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert to grayscale and flip vertically
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flipped = cv2.flip(gray, 0)
        height, width = flipped.shape[:2]
        # canny edge detection
        edges = cv2.Canny(flipped, low_threshold, high_threshold)
        # hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)
        # convert grayscale to color image for visualization
        output_image = cv2.cvtColor(flipped, cv2.COLOR_GRAY2BGR)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)

                    if slope < -0.3:
                        left_lines.append((x1, y1, x2, y2))
                        cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
                    elif slope > 0.3:
                        right_lines.append((x1, y1, x2, y2))
                        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

        if left_lines and right_lines:
            left_x_bottom = int(np.mean([min(line[0], line[2]) for line in left_lines]))
            right_x_bottom = int(np.mean([max(line[0], line[2]) for line in right_lines]))

            lane_center = (left_x_bottom + right_x_bottom) // 2
            cam_center = width // 2
            offset = lane_center - cam_center

            dist_to_left = cam_center - left_x_bottom
            dist_to_right = right_x_bottom - cam_center

            cv2.line(output_image, (lane_center, height - 10), (lane_center, height - 50), (0, 255, 255), 2)  # Yellow
            cv2.line(output_image, (cam_center, height - 10), (cam_center, height - 50), (255, 255, 255), 2)  # White

            # Display
            cv2.putText(output_image, f"Offset: {offset} px", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(output_image, f"← Left Lane Dist: {dist_to_left} px", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(output_image, f"Right Lane Dist: {dist_to_right} px →", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
