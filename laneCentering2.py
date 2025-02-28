import cv2
import numpy as np

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Resize for consistency
    image = cv2.resize(image, (640, 480))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI)
    mask = np.zeros_like(edges)
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width//2, height//2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    
    # Apply mask
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Line Transform for lane detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
    
    if lines is None:
        print("No lane lines detected.")
        return
    
    left_lines = []
    right_lines = []
    
    # Separate left and right lane lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
        if slope < -0.5:  # Left lane (negative slope)
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0.5:  # Right lane (positive slope)
            right_lines.append((x1, y1, x2, y2))

    # Compute lane center
    lane_center_x = width // 2  # Default center
    if left_lines and right_lines:
        left_x = np.mean([x1 for x1, y1, x2, y2 in left_lines])
        right_x = np.mean([x1 for x1, y1, x2, y2 in right_lines])
        lane_center_x = int((left_x + right_x) / 2)

    # Compute steering angle
    car_center_x = width // 2  # Assuming the car's camera is centered
    deviation = lane_center_x - car_center_x
    steering_angle = np.arctan2(deviation, height) * (180 / np.pi)

    # Predict action based on steering angle
    if steering_angle < -10:
        action = "Turn Left"
    elif steering_angle > 10:
        action = "Turn Right"
    else:
        action = "Go Straight"

    # Draw lane lines and center
    for x1, y1, x2, y2 in left_lines:
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for x1, y1, x2, y2 in right_lines:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.circle(image, (lane_center_x, height - 50), 10, (0, 0, 255), -1)
    
    # Display results
    cv2.putText(image, f"Steering: {steering_angle:.2f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, f"Action: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Lane Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Predicted Steering Angle: {steering_angle:.2f}°")
    print(f"Recommended Action: {action}")

# Example usage
process_image("image.jpg")  # Replace with the actual PiCamera image
