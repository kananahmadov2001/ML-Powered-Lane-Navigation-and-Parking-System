import cv2, numpy as np, math, time, logging
from pathlib import Path
from adafruit_pca9685 import PCA9685
import busio
from board import SCL, SDA

logging.basicConfig(level=logging.INFO)

class LaneCentering:
    def __init__(self, out_dir: Path, config):
        self.cfg = config
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Capture is {actual_w}×{actual_h}")

        # Initialize PCA9685 for steering
        self.pca = self._init_pca()

        # Prepare output directory
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True)

        self.frame_idx = 0
        self.prev_steering = 0.0

    def _init_pca(self):
        i2c = busio.I2C(SCL, SDA)
        pca = PCA9685(i2c)
        pca.frequency = self.cfg.PCA_FREQ
        return pca

    def run(self):
        try:
            while True:
                frame = self._grab_frame()
                masked, canvas = self._preprocess(frame)
                left, right = self._detect_lines(masked)
                angle = self._compute_steering(left, right)
                angle = self._smooth(angle)
                self._apply_steering(angle)
                self._annotate(canvas, left, right)
                self._save(canvas)
                time.sleep(self.cfg.LOOP_DELAY)
        except KeyboardInterrupt:
            logging.info("Stopping…")
        finally:
            self._cleanup()

    def _grab_frame(self):
        ret, frame = self.cap.read()
        return cv2.flip(frame, 0)

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        masked = self._apply_roi(inv)
        blurred = cv2.GaussianBlur(masked, self.cfg.BLUR_KERNEL, 0)
        canvas = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        return blurred, canvas

    def _apply_roi(self, img):
        h, w = img.shape[:2]
        poly = np.array([[
            (0, h), (w, h),
            (int(0.7 * w), int(0.35 * h)),
            (int(0.3 * w), int(0.35 * h))
        ]], dtype=np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, poly, 255)
        return cv2.bitwise_and(img, mask)
   
    def _detect_lines(self, img):
        edges = cv2.Canny(img, *self.cfg.CANNY_THRESH)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=self.cfg.HOUGH_THRESH,
            minLineLength=self.cfg.MIN_LEN,
            maxLineGap=self.cfg.MAX_GAP)
        left, right = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:,0]:
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                if slope < -self.cfg.SLOPE_THRESH:
                    left.append((x1, y1, x2, y2))
                elif slope > self.cfg.SLOPE_THRESH:
                    right.append((x1, y1, x2, y2))
        return left, right

    def _compute_steering(self, left, right):
        if not (left and right):
            print(f"[{self.frame_idx}] No valid lanes detected.")
            return 0.0
        # calculate lane center offset and steering
        lx = np.mean([min(x1, x2) for x1, _, x2, _ in left])
        rx = np.mean([max(x1, x2) for x1, _, x2, _ in right])
        center_offset = (self.cfg.FRAME_WIDTH / 2) - ((lx + rx) / 2)
        theta = math.degrees(math.atan2(center_offset, self.cfg.LOOKAHEAD))
        angle = -self.cfg.STEER_KP * theta
        print(f"[{self.frame_idx}] Offset={center_offset:.2f}px | θ={theta:.2f}° | Steering={angle:.2f}°")
        return angle

    def _smooth(self, angle):
        alpha = self.cfg.SMOOTH_ALPHA
        sm = alpha * angle + (1 - alpha) * self.prev_steering
        self.prev_steering = sm
        return sm

    def _apply_steering(self, angle_deg: float):
        a = max(-self.cfg.MAX_STEER_DEG, min(self.cfg.MAX_STEER_DEG, angle_deg))
        pct = a / self.cfg.MAX_STEER_DEG
        neutral = int(65535 * 0.15)
        delta = int(pct * 0.025 * 65535)
        duty = max(0, min(65535, neutral + delta))
        self.pca.channels[self.cfg.STEERING_CHANNEL].duty_cycle = duty

    def _annotate(self, canvas: np.ndarray, left, right):
        for x1, y1, x2, y2 in left:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for x1, y1, x2, y2 in right:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Steer: {self.prev_steering:.1f}°"
        cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _save(self, img):
        path = self.out_dir / f"frame_{self.frame_idx:05}.jpg"
        cv2.imwrite(str(path), img)
        self.frame_idx += 1

    def _cleanup(self):
        self._apply_steering(0)
        self.pca.deinit()
        self.cap.release()

# --- Configuration ---
class Config:
    NEUTRAL_FRAC   = 0.15
    SWING_FRAC     = 0.025   # half the total swing
    STEERING_CH    = 14
    FRAME_WIDTH      = 640
    LOOKAHEAD        = 530
    STEER_KP         = 4
    MAX_STEER_DEG    = 30
    PCA_FREQ         = 100
    BLUR_KERNEL      = (5,5)
    CANNY_THRESH     = (50,120)
    HOUGH_THRESH     = 50
    MIN_LEN          = 50
    MAX_GAP          = 20
    SLOPE_THRESH     = 0.2
    LOOP_DELAY       = 0.1
    SMOOTH_ALPHA     = 0.2
    STEERING_CHANNEL = 14

# --- Entry point ---
if __name__ == "__main__":
    LaneCentering(Path("processed_images"), Config()).run()
