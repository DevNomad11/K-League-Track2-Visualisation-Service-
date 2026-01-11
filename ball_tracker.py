"""
Robust Ball Tracker combining ROI cropping and Kalman filter prediction.

This module improves ball detection in football videos by:
1. Using Kalman filter to predict ball position when detection fails
2. Searching in ROI (Region of Interest) around predicted position with lower confidence
3. Maintaining ball trajectory even through temporary occlusions

Usage:
    from ball_tracker import RobustBallTracker

    tracker = RobustBallTracker(model)
    for frame in video_frames:
        ball_detection, status = tracker.detect(frame)
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class DetectionStatus(Enum):
    """Status of ball detection for each frame."""
    DETECTED = "detected"           # Ball found in primary detection
    ROI_RECOVERED = "roi_recovered" # Ball found in ROI search
    PREDICTED = "predicted"         # Using Kalman prediction (no detection)
    LOST = "lost"                   # Ball completely lost


@dataclass
class BallDetection:
    """Ball detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    center: Tuple[float, float]
    confidence: float
    status: DetectionStatus
    frame_idx: int

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class KalmanBallFilter:
    """
    Kalman filter for ball position and velocity estimation.

    State vector: [x, y, vx, vy] (position and velocity)
    Measurement: [x, y] (position only)
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 10.0):
        """
        Initialize Kalman filter.

        Args:
            process_noise: Process noise covariance (higher = trust measurements more)
            measurement_noise: Measurement noise covariance (higher = trust predictions more)
        """
        # State: [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],   # x = x + vx
            [0, 1, 0, 1],   # y = y + vy
            [0, 0, 1, 0],   # vx = vx
            [0, 0, 0, 1]    # vy = vy
        ], dtype=np.float32)

        # Measurement matrix (we only observe position)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Initial state covariance (high uncertainty)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100

        self.initialized = False
        self.last_bbox_size = None  # Store last known ball size

    def initialize(self, x: float, y: float, bbox_size: Tuple[float, float] = None):
        """Initialize filter with first detection."""
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
        if bbox_size:
            self.last_bbox_size = bbox_size

    def predict(self) -> Tuple[float, float]:
        """Predict next position."""
        if not self.initialized:
            return None

        prediction = self.kf.predict()
        return float(prediction[0]), float(prediction[1])

    def update(self, x: float, y: float, bbox_size: Tuple[float, float] = None):
        """Update filter with new measurement."""
        if not self.initialized:
            self.initialize(x, y, bbox_size)
            return

        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(measurement)

        if bbox_size:
            self.last_bbox_size = bbox_size

    def get_velocity(self) -> Tuple[float, float]:
        """Get current estimated velocity."""
        if not self.initialized:
            return 0.0, 0.0
        return float(self.kf.statePost[2]), float(self.kf.statePost[3])

    def get_predicted_bbox(self, predicted_center: Tuple[float, float]) -> np.ndarray:
        """Get predicted bounding box based on last known size."""
        if self.last_bbox_size is None:
            # Default ball size if unknown
            w, h = 30, 30
        else:
            w, h = self.last_bbox_size

        cx, cy = predicted_center
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


class RobustBallTracker:
    """
    Robust ball tracker combining ROI cropping and Kalman filter prediction.

    Detection strategy:
    1. Primary detection on full frame with normal confidence threshold
    2. If not found, predict position using Kalman filter
    3. Crop ROI around predicted position and search with lower confidence
    4. If still not found, use Kalman prediction as estimated position
    """

    def __init__(
        self,
        model,
        ball_class_id: int = 0,
        primary_conf: float = 0.25,
        roi_conf: float = 0.10,
        roi_scale: float = 4.0,
        max_frames_lost: int = 30,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        device: int = 0,
    ):
        """
        Initialize the robust ball tracker.

        Args:
            model: YOLO model for ball detection
            ball_class_id: Class ID for ball in the model
            primary_conf: Confidence threshold for primary detection
            roi_conf: Lower confidence threshold for ROI search
            roi_scale: Scale factor for ROI size relative to ball size
            max_frames_lost: Max frames to keep predicting before declaring lost
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
            device: CUDA device for inference
        """
        self.model = model
        self.ball_class_id = ball_class_id
        self.primary_conf = primary_conf
        self.roi_conf = roi_conf
        self.roi_scale = roi_scale
        self.max_frames_lost = max_frames_lost
        self.device = device

        # Kalman filter
        self.kalman = KalmanBallFilter(process_noise, measurement_noise)

        # Tracking state
        self.frames_lost = 0
        self.frame_idx = 0
        self.last_detection = None

        # History for analysis
        self.detection_history: List[BallDetection] = []

        # ROI parameters
        self.min_roi_size = 100  # Minimum ROI size in pixels
        self.roi_padding = 50   # Additional padding around predicted position

    def reset(self):
        """Reset tracker state."""
        self.kalman = KalmanBallFilter()
        self.frames_lost = 0
        self.frame_idx = 0
        self.last_detection = None
        self.detection_history = []

    def detect(self, frame: np.ndarray) -> Tuple[Optional[BallDetection], DetectionStatus]:
        """
        Detect ball in frame using multi-stage approach.

        Args:
            frame: Video frame (BGR format)

        Returns:
            Tuple of (BallDetection or None, DetectionStatus)
        """
        self.frame_idx += 1
        frame_height, frame_width = frame.shape[:2]

        # Stage 1: Primary detection on full frame
        ball_bbox, ball_conf = self._detect_ball(frame, self.primary_conf)

        if ball_bbox is not None:
            # Ball found in primary detection
            detection = self._create_detection(ball_bbox, ball_conf, DetectionStatus.DETECTED)
            self._update_state(detection)
            return detection, DetectionStatus.DETECTED

        # Stage 2: ROI search around predicted position
        if self.kalman.initialized:
            predicted_pos = self.kalman.predict()

            if predicted_pos is not None:
                # Get ROI around predicted position
                roi, roi_offset = self._get_roi(frame, predicted_pos, frame_width, frame_height)

                if roi is not None:
                    # Search in ROI with lower confidence
                    roi_bbox, roi_conf = self._detect_ball(roi, self.roi_conf)

                    if roi_bbox is not None:
                        # Convert ROI coordinates back to frame coordinates
                        frame_bbox = self._roi_to_frame_coords(roi_bbox, roi_offset)
                        detection = self._create_detection(
                            frame_bbox, roi_conf, DetectionStatus.ROI_RECOVERED
                        )
                        self._update_state(detection)
                        return detection, DetectionStatus.ROI_RECOVERED

        # Stage 3: Use Kalman prediction if within lost threshold
        if self.kalman.initialized and self.frames_lost < self.max_frames_lost:
            predicted_pos = self.kalman.predict()

            if predicted_pos is not None:
                # Validate prediction is within frame bounds
                px, py = predicted_pos
                if 0 <= px < frame_width and 0 <= py < frame_height:
                    predicted_bbox = self.kalman.get_predicted_bbox(predicted_pos)
                    detection = self._create_detection(
                        predicted_bbox, 0.0, DetectionStatus.PREDICTED
                    )
                    self.frames_lost += 1
                    self.detection_history.append(detection)
                    return detection, DetectionStatus.PREDICTED

        # Stage 4: Ball is lost
        self.frames_lost += 1
        return None, DetectionStatus.LOST

    def _detect_ball(
        self,
        image: np.ndarray,
        conf_threshold: float
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Run YOLO detection on image.

        Returns:
            Tuple of (bbox [x1,y1,x2,y2] or None, confidence)
        """
        results = self.model.predict(
            image,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
            classes=[self.ball_class_id]
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, 0.0

        # Get detection with highest confidence
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        best_conf = float(boxes.conf[best_idx])
        best_bbox = boxes.xyxy[best_idx].cpu().numpy()

        return best_bbox, best_conf

    def _get_roi(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        frame_width: int,
        frame_height: int
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        Extract ROI around predicted position.

        Returns:
            Tuple of (ROI image or None, offset (x, y))
        """
        cx, cy = center

        # Calculate ROI size based on last known ball size
        if self.kalman.last_bbox_size:
            w, h = self.kalman.last_bbox_size
            roi_size = max(w, h) * self.roi_scale
        else:
            roi_size = self.min_roi_size

        roi_size = max(roi_size, self.min_roi_size)
        roi_size += self.roi_padding

        # Account for velocity (expand ROI in direction of movement)
        vx, vy = self.kalman.get_velocity()
        velocity_padding = max(abs(vx), abs(vy)) * 2
        roi_size += velocity_padding

        # Calculate ROI bounds
        half_size = roi_size / 2
        x1 = int(max(0, cx - half_size))
        y1 = int(max(0, cy - half_size))
        x2 = int(min(frame_width, cx + half_size))
        y2 = int(min(frame_height, cy + half_size))

        # Ensure minimum ROI size
        if x2 - x1 < self.min_roi_size or y2 - y1 < self.min_roi_size:
            return None, (0, 0)

        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1)

    def _roi_to_frame_coords(
        self,
        roi_bbox: np.ndarray,
        offset: Tuple[int, int]
    ) -> np.ndarray:
        """Convert ROI coordinates to frame coordinates."""
        ox, oy = offset
        return np.array([
            roi_bbox[0] + ox,
            roi_bbox[1] + oy,
            roi_bbox[2] + ox,
            roi_bbox[3] + oy
        ])

    def _create_detection(
        self,
        bbox: np.ndarray,
        confidence: float,
        status: DetectionStatus
    ) -> BallDetection:
        """Create a BallDetection object."""
        center = (
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        )
        return BallDetection(
            bbox=bbox,
            center=center,
            confidence=confidence,
            status=status,
            frame_idx=self.frame_idx
        )

    def _update_state(self, detection: BallDetection):
        """Update tracker state with new detection."""
        # Update Kalman filter
        bbox_size = (detection.width, detection.height)
        self.kalman.update(detection.center[0], detection.center[1], bbox_size)

        # Reset lost counter
        self.frames_lost = 0

        # Store detection
        self.last_detection = detection
        self.detection_history.append(detection)

    def get_statistics(self) -> dict:
        """Get tracking statistics."""
        if not self.detection_history:
            return {}

        total = len(self.detection_history)
        detected = sum(1 for d in self.detection_history if d.status == DetectionStatus.DETECTED)
        recovered = sum(1 for d in self.detection_history if d.status == DetectionStatus.ROI_RECOVERED)
        predicted = sum(1 for d in self.detection_history if d.status == DetectionStatus.PREDICTED)

        return {
            'total_frames': total,
            'detected': detected,
            'roi_recovered': recovered,
            'predicted': predicted,
            'detection_rate': (detected + recovered) / total * 100,
            'recovery_rate': recovered / max(1, total - detected) * 100 if detected < total else 0,
        }


def draw_ball_detection(
    frame: np.ndarray,
    detection: Optional[BallDetection],
    status: DetectionStatus,
    show_status: bool = True,
    show_trajectory: bool = False,
    trajectory_history: List[BallDetection] = None
) -> np.ndarray:
    """
    Draw ball detection on frame with status indication.

    Args:
        frame: Video frame to draw on
        detection: Ball detection result
        status: Detection status
        show_status: Whether to show status text
        show_trajectory: Whether to draw trajectory line
        trajectory_history: List of past detections for trajectory

    Returns:
        Annotated frame
    """
    # Color based on status
    colors = {
        DetectionStatus.DETECTED: (0, 255, 255),      # Yellow - actual detection
        DetectionStatus.ROI_RECOVERED: (0, 255, 0),   # Green - recovered in ROI
        DetectionStatus.PREDICTED: (0, 165, 255),     # Orange - predicted
        DetectionStatus.LOST: (0, 0, 255),            # Red - lost
    }

    color = colors.get(status, (255, 255, 255))

    if detection is not None:
        # Draw bounding box
        x1, y1, x2, y2 = map(int, detection.bbox)
        thickness = 2 if status == DetectionStatus.DETECTED else 1
        line_type = cv2.LINE_AA if status != DetectionStatus.PREDICTED else cv2.LINE_4

        if status == DetectionStatus.PREDICTED:
            # Dashed box for predicted
            _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, line_type)

        # Draw center point
        cx, cy = int(detection.center[0]), int(detection.center[1])
        cv2.circle(frame, (cx, cy), 4, color, -1)

        # Draw label
        if show_status:
            label = f"Ball [{status.value}]"
            if detection.confidence > 0:
                label += f" {detection.confidence:.2f}"

            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 8),
                (x1 + text_width + 4, y1),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness
            )

    # Draw trajectory
    if show_trajectory and trajectory_history:
        points = []
        for det in trajectory_history[-30:]:  # Last 30 frames
            if det is not None:
                points.append((int(det.center[0]), int(det.center[1])))

        for i in range(1, len(points)):
            alpha = i / len(points)  # Fade older points
            pt_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, points[i-1], points[i], pt_color, 2)

    return frame


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top edge
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length)
    # Bottom edge
    _draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_length)
    # Left edge
    _draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_length)
    # Right edge
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Draw a dashed line."""
    x1, y1 = pt1
    x2, y2 = pt2

    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length == 0:
        return

    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    num_dashes = int(length / dash_length)

    for i in range(0, num_dashes, 2):
        start_x = int(x1 + i * dash_length * dx)
        start_y = int(y1 + i * dash_length * dy)
        end_x = int(x1 + min((i + 1) * dash_length, length) * dx)
        end_y = int(y1 + min((i + 1) * dash_length, length) * dy)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
