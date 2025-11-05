"""
OpenCV-based visual detection module for extracting target position from frames.

This module provides utilities to detect the target (white blob) in the
egocentric grayscale frames using classical computer vision techniques.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class VisualDetector:
    """
    Detects target position in egocentric frames using OpenCV.

    The detector assumes the frame is a 64x64 grayscale image with a white
    target on a black background (radar-like visualization).
    """

    def __init__(
        self,
        frame_size: int = 64,
        min_contour_area: int = 10,
        blur_kernel: int = 5,
    ):
        """
        Initialize the visual detector.

        Args:
            frame_size: Size of the input frame (assumed square)
            min_contour_area: Minimum contour area to be considered a valid target
            blur_kernel: Kernel size for Gaussian blur (must be odd)
        """
        self.frame_size = frame_size
        self.min_contour_area = min_contour_area
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.center = frame_size / 2

    def detect_target(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Detect target position in a single frame.

        Args:
            frame: Grayscale image (H, W) with values in [0, 255]

        Returns:
            Tuple of (position, confidence):
                - position: (x, y) coordinates relative to frame center,
                           or None if no target detected
                - confidence: Detection confidence score (area of largest contour),
                             or None if no target detected
        """
        # Ensure frame is 2D grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)

        # Threshold to isolate white target
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None

        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Check if contour is large enough
        if area < self.min_contour_area:
            return None, None

        # Calculate centroid of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Convert to coordinates relative to center
        # (0, 0) is at center, positive x is right, positive y is down
        relative_x = cx - self.center
        relative_y = cy - self.center

        position = np.array([relative_x, relative_y], dtype=np.float32)
        confidence = area

        return position, confidence

    def detect_target_stacked(
        self, stacked_frames: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Detect target from stacked frames (uses most recent frame).

        Args:
            stacked_frames: Array of shape (num_frames, H, W) or (H, W, num_frames)

        Returns:
            Tuple of (position, confidence) from the most recent frame
        """
        # Handle different frame stacking formats
        if stacked_frames.shape[0] <= 4:  # Likely (num_frames, H, W)
            most_recent_frame = stacked_frames[-1]
        else:  # Likely (H, W, num_frames)
            most_recent_frame = stacked_frames[:, :, -1]

        return self.detect_target(most_recent_frame)

    def visualize_detection(
        self, frame: np.ndarray, position: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize the detection on the frame.

        Args:
            frame: Input grayscale frame
            position: Detected position (if any)

        Returns:
            Colored frame with detection visualization
        """
        # Convert to BGR for visualization
        if len(frame.shape) == 2:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis_frame = frame.copy()

        # Draw center crosshair
        center_int = int(self.center)
        cv2.line(
            vis_frame,
            (center_int - 5, center_int),
            (center_int + 5, center_int),
            (0, 255, 0),
            1,
        )
        cv2.line(
            vis_frame,
            (center_int, center_int - 5),
            (center_int, center_int + 5),
            (0, 255, 0),
            1,
        )

        # Draw detected position
        if position is not None:
            detected_x = int(position[0] + self.center)
            detected_y = int(position[1] + self.center)
            cv2.circle(vis_frame, (detected_x, detected_y), 5, (0, 0, 255), 2)
            cv2.line(
                vis_frame,
                (center_int, center_int),
                (detected_x, detected_y),
                (255, 0, 0),
                1,
            )

        return vis_frame


def extract_position_from_observation(observation: np.ndarray) -> Optional[np.ndarray]:
    """
    Convenience function to extract target position from observation.

    Args:
        observation: Either a single frame (H, W) or stacked frames

    Returns:
        Target position relative to center, or None if not detected
    """
    detector = VisualDetector()

    # Check if observation is stacked frames or single frame
    if len(observation.shape) == 3:
        position, _ = detector.detect_target_stacked(observation)
    else:
        position, _ = detector.detect_target(observation)

    return position
