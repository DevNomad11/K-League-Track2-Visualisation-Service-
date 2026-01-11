"""
View Transformer module for coordinate transformations.

Handles homography-based transformations between image and world coordinates.
"""

import cv2
import numpy as np
import supervision as sv
from typing import Optional

from config import PITCH_CONFIG


class ViewTransformer:
    """
    Transform points between image coordinates and world coordinates using homography.
    This is the simpler approach used by the sports package.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Initialize the ViewTransformer with source and target points.

        Args:
            source: Source points (image coordinates) for homography calculation.
            target: Target points (world coordinates) for homography calculation.

        Raises:
            ValueError: If homography matrix could not be calculated.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    @property
    def inverse_matrix(self) -> np.ndarray:
        """
        Get the inverse homography matrix for world-to-image transformation.

        Returns:
            Inverse homography matrix (3x3).
        """
        if not hasattr(self, '_m_inv'):
            self._m_inv = np.linalg.inv(self.m)
        return self._m_inv

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points using the homography matrix.

        Args:
            points: Points to be transformed (image coordinates).

        Returns:
            Transformed points (world coordinates).
        """
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_points_inverse(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from world coordinates to image coordinates.

        Args:
            points: Points in world coordinates (N, 2).

        Returns:
            Transformed points in image coordinates (N, 2).
        """
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.inverse_matrix)
        return transformed_points.reshape(-1, 2).astype(np.float32)


def create_view_transformer(keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
    """
    Create a ViewTransformer from detected keypoints using the sports package approach.

    This is the simpler field detection method - it filters keypoints by checking
    if coordinates are > 1 (visible), then creates a homography directly.

    Args:
        keypoints: Keypoints detected by the pitch model.

    Returns:
        ViewTransformer instance or None if not enough keypoints.
    """
    if keypoints is None or len(keypoints.xy) == 0:
        return None

    # Simple visibility mask: keypoints with x > 1 and y > 1 are considered visible
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

    visible_keypoints = keypoints.xy[0][mask].astype(np.float32)
    visible_vertices = np.array(PITCH_CONFIG.vertices)[mask].astype(np.float32)

    # Need at least 4 points for homography
    if len(visible_keypoints) < 4:
        return None

    try:
        transformer = ViewTransformer(
            source=visible_keypoints,
            target=visible_vertices
        )
        return transformer
    except ValueError:
        return None
