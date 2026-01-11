"""
Velocity Tracker module for tracking player velocities.

Tracks player positions frame-to-frame and calculates smoothed velocities.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional


class PlayerVelocityTracker:
    """
    Track player velocities from frame-to-frame position changes.

    Uses exponential moving average for smoothing and provides:
    - Velocity estimates in cm/s
    - Run direction for through-pass calculations
    """

    def __init__(
        self,
        smoothing_factor: float = 0.3,
        max_history: int = 10,
        fps: float = 30.0,
        min_speed_threshold: float = 50.0  # cm/s below which player is "stationary"
    ):
        """
        Initialize velocity tracker.

        Args:
            smoothing_factor: EMA smoothing factor (0-1, higher = more responsive)
            max_history: Maximum number of frames to keep in history
            fps: Video frame rate for velocity calculation
            min_speed_threshold: Speed below which player is considered stationary (cm/s)
        """
        self.smoothing_factor = smoothing_factor
        self.max_history = max_history
        self.fps = fps
        self.min_speed_threshold = min_speed_threshold

        # Position history per tracker_id: deque of (position, frame_num)
        self.positions: Dict[int, deque] = {}
        # Smoothed velocities per tracker_id in cm/s
        self.velocities: Dict[int, np.ndarray] = {}
        # Last update frame per tracker_id
        self.last_frame: Dict[int, int] = {}

    def update(self, tracker_id: int, position: np.ndarray, frame_num: int) -> None:
        """
        Update position history and recalculate velocity for a player.

        Args:
            tracker_id: Player's tracker ID
            position: Current position in cm (2,)
            frame_num: Current frame number
        """
        if tracker_id not in self.positions:
            self.positions[tracker_id] = deque(maxlen=self.max_history)
            self.velocities[tracker_id] = np.array([0.0, 0.0])

        self.positions[tracker_id].append((position.copy(), frame_num))
        self.last_frame[tracker_id] = frame_num

        # Calculate velocity from last two positions
        if len(self.positions[tracker_id]) >= 2:
            pos_curr, frame_curr = self.positions[tracker_id][-1]
            pos_prev, frame_prev = self.positions[tracker_id][-2]

            frame_diff = frame_curr - frame_prev
            if frame_diff > 0:
                # Velocity in cm/frame, then convert to cm/s
                vel_per_frame = (pos_curr - pos_prev) / frame_diff
                vel_per_sec = vel_per_frame * self.fps

                # Apply exponential moving average
                self.velocities[tracker_id] = (
                    self.smoothing_factor * vel_per_sec +
                    (1 - self.smoothing_factor) * self.velocities[tracker_id]
                )

    def get_velocity(self, tracker_id: int) -> Optional[np.ndarray]:
        """
        Get smoothed velocity estimate for a player.

        Args:
            tracker_id: Player's tracker ID

        Returns:
            Velocity in cm/s (2,) or None if not tracked
        """
        return self.velocities.get(tracker_id)

    def get_speed(self, tracker_id: int) -> float:
        """
        Get speed (velocity magnitude) for a player.

        Args:
            tracker_id: Player's tracker ID

        Returns:
            Speed in cm/s
        """
        vel = self.velocities.get(tracker_id)
        if vel is None:
            return 0.0
        return float(np.linalg.norm(vel))

    def get_run_direction(
        self,
        tracker_id: int,
        attack_direction: np.ndarray = None,
        default_direction: np.ndarray = None
    ) -> np.ndarray:
        """
        Get estimated run direction for a player.

        Priority:
        1. Use actual velocity if above threshold
        2. Use attack_direction if provided (toward goal)
        3. Use default_direction if provided
        4. Fall back to (1, 0)

        Args:
            tracker_id: Player's tracker ID
            attack_direction: Direction toward attacking goal (unit vector)
            default_direction: Default direction if player is stationary

        Returns:
            Unit vector (2,) representing run direction
        """
        vel = self.velocities.get(tracker_id)

        if vel is not None:
            speed = np.linalg.norm(vel)
            if speed > self.min_speed_threshold:
                return vel / speed  # Normalize to unit vector

        # Player is stationary or not tracked
        if attack_direction is not None:
            norm = np.linalg.norm(attack_direction)
            if norm > 0:
                return attack_direction / norm

        if default_direction is not None:
            norm = np.linalg.norm(default_direction)
            if norm > 0:
                return default_direction / norm

        return np.array([1.0, 0.0])  # Default: toward right

    def get_velocities_for_ids(self, tracker_ids: np.ndarray) -> np.ndarray:
        """
        Get velocities for multiple tracker IDs.

        Args:
            tracker_ids: Array of tracker IDs

        Returns:
            Array of velocities (N, 2) in cm/s, zeros for unknown IDs
        """
        velocities = np.zeros((len(tracker_ids), 2), dtype=np.float32)
        for i, tid in enumerate(tracker_ids):
            vel = self.velocities.get(int(tid))
            if vel is not None:
                velocities[i] = vel
        return velocities

    def cleanup_old_tracks(self, current_frame: int, max_age: int = 90) -> None:
        """
        Remove tracks that haven't been updated recently.

        Args:
            current_frame: Current frame number
            max_age: Maximum frames since last update before removal
        """
        to_remove = []
        for tid, last_frame in self.last_frame.items():
            if current_frame - last_frame > max_age:
                to_remove.append(tid)

        for tid in to_remove:
            del self.positions[tid]
            del self.velocities[tid]
            del self.last_frame[tid]
