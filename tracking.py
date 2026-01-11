"""
Tracking module for player ID management.

Contains FixedIDTrackManager and TeamTrackManager classes.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Set

import supervision as sv

from config import REID_MODEL_PATH

# Check if boxmot is available
try:
    from boxmot import BotSort
    BOTSORT_AVAILABLE = True
except ImportError:
    BOTSORT_AVAILABLE = False
    BotSort = None


class FixedIDTrackManager:
    """Manages a fixed pool of track IDs (1-N) for football players."""

    def __init__(self, max_players: int = 22, max_lost_frames: int = 150):
        self.max_players = max_players
        self.max_lost_frames = max_lost_frames
        self.id_mapping: Dict[int, int] = {}
        self.available_ids: Set[int] = set(range(1, max_players + 1))
        self.last_positions: Dict[int, tuple] = {}
        self.active_ids: Set[int] = set()
        self.id_offset: int = 0  # For team-specific offsets

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _find_closest_lost_id(self, center):
        if not self.available_ids:
            return None

        best_id = None
        best_dist = float('inf')

        for fixed_id in self.available_ids:
            if fixed_id in self.last_positions:
                last_pos, frames_lost = self.last_positions[fixed_id]
                dist = self._distance(center, last_pos) * (1 + frames_lost / self.max_lost_frames)
                if dist < best_dist:
                    best_dist = dist
                    best_id = fixed_id

        if best_id is None and self.available_ids:
            best_id = min(self.available_ids)

        return best_id

    def update(self, tracks: np.ndarray) -> np.ndarray:
        if len(tracks) == 0:
            self._update_lost_frames()
            return tracks

        self.active_ids = set()

        for i in range(len(tracks)):
            botsort_id = int(tracks[i, 4])
            bbox = tracks[i, 0:4]
            center = self._get_center(bbox)

            if botsort_id in self.id_mapping:
                fixed_id = self.id_mapping[botsort_id]
            else:
                fixed_id = self._find_closest_lost_id(center)

                if fixed_id is not None:
                    self.id_mapping[botsort_id] = fixed_id
                    self.available_ids.discard(fixed_id)
                else:
                    fixed_id = (botsort_id % self.max_players) + 1
                    self.id_mapping[botsort_id] = fixed_id

            tracks[i, 4] = fixed_id
            self.active_ids.add(fixed_id)
            self.last_positions[fixed_id] = (center, 0)

        self._update_lost_frames()

        return tracks

    def _update_lost_frames(self):
        ids_to_remove = []

        for fixed_id, (pos, frames_lost) in list(self.last_positions.items()):
            if fixed_id not in self.active_ids:
                frames_lost += 1

                if frames_lost > self.max_lost_frames:
                    ids_to_remove.append(fixed_id)
                    self.available_ids.add(fixed_id)
                    self.id_mapping = {k: v for k, v in self.id_mapping.items() if v != fixed_id}
                else:
                    self.last_positions[fixed_id] = (pos, frames_lost)

        for fixed_id in ids_to_remove:
            del self.last_positions[fixed_id]

    def get_stats(self) -> dict:
        return {
            'active_ids': len(self.active_ids),
            'available_ids': len(self.available_ids),
            'total_mappings': len(self.id_mapping),
        }


class TeamTrackManager:
    """Manages separate BoT-SORT trackers for each team."""

    def __init__(
        self,
        max_players_per_team: int = 11,
        max_lost_frames: int = 150,
        device: str = 'cuda:0',
        cmc_method: str = 'ecc',
        with_reid: bool = True,
        fps: int = 30,
    ):
        self.max_players_per_team = max_players_per_team
        self.max_lost_frames = max_lost_frames
        self.device = device
        self.cmc_method = cmc_method
        self.with_reid = with_reid
        self.fps = fps

        self.team_trackers: Dict[int, object] = {}
        self.team_id_managers: Dict[int, FixedIDTrackManager] = {}

        for team_id in [0, 1]:
            self._init_team_tracker(team_id)

        self._init_referee_tracker()

    def _init_team_tracker(self, team_id: int):
        if not BOTSORT_AVAILABLE:
            return

        torch_device = torch.device(self.device)

        self.team_trackers[team_id] = BotSort(
            reid_weights=Path(REID_MODEL_PATH),
            device=torch_device,
            half=torch.cuda.is_available(),
            track_high_thresh=0.2,
            track_low_thresh=0.05,
            new_track_thresh=0.2,
            track_buffer=self.fps * 3,
            match_thresh=0.7,
            cmc_method=self.cmc_method,
            proximity_thresh=0.6,
            appearance_thresh=0.2,
            with_reid=self.with_reid,
        )

        id_offset = team_id * self.max_players_per_team
        self.team_id_managers[team_id] = FixedIDTrackManager(
            max_players=self.max_players_per_team,
            max_lost_frames=self.max_lost_frames,
        )
        self.team_id_managers[team_id].id_offset = id_offset

    def _init_referee_tracker(self):
        if not BOTSORT_AVAILABLE:
            return

        torch_device = torch.device(self.device)

        self.referee_tracker = BotSort(
            reid_weights=Path(REID_MODEL_PATH),
            device=torch_device,
            half=torch.cuda.is_available(),
            track_high_thresh=0.2,
            track_low_thresh=0.05,
            new_track_thresh=0.2,
            track_buffer=self.fps * 3,
            match_thresh=0.7,
            cmc_method=self.cmc_method,
            proximity_thresh=0.6,
            appearance_thresh=0.2,
            with_reid=self.with_reid,
        )

        self.referee_id_manager = FixedIDTrackManager(
            max_players=4,
            max_lost_frames=self.max_lost_frames,
        )
        self.referee_id_manager.id_offset = self.max_players_per_team * 2

    def update_team(self, team_id: int, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        if len(detections) == 0 or team_id not in self.team_trackers:
            return sv.Detections.empty()

        dets = detections_to_botsort_format(detections, frame.shape)
        tracks = self.team_trackers[team_id].update(dets, frame)

        if len(tracks) > 0:
            tracks = self.team_id_managers[team_id].update(tracks)
            id_offset = team_id * self.max_players_per_team
            tracks[:, 4] = tracks[:, 4] + id_offset

        return botsort_output_to_detections(tracks, detections)

    def update_referees(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        if len(detections) == 0:
            return sv.Detections.empty()

        dets = detections_to_botsort_format(detections, frame.shape)
        tracks = self.referee_tracker.update(dets, frame)

        if len(tracks) > 0:
            tracks = self.referee_id_manager.update(tracks)
            tracks[:, 4] = tracks[:, 4] + self.referee_id_manager.id_offset

        return botsort_output_to_detections(tracks, detections)

    def get_stats(self) -> dict:
        stats = {}
        for team_id in [0, 1]:
            if team_id in self.team_id_managers:
                stats[f'team_{team_id}'] = self.team_id_managers[team_id].get_stats()
        return stats


def detections_to_botsort_format(detections: sv.Detections, frame_shape: tuple) -> np.ndarray:
    """Convert supervision Detections to BoT-SORT input format."""
    if len(detections) == 0:
        return np.empty((0, 6))

    dets = np.column_stack([
        detections.xyxy,
        detections.confidence,
        detections.class_id,
    ])

    return dets


def botsort_output_to_detections(tracks: np.ndarray, original_detections: sv.Detections) -> sv.Detections:
    """Convert BoT-SORT output back to supervision Detections."""
    if len(tracks) == 0:
        return sv.Detections.empty()

    xyxy = tracks[:, 0:4]
    tracker_ids = tracks[:, 4].astype(int)
    confidences = tracks[:, 5]
    class_ids = tracks[:, 6].astype(int)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=class_ids,
        tracker_id=tracker_ids,
    )

    return detections
