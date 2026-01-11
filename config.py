"""
Configuration module for football video inference.

Contains pitch configuration, model paths, colors, and default thresholds.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict


# =============================================================================
# Project Paths (all paths relative to this file's directory)
# =============================================================================

_SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = str(_SCRIPT_DIR)  # final/ folder is the project root

INPUT_VIDEO_DIR = os.path.join(PROJECT_DIR, 'input')
OUTPUT_VIDEO_DIR = os.path.join(PROJECT_DIR, 'output')

# Model paths (all in models/ subdirectory)
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
BALL_MODEL_PATH = os.path.join(MODELS_DIR, 'ball_detect.pt')
PLAYER_MODEL_PATH = os.path.join(MODELS_DIR, 'player_detect.pt')
PITCH_MODEL_PATH = os.path.join(MODELS_DIR, 'pitch_detect.pt')

# Re-ID model path
REID_MODEL_PATH = os.path.join(MODELS_DIR, 'osnet_x0_25_msmt17.pt')

# Pass success model path
PASS_SUCCESS_MODEL_PATH = os.path.join(MODELS_DIR, 'pass_success_model.pkl')

# Data paths
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
XT_GRID_PATH = os.path.join(DATA_DIR, 'xt_grid.npy')


# =============================================================================
# Class IDs from player model
# =============================================================================

PLAYER_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
REFEREE_CLASS_ID = 2


# =============================================================================
# Team Colors
# =============================================================================

TEAM_COLORS: Dict[int, str] = {
    0: '#FF1493',  # Team 0: Deep Pink
    1: '#00BFFF',  # Team 1: Deep Sky Blue
}

TEAM_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (147, 20, 255),   # Deep Pink BGR
    1: (255, 191, 0),    # Deep Sky Blue BGR
}

GOALKEEPER_COLORS: Dict[int, str] = {
    0: '#FF69B4',  # Team 0 goalkeeper: Hot Pink
    1: '#87CEEB',  # Team 1 goalkeeper: Sky Blue
}

REFEREE_COLOR = '#FFD700'  # Gold for referees
BALL_COLOR_HEX = '#00FF00'  # Green for ball


# =============================================================================
# Default Confidence Thresholds
# =============================================================================

DEFAULT_CONF_THRESHOLDS: Dict[str, float] = {
    'ball': 0.07,
    'player': 0.25,
    'goalkeeper': 0.25,
    'referee': 0.25,
    'pitch': 0.5,
}


# =============================================================================
# Soccer Pitch Configuration
# =============================================================================

@dataclass
class SoccerPitchConfiguration:
    """Soccer pitch configuration with dimensions in centimeters."""
    width: int = 7000  # [cm]
    length: int = 12000  # [cm]
    penalty_box_width: int = 4100  # [cm]
    penalty_box_length: int = 2015  # [cm]
    goal_box_width: int = 1832  # [cm]
    goal_box_length: int = 550  # [cm]
    centre_circle_radius: int = 915  # [cm]
    penalty_spot_distance: int = 1100  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """Get the 32 vertices of the pitch."""
        return [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])


# Global pitch configuration instance
PITCH_CONFIG = SoccerPitchConfiguration()
