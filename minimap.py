"""
Minimap module for creating 2D pitch representations.

Contains functions for creating and drawing on minimaps.
"""

import cv2
import numpy as np
from typing import Tuple

from config import PITCH_CONFIG
from pitch_drawing import draw_pitch


def create_minimap(width: int = 350, height: int = 230) -> np.ndarray:
    """
    Create a blank 2D pitch minimap.

    Args:
        width: Minimap width in pixels.
        height: Minimap height in pixels.

    Returns:
        Minimap image.
    """
    scale = min(
        (width - 20) / PITCH_CONFIG.length,
        (height - 20) / PITCH_CONFIG.width
    )
    padding = 10

    minimap = draw_pitch(
        config=PITCH_CONFIG,
        padding=padding,
        line_thickness=2,
        point_radius=4,
        scale=scale
    )

    # Resize to exact dimensions if needed
    if minimap.shape[0] != height or minimap.shape[1] != width:
        minimap = cv2.resize(minimap, (width, height))

    return minimap


def draw_players_on_minimap(
    minimap: np.ndarray,
    player_world_positions: np.ndarray,
    team_ids: np.ndarray,
    team_0_color: Tuple[int, int, int] = (147, 20, 255),   # Deep Pink BGR
    team_1_color: Tuple[int, int, int] = (255, 191, 0),    # Deep Sky Blue BGR
) -> np.ndarray:
    """
    Draw player positions on the minimap.

    Args:
        minimap: The minimap image.
        player_world_positions: World coordinates of players (N, 2).
        team_ids: Team ID for each player (N,).
        team_0_color: BGR color for team 0.
        team_1_color: BGR color for team 1.

    Returns:
        Minimap with players drawn.
    """
    if len(player_world_positions) == 0:
        return minimap

    result = minimap.copy()
    height, width = result.shape[:2]

    scale = min(
        (width - 20) / PITCH_CONFIG.length,
        (height - 20) / PITCH_CONFIG.width
    )
    padding = 10

    team_colors = {0: team_0_color, 1: team_1_color}

    for world_pt, team_id in zip(player_world_positions, team_ids):
        x = int(world_pt[0] * scale) + padding
        y = int(world_pt[1] * scale) + padding

        if 0 <= x < width and 0 <= y < height:
            color = team_colors.get(team_id, (255, 255, 255))
            cv2.circle(result, (x, y), 4, color, -1)
            cv2.circle(result, (x, y), 4, (0, 0, 0), 1)

    return result


def draw_ball_on_minimap(
    minimap: np.ndarray,
    ball_world_position: np.ndarray,
) -> np.ndarray:
    """
    Draw ball position on the minimap.

    Args:
        minimap: The minimap image.
        ball_world_position: World coordinates of the ball (2,).

    Returns:
        Minimap with ball drawn.
    """
    if ball_world_position is None:
        return minimap

    result = minimap.copy()
    height, width = result.shape[:2]

    scale = min(
        (width - 20) / PITCH_CONFIG.length,
        (height - 20) / PITCH_CONFIG.width
    )
    padding = 10

    x = int(ball_world_position[0] * scale) + padding
    y = int(ball_world_position[1] * scale) + padding

    if 0 <= x < width and 0 <= y < height:
        cv2.circle(result, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(result, (x, y), 5, (0, 0, 0), 2)

    return result


def draw_voronoi_on_minimap(
    minimap: np.ndarray,
    team_0_world: np.ndarray,
    team_1_world: np.ndarray,
    opacity: float = 0.3,
) -> np.ndarray:
    """
    Draw Voronoi diagram on the minimap.

    Args:
        minimap: The minimap image.
        team_0_world: World coordinates of team 0 players.
        team_1_world: World coordinates of team 1 players.
        opacity: Opacity of the Voronoi overlay.

    Returns:
        Minimap with Voronoi diagram.
    """
    if len(team_0_world) == 0 or len(team_1_world) == 0:
        return minimap

    height, width = minimap.shape[:2]
    scale = min(
        (width - 20) / PITCH_CONFIG.length,
        (height - 20) / PITCH_CONFIG.width
    )

    voronoi = np.zeros_like(minimap, dtype=np.uint8)

    team_0_color = np.array([147, 20, 255], dtype=np.uint8)  # Deep Pink BGR
    team_1_color = np.array([255, 191, 0], dtype=np.uint8)   # Sky Blue BGR

    y_coords, x_coords = np.indices((height, width))

    padding = 10
    world_x = (x_coords - padding) / scale
    world_y = (y_coords - padding) / scale

    def calculate_min_distance(positions, wx, wy):
        if len(positions) == 0:
            return np.full_like(wx, np.inf, dtype=np.float32)
        px = positions[:, 0][:, None, None]
        py = positions[:, 1][:, None, None]
        distances = np.sqrt((px - wx) ** 2 + (py - wy) ** 2)
        return np.min(distances, axis=0)

    min_dist_0 = calculate_min_distance(team_0_world, world_x, world_y)
    min_dist_1 = calculate_min_distance(team_1_world, world_x, world_y)

    control_mask = min_dist_0 < min_dist_1

    voronoi[control_mask] = team_0_color
    voronoi[~control_mask] = team_1_color

    result = cv2.addWeighted(minimap, 1 - opacity, voronoi, opacity, 0)
    return result


def overlay_minimap(
    frame: np.ndarray,
    minimap: np.ndarray,
    position: str = 'bottom_right',
    margin: int = 20,
    alpha: float = 0.85
) -> np.ndarray:
    """
    Overlay minimap on the frame.

    Args:
        frame: Video frame.
        minimap: Minimap image.
        position: Position on frame ('bottom_right', 'bottom_left', 'top_right', 'top_left').
        margin: Margin from edges in pixels.
        alpha: Opacity of the minimap.

    Returns:
        Frame with minimap overlay.
    """
    result = frame.copy()
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]

    if position == 'bottom_right':
        x = fw - mw - margin
        y = fh - mh - margin
    elif position == 'bottom_left':
        x = margin
        y = fh - mh - margin
    elif position == 'top_right':
        x = fw - mw - margin
        y = margin
    else:  # top_left
        x = margin
        y = margin

    roi = result[y:y+mh, x:x+mw]
    blended = cv2.addWeighted(roi, 1 - alpha, minimap, alpha, 0)
    result[y:y+mh, x:x+mw] = blended

    cv2.rectangle(result, (x, y), (x + mw, y + mh), (255, 255, 255), 2)

    return result
