"""
Voronoi module for territorial control visualization.

Contains static and dynamic Voronoi diagram functions.
"""

import cv2
import numpy as np
from typing import Tuple

from config import SoccerPitchConfiguration, PITCH_CONFIG
from view_transformer import ViewTransformer


def create_voronoi_world_image(
    team_0_world: np.ndarray,
    team_1_world: np.ndarray,
    config: SoccerPitchConfiguration = None,
    team_0_color: Tuple[int, int, int] = (147, 20, 255),   # Deep Pink BGR
    team_1_color: Tuple[int, int, int] = (255, 191, 0),    # Sky Blue BGR
    scale: float = 0.1,
) -> np.ndarray:
    """
    Create a Voronoi diagram image in world coordinate space.

    Args:
        team_0_world: World coordinates of team 0 players (N, 2) in cm.
        team_1_world: World coordinates of team 1 players (M, 2) in cm.
        config: Soccer pitch configuration.
        team_0_color: BGR color tuple for team 0's territory.
        team_1_color: BGR color tuple for team 1's territory.
        scale: Pixels per centimeter.

    Returns:
        BGRA image (H, W, 4) with Voronoi regions and alpha channel.
    """
    if config is None:
        config = PITCH_CONFIG

    img_width = int(config.length * scale)
    img_height = int(config.width * scale)

    voronoi_img = np.zeros((img_height, img_width, 4), dtype=np.uint8)

    if len(team_0_world) == 0 or len(team_1_world) == 0:
        return voronoi_img

    py_coords, px_coords = np.indices((img_height, img_width))
    world_x = px_coords / scale
    world_y = py_coords / scale

    def calculate_min_distance(positions: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
        if len(positions) == 0:
            return np.full_like(wx, np.inf, dtype=np.float32)
        px = positions[:, 0][:, None, None]
        py = positions[:, 1][:, None, None]
        distances = np.sqrt((px - wx) ** 2 + (py - wy) ** 2)
        return np.min(distances, axis=0)

    min_dist_0 = calculate_min_distance(team_0_world, world_x, world_y)
    min_dist_1 = calculate_min_distance(team_1_world, world_x, world_y)

    team_0_control = min_dist_0 < min_dist_1

    voronoi_img[team_0_control, :3] = team_0_color
    voronoi_img[~team_0_control, :3] = team_1_color
    voronoi_img[:, :, 3] = 255

    return voronoi_img


def draw_voronoi_on_frame(
    frame: np.ndarray,
    view_transformer: ViewTransformer,
    team_0_world: np.ndarray,
    team_1_world: np.ndarray,
    config: SoccerPitchConfiguration = None,
    team_0_color: Tuple[int, int, int] = (147, 20, 255),
    team_1_color: Tuple[int, int, int] = (255, 191, 0),
    opacity: float = 0.3,
    voronoi_scale: float = 0.1,
) -> np.ndarray:
    """
    Draw Voronoi diagram overlay directly on the video frame.

    Args:
        frame: Original video frame (BGR, H x W x 3).
        view_transformer: ViewTransformer with valid homography.
        team_0_world: World coordinates of team 0 players (N, 2).
        team_1_world: World coordinates of team 1 players (M, 2).
        config: Soccer pitch configuration.
        team_0_color: BGR color for team 0's territory.
        team_1_color: BGR color for team 1's territory.
        opacity: Opacity of the Voronoi overlay (0.0 to 1.0).
        voronoi_scale: Scale for world-space Voronoi image.

    Returns:
        Frame with Voronoi overlay blended in.
    """
    if config is None:
        config = PITCH_CONFIG

    if len(team_0_world) == 0 or len(team_1_world) == 0:
        return frame

    voronoi_world = create_voronoi_world_image(
        team_0_world=team_0_world,
        team_1_world=team_1_world,
        config=config,
        team_0_color=team_0_color,
        team_1_color=team_1_color,
        scale=voronoi_scale,
    )

    frame_h, frame_w = frame.shape[:2]

    scale_matrix = np.array([
        [1.0 / voronoi_scale, 0, 0],
        [0, 1.0 / voronoi_scale, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    try:
        H_inv = view_transformer.inverse_matrix

        if np.any(np.isnan(H_inv)) or np.any(np.isinf(H_inv)):
            return frame

        cond = np.linalg.cond(view_transformer.m)
        if cond > 1e10:
            return frame

        M_combined = H_inv @ scale_matrix

    except (np.linalg.LinAlgError, ValueError):
        return frame

    voronoi_warped = cv2.warpPerspective(
        src=voronoi_world,
        M=M_combined,
        dsize=(frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    result = frame.copy()
    alpha = voronoi_warped[:, :, 3].astype(np.float32) / 255.0 * opacity
    alpha_3ch = alpha[:, :, np.newaxis]

    voronoi_bgr = voronoi_warped[:, :, :3].astype(np.float32)
    frame_float = frame.astype(np.float32)

    blended = alpha_3ch * voronoi_bgr + (1 - alpha_3ch) * frame_float
    result = blended.astype(np.uint8)

    return result


def create_dynamic_voronoi_world_image(
    team_0_positions: np.ndarray,
    team_0_velocities: np.ndarray,
    team_1_positions: np.ndarray,
    team_1_velocities: np.ndarray,
    config: SoccerPitchConfiguration = None,
    team_0_color: Tuple[int, int, int] = (147, 20, 255),
    team_1_color: Tuple[int, int, int] = (255, 191, 0),
    scale: float = 0.1,
    max_speed: float = 700.0,
    reaction_time: float = 0.0
) -> np.ndarray:
    """
    Create a dynamic Voronoi diagram based on who can reach each point FIRST.

    Args:
        team_0_positions: World coordinates of team 0 players (N, 2) in cm.
        team_0_velocities: Velocities of team 0 players (N, 2) in cm/s.
        team_1_positions: World coordinates of team 1 players (M, 2) in cm.
        team_1_velocities: Velocities of team 1 players (M, 2) in cm/s.
        config: Soccer pitch configuration.
        team_0_color: BGR color tuple for team 0's territory.
        team_1_color: BGR color tuple for team 1's territory.
        scale: Pixels per centimeter.
        max_speed: Maximum player sprint speed in cm/s.
        reaction_time: Reaction time before players can start moving.

    Returns:
        BGRA image (H, W, 4) with Voronoi regions and alpha channel.
    """
    if config is None:
        config = PITCH_CONFIG

    img_width = int(config.length * scale)
    img_height = int(config.width * scale)

    voronoi_img = np.zeros((img_height, img_width, 4), dtype=np.uint8)

    if len(team_0_positions) == 0 or len(team_1_positions) == 0:
        return voronoi_img

    py_coords, px_coords = np.indices((img_height, img_width))
    world_x = px_coords / scale
    world_y = py_coords / scale

    from dynamic_pitch_control import calculate_min_time_to_reach_team

    min_time_0 = calculate_min_time_to_reach_team(
        team_0_positions, team_0_velocities, world_x, world_y,
        max_speed, reaction_time
    )
    min_time_1 = calculate_min_time_to_reach_team(
        team_1_positions, team_1_velocities, world_x, world_y,
        max_speed, reaction_time
    )

    team_0_control = min_time_0 < min_time_1

    voronoi_img[team_0_control, :3] = team_0_color
    voronoi_img[~team_0_control, :3] = team_1_color
    voronoi_img[:, :, 3] = 255

    return voronoi_img


def draw_dynamic_voronoi_on_frame(
    frame: np.ndarray,
    view_transformer: ViewTransformer,
    team_0_positions: np.ndarray,
    team_0_velocities: np.ndarray,
    team_1_positions: np.ndarray,
    team_1_velocities: np.ndarray,
    config: SoccerPitchConfiguration = None,
    team_0_color: Tuple[int, int, int] = (147, 20, 255),
    team_1_color: Tuple[int, int, int] = (255, 191, 0),
    opacity: float = 0.3,
    voronoi_scale: float = 0.1,
    max_speed: float = 700.0
) -> np.ndarray:
    """
    Draw dynamic (velocity-weighted) Voronoi diagram overlay on the video frame.

    Args:
        frame: Original video frame (BGR, H x W x 3).
        view_transformer: ViewTransformer with valid homography.
        team_0_positions: World coordinates of team 0 players (N, 2) in cm.
        team_0_velocities: Velocities of team 0 players (N, 2) in cm/s.
        team_1_positions: World coordinates of team 1 players (M, 2) in cm.
        team_1_velocities: Velocities of team 1 players (M, 2) in cm/s.
        config: Soccer pitch configuration.
        team_0_color: BGR color for team 0's territory.
        team_1_color: BGR color for team 1's territory.
        opacity: Opacity of the Voronoi overlay (0.0 to 1.0).
        voronoi_scale: Scale for world-space Voronoi image.
        max_speed: Maximum player sprint speed in cm/s.

    Returns:
        Frame with dynamic Voronoi overlay blended in.
    """
    if config is None:
        config = PITCH_CONFIG

    if len(team_0_positions) == 0 or len(team_1_positions) == 0:
        return frame

    voronoi_world = create_dynamic_voronoi_world_image(
        team_0_positions=team_0_positions,
        team_0_velocities=team_0_velocities,
        team_1_positions=team_1_positions,
        team_1_velocities=team_1_velocities,
        config=config,
        team_0_color=team_0_color,
        team_1_color=team_1_color,
        scale=voronoi_scale,
        max_speed=max_speed,
    )

    frame_h, frame_w = frame.shape[:2]

    scale_matrix = np.array([
        [1.0 / voronoi_scale, 0, 0],
        [0, 1.0 / voronoi_scale, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    try:
        H_inv = view_transformer.inverse_matrix

        if np.any(np.isnan(H_inv)) or np.any(np.isinf(H_inv)):
            return frame

        cond = np.linalg.cond(view_transformer.m)
        if cond > 1e10:
            return frame

        M_combined = H_inv @ scale_matrix

    except (np.linalg.LinAlgError, ValueError):
        return frame

    voronoi_warped = cv2.warpPerspective(
        src=voronoi_world,
        M=M_combined,
        dsize=(frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    result = frame.copy()
    alpha = voronoi_warped[:, :, 3].astype(np.float32) / 255.0 * opacity
    alpha_3ch = alpha[:, :, np.newaxis]

    voronoi_bgr = voronoi_warped[:, :, :3].astype(np.float32)
    frame_float = frame.astype(np.float32)

    blended = alpha_3ch * voronoi_bgr + (1 - alpha_3ch) * frame_float
    result = blended.astype(np.uint8)

    return result
