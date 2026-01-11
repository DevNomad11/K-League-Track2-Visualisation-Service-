"""
Pitch Drawing module for creating soccer pitch visualizations.

Contains functions for drawing pitch diagrams and overlays.
"""

import cv2
import numpy as np
import supervision as sv
from typing import Optional

from config import SoccerPitchConfiguration, PITCH_CONFIG


def draw_pitch(
    config: SoccerPitchConfiguration = None,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draw a soccer pitch image.

    Args:
        config: Pitch configuration with dimensions.
        background_color: Color of the pitch background.
        line_color: Color of the pitch lines.
        padding: Padding around the pitch in pixels.
        line_thickness: Thickness of the pitch lines.
        point_radius: Radius of penalty spot points.
        scale: Scaling factor for pitch dimensions.

    Returns:
        Image of the soccer pitch.
    """
    if config is None:
        config = PITCH_CONFIG

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    # Draw center circle
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    # Draw penalty spots
    penalty_spots = [
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    return pitch_image


def draw_points_on_pitch(
    xy: np.ndarray,
    config: SoccerPitchConfiguration = None,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draw points on a soccer pitch.

    Args:
        xy: Array of (x, y) world coordinates to draw.
        config: Pitch configuration.
        face_color: Color of point faces.
        edge_color: Color of point edges.
        radius: Radius of points in pixels.
        thickness: Thickness of point edges.
        padding: Padding around the pitch.
        scale: Scaling factor.
        pitch: Existing pitch image (creates new one if None).

    Returns:
        Pitch image with points drawn.
    """
    if config is None:
        config = PITCH_CONFIG

    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch


def draw_pitch_voronoi_diagram(
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    config: SoccerPitchConfiguration = None,
    team_1_color: sv.Color = sv.Color.from_hex('#FF1493'),
    team_2_color: sv.Color = sv.Color.from_hex('#00BFFF'),
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draw a Voronoi diagram on the pitch showing team territorial control.

    Args:
        team_1_xy: World coordinates of team 1 players.
        team_2_xy: World coordinates of team 2 players.
        config: Pitch configuration.
        team_1_color: Color for team 1's territory.
        team_2_color: Color for team 2's territory.
        opacity: Opacity of the Voronoi overlay.
        padding: Padding around the pitch.
        scale: Scaling factor.
        pitch: Existing pitch image.

    Returns:
        Pitch image with Voronoi diagram overlay.
    """
    if config is None:
        config = PITCH_CONFIG

    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    if len(team_1_xy) == 0 or len(team_2_xy) == 0:
        return pitch

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay
