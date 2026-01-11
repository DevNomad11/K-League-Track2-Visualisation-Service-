"""
Pass Visualization module for drawing pass analysis overlays.

Contains functions for pass success, xT, and through-pass visualization.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

from view_transformer import ViewTransformer
from dynamic_pitch_control import PassOption


def find_ball_possessor(
    ball_world_pos: np.ndarray,
    player_world_positions: np.ndarray,
    player_team_ids: np.ndarray,
    player_tracker_ids: np.ndarray,
    threshold_cm: float = 300.0
) -> Optional[dict]:
    """
    Find the player in possession of the ball.

    Args:
        ball_world_pos: (2,) ball position in cm
        player_world_positions: (N, 2) array of player positions in cm
        player_team_ids: (N,) array of team IDs (0 or 1)
        player_tracker_ids: (N,) array of tracker IDs
        threshold_cm: Max distance in cm to consider possession (default 300cm = 3m)

    Returns:
        dict with possessor info or None if no possession
    """
    if len(player_world_positions) == 0 or ball_world_pos is None:
        return None

    distances = np.linalg.norm(player_world_positions - ball_world_pos, axis=1)
    closest_idx = np.argmin(distances)

    if distances[closest_idx] <= threshold_cm:
        return {
            'index': closest_idx,
            'team_id': int(player_team_ids[closest_idx]),
            'tracker_id': int(player_tracker_ids[closest_idx]),
            'world_pos': player_world_positions[closest_idx],
            'distance_cm': float(distances[closest_idx])
        }
    return None


def get_success_color(prob: float) -> Tuple[int, int, int]:
    """
    Get BGR color based on pass success probability.
    Green (high) -> Yellow (medium) -> Red (low)
    """
    if prob >= 0.8:
        return (0, 255, 0)
    elif prob >= 0.6:
        ratio = (prob - 0.6) / 0.2
        return (0, 255, int(255 * (1 - ratio)))
    elif prob >= 0.4:
        ratio = (prob - 0.4) / 0.2
        return (0, int(255 * ratio) + int(165 * (1 - ratio)), 255)
    else:
        ratio = prob / 0.4
        return (0, int(165 * ratio), 255)


def get_expected_value_color(ev: float, max_ev: float = 0.15, is_best: bool = False) -> Tuple[int, int, int]:
    """
    Get BGR color based on pass option status.
    Best option: Green
    Other options: White
    """
    if is_best:
        return (0, 255, 0)
    else:
        return (255, 255, 255)


def normalize_coords_for_attack_direction(
    x: np.ndarray,
    y: np.ndarray,
    attack_left_to_right: bool,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize coordinates to L->R attack direction.

    Args:
        x: X coordinates in meters (0-105)
        y: Y coordinates in meters (0-68)
        attack_left_to_right: True if team attacks from left to right
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters

    Returns:
        Normalized (x, y) coordinates
    """
    if attack_left_to_right:
        return x, y
    else:
        return pitch_length - x, pitch_width - y


def draw_pass_success_lines(
    frame: np.ndarray,
    view_transformer: ViewTransformer,
    passer_world_pos: np.ndarray,
    teammate_world_positions: np.ndarray,
    pass_predictor,
    passer_image_pos: np.ndarray = None,
    teammate_image_positions: np.ndarray = None,
    attack_left_to_right: bool = True
) -> np.ndarray:
    """
    Draw pass success rate lines from passer to all teammates.

    Args:
        frame: Video frame to draw on
        view_transformer: For world-to-image coordinate transformation
        passer_world_pos: (2,) passer position in cm
        teammate_world_positions: (N, 2) array of teammate positions in cm
        pass_predictor: PassSuccessPredictor instance
        passer_image_pos: Optional (2,) passer position in pixels
        teammate_image_positions: Optional (N, 2) teammate positions in pixels
        attack_left_to_right: True if possessing team attacks left->right

    Returns:
        Annotated frame
    """
    if len(teammate_world_positions) == 0:
        return frame

    # Convert coordinates: pipeline uses cm, model uses meters
    passer_m_x = passer_world_pos[0] / 100.0
    passer_m_y = passer_world_pos[1] / 100.0
    teammates_m_x = teammate_world_positions[:, 0] / 100.0
    teammates_m_y = teammate_world_positions[:, 1] / 100.0

    # Normalize coordinates
    passer_norm_x, passer_norm_y = normalize_coords_for_attack_direction(
        np.array([passer_m_x]), np.array([passer_m_y]), attack_left_to_right
    )
    teammates_norm_x, teammates_norm_y = normalize_coords_for_attack_direction(
        teammates_m_x, teammates_m_y, attack_left_to_right
    )

    # Predict pass success
    success_probs = pass_predictor.predict_batch(
        np.full(len(teammates_norm_x), passer_norm_x[0]),
        np.full(len(teammates_norm_y), passer_norm_y[0]),
        teammates_norm_x,
        teammates_norm_y
    )

    # Get image coordinates
    if passer_image_pos is None:
        passer_image_pos = view_transformer.transform_points_inverse(
            passer_world_pos.reshape(1, 2)
        )[0]

    if teammate_image_positions is None:
        teammate_image_positions = view_transformer.transform_points_inverse(
            teammate_world_positions
        )

    # Draw lines and labels
    for teammate_img, prob in zip(teammate_image_positions, success_probs):
        color = get_success_color(prob)

        pt1 = (int(passer_image_pos[0]), int(passer_image_pos[1]))
        pt2 = (int(teammate_img[0]), int(teammate_img[1]))

        h, w = frame.shape[:2]
        if not (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
            continue

        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        label = f"{prob*100:.0f}%"

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame,
            (mid_x - 2, mid_y - text_h - 2),
            (mid_x + text_w + 2, mid_y + baseline + 2),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            frame, label, (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        )

    return frame


def draw_pass_xt_lines(
    frame: np.ndarray,
    view_transformer: ViewTransformer,
    passer_world_pos: np.ndarray,
    teammate_world_positions: np.ndarray,
    pass_predictor,
    xt_calculator,
    passer_image_pos: np.ndarray = None,
    teammate_image_positions: np.ndarray = None,
    attack_left_to_right: bool = True,
    highlight_best: bool = True
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Draw pass routes with expected value (pass success x xT) visualization.

    Args:
        frame: Video frame to draw on
        view_transformer: For world-to-image coordinate transformation
        passer_world_pos: (2,) passer position in cm
        teammate_world_positions: (N, 2) array of teammate positions in cm
        pass_predictor: PassSuccessPredictor instance
        xt_calculator: XTCalculator instance with loaded xT grid
        passer_image_pos: Optional (2,) passer position in pixels
        teammate_image_positions: Optional (N, 2) teammate positions in pixels
        attack_left_to_right: True if possessing team attacks left->right
        highlight_best: Whether to highlight the best pass option

    Returns:
        Tuple of (annotated frame, index of best pass option or None)
    """
    if len(teammate_world_positions) == 0:
        return frame, None

    # Convert coordinates
    passer_m_x = passer_world_pos[0] / 100.0
    passer_m_y = passer_world_pos[1] / 100.0
    teammates_m_x = teammate_world_positions[:, 0] / 100.0
    teammates_m_y = teammate_world_positions[:, 1] / 100.0

    # Normalize coordinates
    passer_norm_x, passer_norm_y = normalize_coords_for_attack_direction(
        np.array([passer_m_x]), np.array([passer_m_y]), attack_left_to_right
    )
    teammates_norm_x, teammates_norm_y = normalize_coords_for_attack_direction(
        teammates_m_x, teammates_m_y, attack_left_to_right
    )

    # Predict pass success
    success_probs = pass_predictor.predict_batch(
        np.full(len(teammates_norm_x), passer_norm_x[0]),
        np.full(len(teammates_norm_y), passer_norm_y[0]),
        teammates_norm_x,
        teammates_norm_y
    )

    # Get xT values
    xt_values = xt_calculator.get_xt_batch(teammates_norm_x, teammates_norm_y)

    # Calculate expected values
    expected_values = success_probs * xt_values

    best_idx = np.argmax(expected_values) if highlight_best else None

    # Get image coordinates
    if passer_image_pos is None:
        passer_image_pos = view_transformer.transform_points_inverse(
            passer_world_pos.reshape(1, 2)
        )[0]

    if teammate_image_positions is None:
        teammate_image_positions = view_transformer.transform_points_inverse(
            teammate_world_positions
        )

    max_ev = max(expected_values.max(), 0.05)

    # Draw lines and labels
    for i, (teammate_img, prob, xt, ev) in enumerate(
        zip(teammate_image_positions, success_probs, xt_values, expected_values)
    ):
        is_best = (i == best_idx)
        color = get_expected_value_color(ev, max_ev, is_best)

        pt1 = (int(passer_image_pos[0]), int(passer_image_pos[1]))
        pt2 = (int(teammate_img[0]), int(teammate_img[1]))

        h, w = frame.shape[:2]
        if not (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
            continue

        thickness = 4 if is_best else 2

        if is_best:
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
        else:
            overlay = frame.copy()
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2

        label = f"EV:{ev:.3f}"
        if is_best:
            label = f"BEST {label}"

        font_scale = 0.6 if is_best else 0.5
        font_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        bg_color = (0, 100, 0) if is_best else (50, 50, 50)

        if is_best:
            cv2.rectangle(
                frame,
                (mid_x - 2, mid_y - text_h - 4),
                (mid_x + text_w + 2, mid_y + baseline + 2),
                bg_color,
                -1
            )
            cv2.putText(
                frame, label, (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA
            )

            info_label = f"P:{prob*100:.0f}% xT:{xt:.3f}"
            (info_w, info_h), _ = cv2.getTextSize(
                info_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                frame,
                (mid_x - 2, mid_y + baseline + 4),
                (mid_x + info_w + 2, mid_y + baseline + info_h + 8),
                (0, 100, 0),
                -1
            )
            cv2.putText(
                frame, info_label, (mid_x, mid_y + baseline + info_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )
        else:
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (mid_x - 2, mid_y - text_h - 4),
                (mid_x + text_w + 2, mid_y + baseline + 2),
                bg_color,
                -1
            )
            cv2.putText(
                overlay, label, (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA
            )
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame, best_idx


def draw_target_zone(
    frame: np.ndarray,
    view_transformer: ViewTransformer,
    optimal_point: np.ndarray,
    radius: float = 200.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    opacity: float = 0.3
) -> np.ndarray:
    """
    Draw a highlighted target zone around the optimal receiving point.

    Args:
        frame: Video frame to draw on
        view_transformer: For world-to-image transformation
        optimal_point: Optimal receiving point in cm (2,)
        radius: Zone radius in cm
        color: BGR color tuple
        opacity: Transparency (0-1)

    Returns:
        Frame with target zone drawn
    """
    center_img = view_transformer.transform_points_inverse(
        optimal_point.reshape(1, 2)
    )[0]

    edge_point = optimal_point + np.array([radius, 0])
    edge_img = view_transformer.transform_points_inverse(
        edge_point.reshape(1, 2)
    )[0]

    img_radius = int(np.linalg.norm(edge_img - center_img))
    img_radius = max(img_radius, 5)

    overlay = frame.copy()
    cv2.circle(
        overlay,
        (int(center_img[0]), int(center_img[1])),
        img_radius,
        color,
        -1
    )

    result = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    return result


def draw_pass_xt_lines_v11(
    frame: np.ndarray,
    view_transformer: ViewTransformer,
    passer_world_pos: np.ndarray,
    pass_options: List[PassOption],
    highlight_zones: bool = True,
    top_n_display: int = 3
) -> np.ndarray:
    """
    Draw pass routes to optimal receiving points (v11 style).

    Features:
    - Arrows from ball to optimal receiving point (not current position)
    - Highlighted target zones
    - Best option in solid green, others semi-transparent white

    Args:
        frame: Video frame to draw on
        view_transformer: For world-to-image transformation
        passer_world_pos: Passer's position in cm (2,)
        pass_options: List of PassOption objects (already sorted by EV)
        highlight_zones: Whether to draw target zone circles
        top_n_display: Number of options to display

    Returns:
        Annotated frame
    """
    if len(pass_options) == 0:
        return frame

    result = frame.copy()

    passer_img = view_transformer.transform_points_inverse(
        passer_world_pos.reshape(1, 2)
    )[0]

    display_options = pass_options[:top_n_display]

    # Draw in reverse order so best option is on top
    for rank, option in enumerate(reversed(display_options)):
        is_best = (rank == len(display_options) - 1)

        optimal_img = view_transformer.transform_points_inverse(
            option.optimal_point.reshape(1, 2)
        )[0]

        pt1 = (int(passer_img[0]), int(passer_img[1]))
        pt2 = (int(optimal_img[0]), int(optimal_img[1]))

        h, w = result.shape[:2]
        if not (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
            continue

        # Draw target zone
        if highlight_zones:
            zone_color = (0, 255, 0) if is_best else (255, 255, 255)
            zone_opacity = 0.3 if is_best else 0.15
            result = draw_target_zone(
                result, view_transformer, option.optimal_point,
                radius=200.0, color=zone_color, opacity=zone_opacity
            )

        # Draw line/arrow
        if is_best:
            color = (0, 255, 0)
            thickness = 4
            cv2.line(result, pt1, pt2, color, thickness, cv2.LINE_AA)
            # Draw arrowhead
            arrow_len = 20
            angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            arr_pt1 = (
                int(pt2[0] - arrow_len * np.cos(angle - np.pi/6)),
                int(pt2[1] - arrow_len * np.sin(angle - np.pi/6))
            )
            arr_pt2 = (
                int(pt2[0] - arrow_len * np.cos(angle + np.pi/6)),
                int(pt2[1] - arrow_len * np.sin(angle + np.pi/6))
            )
            cv2.line(result, pt2, arr_pt1, color, thickness, cv2.LINE_AA)
            cv2.line(result, pt2, arr_pt2, color, thickness, cv2.LINE_AA)
        else:
            color = (255, 255, 255)
            thickness = 2
            overlay = result.copy()
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
            result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)

        # Draw label
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2

        if is_best:
            label = f"BEST EV:{option.expected_value:.3f}"
            info_label = f"P:{option.pass_success*100:.0f}% xT:{option.xt_value:.3f}"

            font_scale = 0.6
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            cv2.rectangle(
                result,
                (mid_x - 2, mid_y - text_h - 4),
                (mid_x + text_w + 2, mid_y + 4),
                (0, 100, 0),
                -1
            )
            cv2.putText(
                result, label, (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                font_thickness, cv2.LINE_AA
            )

            (info_w, info_h), _ = cv2.getTextSize(
                info_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                result,
                (mid_x - 2, mid_y + 6),
                (mid_x + info_w + 2, mid_y + info_h + 12),
                (0, 100, 0),
                -1
            )
            cv2.putText(
                result, info_label, (mid_x, mid_y + info_h + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                1, cv2.LINE_AA
            )
        else:
            label = f"EV:{option.expected_value:.3f}"
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (mid_x - 2, mid_y - text_h - 4),
                (mid_x + text_w + 2, mid_y + 4),
                (50, 50, 50),
                -1
            )
            cv2.putText(
                overlay, label, (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness, cv2.LINE_AA
            )
            result = cv2.addWeighted(overlay, 0.4, result, 0.6, 0)

    return result
