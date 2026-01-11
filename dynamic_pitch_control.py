"""
Dynamic Pitch Control module for velocity-weighted analysis.

Contains time-to-reach calculations, control masks, and pass option analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from config import SoccerPitchConfiguration, PITCH_CONFIG
from velocity_tracker import PlayerVelocityTracker


@dataclass
class PassOption:
    """Represents a potential pass option with EV analysis."""
    player_idx: int               # Index in teammates array
    tracker_id: int               # Player's tracker ID
    current_pos: np.ndarray       # Current position in cm
    optimal_point: np.ndarray     # Best receiving point within control zone (cm)
    run_direction: np.ndarray     # Estimated run direction (unit vector)
    pass_success: float           # Probability of pass success
    xt_value: float               # xT at optimal point
    expected_value: float         # pass_success * xt_value


def calculate_time_to_reach(
    player_pos: np.ndarray,
    player_velocity: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    max_speed: float = 700.0,
    reaction_time: float = 0.0
) -> np.ndarray:
    """
    Calculate time for a player to reach each target point.

    Uses a simple physics model that considers:
    - Current velocity component toward target (momentum advantage)
    - Maximum sprint speed for acceleration

    Args:
        player_pos: Player position (2,) in cm
        player_velocity: Player velocity (2,) in cm/s
        target_x: Grid of target x coordinates (H, W)
        target_y: Grid of target y coordinates (H, W)
        max_speed: Maximum sprint speed in cm/s (~7 m/s = 700 cm/s)
        reaction_time: Reaction time before player starts moving (seconds)

    Returns:
        Array of times (H, W) in seconds
    """
    dx = target_x - player_pos[0]
    dy = target_y - player_pos[1]
    distance = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    distance = np.maximum(distance, 1.0)

    # Direction to target (normalized)
    dir_x = dx / distance
    dir_y = dy / distance

    # Velocity component toward target
    vel_toward_target = player_velocity[0] * dir_x + player_velocity[1] * dir_y

    # Effective speed considers current momentum
    momentum_factor = 0.3
    effective_speed = max_speed + momentum_factor * np.clip(vel_toward_target, -max_speed, max_speed)
    effective_speed = np.maximum(effective_speed, max_speed * 0.5)

    # Time = reaction + distance / effective_speed
    time = reaction_time + distance / effective_speed

    return time


def calculate_min_time_to_reach_team(
    positions: np.ndarray,
    velocities: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    max_speed: float = 700.0,
    reaction_time: float = 0.0
) -> np.ndarray:
    """
    Calculate minimum time for any player on the team to reach each point.

    Args:
        positions: Team player positions (N, 2) in cm
        velocities: Team player velocities (N, 2) in cm/s
        target_x: Grid of target x coordinates (H, W)
        target_y: Grid of target y coordinates (H, W)
        max_speed: Maximum sprint speed in cm/s
        reaction_time: Reaction time in seconds

    Returns:
        Array of minimum times (H, W) in seconds
    """
    if len(positions) == 0:
        return np.full_like(target_x, np.inf, dtype=np.float32)

    all_times = []
    for i in range(len(positions)):
        time = calculate_time_to_reach(
            positions[i], velocities[i], target_x, target_y,
            max_speed, reaction_time
        )
        all_times.append(time)

    all_times = np.stack(all_times, axis=0)
    min_time = np.min(all_times, axis=0)

    return min_time


def compute_player_dynamic_control_mask(
    player_idx: int,
    player_pos: np.ndarray,
    player_velocity: np.ndarray,
    all_positions: np.ndarray,
    all_velocities: np.ndarray,
    pitch_config: SoccerPitchConfiguration = None,
    grid_resolution: int = 100,
    max_speed: float = 700.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a player's dynamic control zone (where they can reach first).

    Args:
        player_idx: Index of the player in all_positions
        player_pos: Player's position (2,) in cm
        player_velocity: Player's velocity (2,) in cm/s
        all_positions: All player positions (N, 2) in cm
        all_velocities: All player velocities (N, 2) in cm/s
        pitch_config: Soccer pitch configuration
        grid_resolution: Number of grid points along length axis
        max_speed: Maximum sprint speed in cm/s

    Returns:
        Tuple of:
        - mask: (H, W) boolean where this player reaches first
        - grid_x: (H, W) x coordinates in cm
        - grid_y: (H, W) y coordinates in cm
    """
    if pitch_config is None:
        pitch_config = PITCH_CONFIG

    aspect_ratio = pitch_config.width / pitch_config.length
    grid_height = int(grid_resolution * aspect_ratio)

    x = np.linspace(0, pitch_config.length, grid_resolution)
    y = np.linspace(0, pitch_config.width, grid_height)
    grid_x, grid_y = np.meshgrid(x, y)

    # Calculate time for this player to reach each point
    player_time = calculate_time_to_reach(
        player_pos, player_velocity, grid_x, grid_y, max_speed
    )

    # Calculate time for all other players
    all_min_time = np.full_like(grid_x, np.inf, dtype=np.float32)
    for i in range(len(all_positions)):
        if i == player_idx:
            continue
        other_time = calculate_time_to_reach(
            all_positions[i], all_velocities[i], grid_x, grid_y, max_speed
        )
        all_min_time = np.minimum(all_min_time, other_time)

    # Player controls where they can reach before anyone else
    control_mask = player_time < all_min_time

    return control_mask, grid_x, grid_y


def find_optimal_pass_point(
    ball_pos: np.ndarray,
    player_pos: np.ndarray,
    run_direction: np.ndarray,
    control_mask: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    pass_predictor,
    xt_calculator,
    attack_left_to_right: bool,
    pitch_config: SoccerPitchConfiguration = None,
    max_run_distance: float = 1500.0,
    num_samples: int = 20
) -> Tuple[np.ndarray, float, float, float]:
    """
    Find the optimal receiving point within a player's control zone.

    Args:
        ball_pos: Ball position in cm (2,)
        player_pos: Player's current position in cm (2,)
        run_direction: Player's run direction (unit vector)
        control_mask: Boolean mask of player's control zone
        grid_x: Grid x coordinates in cm
        grid_y: Grid y coordinates in cm
        pass_predictor: PassSuccessPredictor instance
        xt_calculator: XTCalculator instance
        attack_left_to_right: Whether team attacks left to right
        pitch_config: Soccer pitch configuration
        max_run_distance: Maximum look-ahead distance in cm
        num_samples: Number of sample points along run direction

    Returns:
        Tuple of (optimal_point, pass_success, xt_value, expected_value)
    """
    if pitch_config is None:
        pitch_config = PITCH_CONFIG

    # Generate sample points along run direction
    distances = np.linspace(0, max_run_distance, num_samples)
    sample_points = np.array([
        player_pos + d * run_direction for d in distances
    ])

    # Filter points within pitch bounds
    valid_mask = (
        (sample_points[:, 0] >= 0) &
        (sample_points[:, 0] <= pitch_config.length) &
        (sample_points[:, 1] >= 0) &
        (sample_points[:, 1] <= pitch_config.width)
    )

    # Check which points are within player's control zone
    grid_res_x = grid_x.shape[1]
    grid_res_y = grid_x.shape[0]

    for i in range(len(sample_points)):
        if not valid_mask[i]:
            continue
        ix = int(sample_points[i, 0] / pitch_config.length * (grid_res_x - 1))
        iy = int(sample_points[i, 1] / pitch_config.width * (grid_res_y - 1))
        ix = np.clip(ix, 0, grid_res_x - 1)
        iy = np.clip(iy, 0, grid_res_y - 1)

        if not control_mask[iy, ix]:
            valid_mask[i] = False

    # If no valid points, return current position
    if not np.any(valid_mask):
        ball_m = ball_pos / 100.0
        player_m = player_pos / 100.0

        if attack_left_to_right:
            start_x, start_y = ball_m[0], ball_m[1]
            end_x, end_y = player_m[0], player_m[1]
        else:
            start_x, start_y = 105 - ball_m[0], 68 - ball_m[1]
            end_x, end_y = 105 - player_m[0], 68 - player_m[1]

        try:
            pass_success = float(pass_predictor.predict(start_x, start_y, end_x, end_y))
            xt_value = float(xt_calculator.get_xt(end_x, end_y))
        except:
            pass_success = 0.5
            xt_value = 0.01

        expected_value = pass_success * xt_value
        return player_pos.copy(), pass_success, xt_value, expected_value

    # Calculate EV for each valid point
    valid_points = sample_points[valid_mask]
    best_ev = -1.0
    best_idx = 0
    best_pass_success = 0.0
    best_xt = 0.0

    ball_m = ball_pos / 100.0

    for i, point in enumerate(valid_points):
        point_m = point / 100.0

        if attack_left_to_right:
            start_x, start_y = ball_m[0], ball_m[1]
            end_x, end_y = point_m[0], point_m[1]
        else:
            start_x, start_y = 105 - ball_m[0], 68 - ball_m[1]
            end_x, end_y = 105 - point_m[0], 68 - point_m[1]

        try:
            pass_success = float(pass_predictor.predict(start_x, start_y, end_x, end_y))
            xt_value = float(xt_calculator.get_xt(end_x, end_y))
        except:
            pass_success = 0.5
            xt_value = 0.01

        ev = pass_success * xt_value

        if ev > best_ev:
            best_ev = ev
            best_idx = i
            best_pass_success = pass_success
            best_xt = xt_value

    return valid_points[best_idx], best_pass_success, best_xt, best_ev


def calculate_all_pass_options(
    ball_pos: np.ndarray,
    passer_pos: np.ndarray,
    teammate_positions: np.ndarray,
    teammate_ids: np.ndarray,
    all_positions: np.ndarray,
    all_velocities: np.ndarray,
    possessor_team_id: int,
    all_team_ids: np.ndarray,
    velocity_tracker: PlayerVelocityTracker,
    pass_predictor,
    xt_calculator,
    attack_left_to_right: bool,
    pitch_config: SoccerPitchConfiguration = None,
    max_speed: float = 700.0,
    max_run_distance: float = 1500.0,
    top_n: int = 5
) -> List[PassOption]:
    """
    Calculate pass options for all teammates and return top N by EV.

    Args:
        ball_pos: Ball position in cm (2,)
        passer_pos: Passer's position in cm (2,)
        teammate_positions: Positions of teammates (excluding passer) in cm (N, 2)
        teammate_ids: Tracker IDs of teammates (N,)
        all_positions: All player positions on field (M, 2)
        all_velocities: All player velocities (M, 2)
        possessor_team_id: Team ID of the ball possessor
        all_team_ids: Team IDs for all players (M,)
        velocity_tracker: PlayerVelocityTracker instance
        pass_predictor: PassSuccessPredictor instance
        xt_calculator: XTCalculator instance
        attack_left_to_right: Whether possessing team attacks left to right
        pitch_config: Soccer pitch configuration
        max_speed: Maximum sprint speed in cm/s
        max_run_distance: Maximum look-ahead distance for through passes in cm
        top_n: Number of top options to return

    Returns:
        List of PassOption objects sorted by expected_value descending
    """
    if pitch_config is None:
        pitch_config = PITCH_CONFIG

    options = []

    # Attack direction vector
    if attack_left_to_right:
        attack_direction = np.array([1.0, 0.0])
    else:
        attack_direction = np.array([-1.0, 0.0])

    for i, (pos, tid) in enumerate(zip(teammate_positions, teammate_ids)):
        # Get player's velocity
        vel = velocity_tracker.get_velocity(int(tid))
        if vel is None:
            vel = np.array([0.0, 0.0])

        # Get run direction
        run_dir = velocity_tracker.get_run_direction(
            int(tid), attack_direction=attack_direction
        )

        # Find player's index in all_positions
        player_idx = -1
        for j in range(len(all_positions)):
            if np.allclose(all_positions[j], pos, atol=1.0):
                player_idx = j
                break

        if player_idx == -1:
            continue

        # Compute dynamic control zone
        control_mask, grid_x, grid_y = compute_player_dynamic_control_mask(
            player_idx, pos, vel, all_positions, all_velocities,
            pitch_config, grid_resolution=50, max_speed=max_speed
        )

        # Find optimal pass point
        optimal_point, pass_success, xt_value, ev = find_optimal_pass_point(
            ball_pos, pos, run_dir, control_mask, grid_x, grid_y,
            pass_predictor, xt_calculator, attack_left_to_right,
            pitch_config, max_run_distance
        )

        option = PassOption(
            player_idx=i,
            tracker_id=int(tid),
            current_pos=pos.copy(),
            optimal_point=optimal_point,
            run_direction=run_dir,
            pass_success=pass_success,
            xt_value=xt_value,
            expected_value=ev
        )
        options.append(option)

    # Sort by EV descending and return top N
    options.sort(key=lambda x: x.expected_value, reverse=True)
    return options[:top_n]
