"""
Football Video Inference - Main Entry Point

Run inference on videos using trained YOLO models with BoT-SORT tracking
and pitch visualization.

Version 11: All features from v10 + Dynamic pitch control + Through-pass suggestions.

Features:
- Ball detection with box corner annotation
- Player/goalkeeper/referee detection with team classification
- BoT-SORT (Bag of Tricks for SORT) with Camera Motion Compensation
- Team classification using SigLIP + UMAP + KMeans clustering
- Pitch keypoint detection with 32 keypoints
- Mini-map visualization showing player positions on 2D pitch
- Voronoi diagram (static and dynamic)
- Pass success rate visualization
- xT-weighted pass route visualization
- Dynamic pitch control visualization (velocity-weighted Voronoi)
- Through-pass suggestions with optimal receiving points

Usage:
    python inference.py --input video.mp4

    # Dynamic pitch control visualization
    python inference.py --input video.mp4 --dynamic-voronoi --team-0-attacks-left

    # Through-pass suggestions
    python inference.py --input video.mp4 --pass-xt-v11 --team-0-attacks-left

Requires: pip install boxmot xgboost
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

import supervision as sv

# Add parent directory to path for imports from scripts folder
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

# Import modules
from config import (
    PROJECT_DIR,
    INPUT_VIDEO_DIR,
    OUTPUT_VIDEO_DIR,
    BALL_MODEL_PATH,
    PLAYER_MODEL_PATH,
    PITCH_MODEL_PATH,
    REID_MODEL_PATH,
    XT_GRID_PATH,
    PLAYER_CLASS_ID,
    GOALKEEPER_CLASS_ID,
    REFEREE_CLASS_ID,
    DEFAULT_CONF_THRESHOLDS,
    PITCH_CONFIG,
)
from view_transformer import ViewTransformer, create_view_transformer
from minimap import create_minimap, draw_players_on_minimap, draw_ball_on_minimap, draw_voronoi_on_minimap, overlay_minimap
from voronoi import draw_voronoi_on_frame, draw_dynamic_voronoi_on_frame
from velocity_tracker import PlayerVelocityTracker
from dynamic_pitch_control import calculate_all_pass_options
from pass_visualization import find_ball_possessor, draw_pass_success_lines, draw_pass_xt_lines, draw_pass_xt_lines_v11
from tracking import FixedIDTrackManager, TeamTrackManager, detections_to_botsort_format, botsort_output_to_detections, BOTSORT_AVAILABLE
from annotators import create_team_annotators, annotate_with_teams, annotate_goalkeepers_with_teams
from utils import load_models, create_botsort_tracker, collect_player_crops_for_fitting

# Optional imports with graceful fallback
BALL_TRACKER_AVAILABLE = False
try:
    from ball_tracker import RobustBallTracker, DetectionStatus, draw_ball_detection
    BALL_TRACKER_AVAILABLE = True
except ImportError:
    print("Warning: ball_tracker module not available. Robust ball tracking disabled.")
    RobustBallTracker = None
    DetectionStatus = None
    draw_ball_detection = None

TEAM_CLASSIFIER_AVAILABLE = False
try:
    from team_classifier import TeamClassifier, get_player_crops, resolve_goalkeeper_team_id
    TEAM_CLASSIFIER_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import team_classifier: {e}")
    TeamClassifier = None
    get_player_crops = None
    resolve_goalkeeper_team_id = None

PASS_PREDICTOR_AVAILABLE = False
try:
    from pass_success_predictor import PassSuccessPredictor
    PASS_PREDICTOR_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    print(f"Warning: Could not import pass_success_predictor: {e}")
    PassSuccessPredictor = None

XT_CALCULATOR_AVAILABLE = False
try:
    from calculate_xt_grid import XTCalculator
    XT_CALCULATOR_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    print(f"Warning: Could not import calculate_xt_grid: {e}")
    XTCalculator = None


def process_video(
    input_path,
    output_path=None,
    ball_model_path=None,
    player_model_path=None,
    pitch_model_path=None,
    use_ball=True,
    use_player=True,
    use_pitch=True,
    conf_thresholds=None,
    device=0,
    fit_frames=500,
    show_minimap=True,
    show_keypoints=False,
    show_voronoi=False,
    voronoi_opacity=0.3,
    show_voronoi_frame=False,
    voronoi_frame_opacity=0.25,
    robust_ball_tracking=False,
    roi_conf=0.10,
    max_frames_lost=30,
    show_trajectory=False,
    no_tracker=False,
    cmc_method='ecc',
    no_cmc=False,
    with_reid=True,
    show_traces=False,
    max_players=22,
    no_team_classification=False,
    show_pass_success=False,
    possession_threshold=3.0,
    team_0_attacks_ltr=True,
    show_pass_xt=False,
    xt_grid_path=None,
    show_dynamic_voronoi=False,
    max_player_speed=7.0,
    show_pass_xt_v11=False,
    top_n_passes=3,
    max_run_distance=15.0,
    highlight_zones=True,
):
    """Process a video file with detection models, BoT-SORT tracking, and pitch visualization."""

    # Check boxmot availability
    if not no_tracker and not BOTSORT_AVAILABLE:
        print("Warning: boxmot package not installed!")
        print("Install with: pip install boxmot")
        print("Continuing without tracking...")
        no_tracker = True

    if conf_thresholds is None:
        conf_thresholds = DEFAULT_CONF_THRESHOLDS.copy()

    # Load models
    models = load_models(ball_model_path, player_model_path, pitch_model_path,
                         use_ball, use_player, use_pitch)

    if not models:
        print("Error: No models loaded!")
        return

    # Handle no_tracker mode
    if no_tracker:
        print("Tracking disabled - showing raw detections")
        if robust_ball_tracking:
            print("Warning: --no-tracker overrides --robust-ball-tracking")
            robust_ball_tracking = False

    # Initialize robust ball tracker
    ball_tracker = None
    if robust_ball_tracking and 'ball' in models and BALL_TRACKER_AVAILABLE:
        ball_tracker = RobustBallTracker(
            model=models['ball'],
            ball_class_id=0,
            primary_conf=conf_thresholds.get('ball', 0.25),
            roi_conf=roi_conf,
            max_frames_lost=max_frames_lost,
            device=device,
        )
        print(f"Robust ball tracking enabled")

    # Initialize pass success predictor
    pass_predictor = None
    possession_threshold_cm = possession_threshold * 100
    if show_pass_success or show_pass_xt or show_pass_xt_v11:
        if PASS_PREDICTOR_AVAILABLE:
            try:
                pass_predictor = PassSuccessPredictor()
                print(f"Pass success prediction enabled")
            except FileNotFoundError as e:
                print(f"Warning: Could not load pass success model: {e}")
                show_pass_success = show_pass_xt = show_pass_xt_v11 = False
        else:
            show_pass_success = show_pass_xt = show_pass_xt_v11 = False

    # Initialize xT calculator
    xt_calculator = None
    if show_pass_xt or show_pass_xt_v11:
        if XT_CALCULATOR_AVAILABLE:
            try:
                xt_calculator = XTCalculator()
                grid_path = xt_grid_path or XT_GRID_PATH
                xt_calculator.load_grid(grid_path)
                print(f"xT calculator loaded from: {grid_path}")
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load xT grid: {e}")
                show_pass_xt = show_pass_xt_v11 = False
        else:
            show_pass_xt = show_pass_xt_v11 = False

    # Initialize velocity tracker
    velocity_tracker = None
    max_player_speed_cms = max_player_speed * 100
    max_run_distance_cm = max_run_distance * 100
    if show_dynamic_voronoi or show_pass_xt_v11:
        velocity_tracker = PlayerVelocityTracker(
            smoothing_factor=0.3,
            max_history=10,
            fps=30.0,
            min_speed_threshold=50.0
        )
        if show_dynamic_voronoi:
            print(f"Dynamic pitch control enabled (max speed: {max_player_speed} m/s)")
        if show_pass_xt_v11:
            print(f"Through-pass visualization enabled (top {top_n_passes})")

    # Initialize team classifier
    team_classifier = None
    use_team_tracking = False
    if 'player' in models and TEAM_CLASSIFIER_AVAILABLE and not no_team_classification:
        print("\n--- Initializing Team Classifier ---")
        team_classifier = TeamClassifier(device='cuda', batch_size=32)

        player_crops = collect_player_crops_for_fitting(
            video_path=input_path,
            player_model=models['player'],
            conf_threshold=conf_thresholds.get('player', 0.25),
            device=device,
            max_frames=fit_frames,
            sample_interval=5,
            get_player_crops=get_player_crops,
        )

        if len(player_crops) >= 10:
            team_classifier.fit(player_crops)
            use_team_tracking = not no_tracker
            print("Team classification enabled")
        else:
            print(f"Warning: Not enough player crops ({len(player_crops)})")
            team_classifier = None

    # Initialize annotators
    annotators = create_team_annotators()

    # Open video
    print(f"\nOpening video: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

    # Update velocity tracker FPS
    if velocity_tracker is not None:
        velocity_tracker.fps = float(fps)

    # Determine CMC method
    effective_cmc = None if no_cmc else cmc_method
    device_str = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'

    # Initialize trackers
    team_track_manager = None
    player_tracker = None
    fixed_id_manager = None

    if not no_tracker and use_player:
        print(f"\nInitializing BoT-SORT tracker:")
        print(f"  Device: {device_str}")
        print(f"  CMC method: {effective_cmc or 'disabled'}")

        if use_team_tracking:
            team_track_manager = TeamTrackManager(
                max_players_per_team=11,
                max_lost_frames=fps * 5,
                device=device_str,
                cmc_method=effective_cmc,
                with_reid=with_reid,
                fps=fps,
            )
            print("  Team tracking: ENABLED")
        else:
            player_tracker = create_botsort_tracker(
                device=device_str,
                half=torch.cuda.is_available(),
                cmc_method=effective_cmc,
                track_buffer=fps * 3,
                with_reid=with_reid,
            )
            fixed_id_manager = FixedIDTrackManager(
                max_players=max_players,
                max_lost_frames=fps * 5,
            )
            print(f"  Single tracker mode")

    # Setup output path
    if output_path is None:
        input_name = Path(input_path).stem
        os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
        suffix = "_final"
        if no_cmc:
            suffix += "_nocmc"
        output_path = os.path.join(OUTPUT_VIDEO_DIR, f"{input_name}{suffix}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Output: {output_path}")
    print(f"Processing frames...")

    frame_num = 0
    detection_stats = {'detected': 0, 'roi_recovered': 0, 'predicted': 0, 'lost': 0}
    track_stats = {'total_tracks': set(), 'active_tracks_per_frame': []}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()
        view_transformer = None
        ball_world_position = None
        player_world_positions = []
        player_team_ids_list = []
        player_tracker_ids_list = []

        # Pitch detection
        if 'pitch' in models:
            pitch_results = models['pitch'].predict(
                frame, conf=conf_thresholds.get('pitch', 0.5),
                device=device, verbose=False
            )
            if len(pitch_results) > 0:
                keypoints = sv.KeyPoints.from_ultralytics(pitch_results[0])
                if show_keypoints and keypoints is not None and len(keypoints.xy) > 0:
                    for i, (x, y) in enumerate(keypoints.xy[0]):
                        if x > 1 and y > 1:
                            cv2.circle(annotated_frame, (int(x), int(y)), 6, (0, 255, 255), -1)
                view_transformer = create_view_transformer(keypoints)

        # Ball detection
        ball_image_position = None
        if 'ball' in models:
            if ball_tracker:
                detection, status = ball_tracker.detect(frame)
                detection_stats[status.value] += 1
                trajectory = ball_tracker.detection_history if show_trajectory else None
                annotated_frame = draw_ball_detection(
                    annotated_frame, detection, status,
                    show_status=True, show_trajectory=show_trajectory,
                    trajectory_history=trajectory
                )
                if detection is not None:
                    ball_image_position = np.array([[(detection[0] + detection[2]) / 2, detection[3]]])
            else:
                ball_results = models['ball'].predict(
                    frame, conf=conf_thresholds.get('ball', 0.07),
                    device=device, verbose=False
                )
                if len(ball_results) > 0:
                    ball_detections = sv.Detections.from_ultralytics(ball_results[0])
                    if len(ball_detections) > 0:
                        best_idx = np.argmax(ball_detections.confidence)
                        ball_detections = ball_detections[[best_idx]]
                        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
                        annotated_frame = annotators['ball'].annotate(
                            scene=annotated_frame, detections=ball_detections
                        )
                        box = ball_detections.xyxy[0]
                        ball_image_position = np.array([[(box[0] + box[2]) / 2, box[3]]])

        # Transform ball to world coordinates
        if view_transformer is not None and ball_image_position is not None:
            try:
                ball_world_position = view_transformer.transform_points(ball_image_position.astype(np.float32))[0]
            except Exception:
                ball_world_position = None

        # Player detection
        if 'player' in models:
            player_conf = max(
                conf_thresholds.get('player', 0.25),
                conf_thresholds.get('goalkeeper', 0.25),
                conf_thresholds.get('referee', 0.25)
            )
            player_results = models['player'].predict(
                frame, conf=player_conf, device=device, verbose=False
            )

            for result in player_results:
                detections = sv.Detections.from_ultralytics(result)
                if len(detections) == 0:
                    continue
                detections = detections.with_nms(threshold=0.5, class_agnostic=True)
                if len(detections) == 0:
                    continue

                player_mask = detections.class_id == PLAYER_CLASS_ID
                goalkeeper_mask = detections.class_id == GOALKEEPER_CLASS_ID
                referee_mask = detections.class_id == REFEREE_CLASS_ID

                players = detections[player_mask]
                goalkeepers = detections[goalkeeper_mask]
                referees = detections[referee_mask]

                all_image_positions = []
                all_team_ids = []
                all_tracker_ids = []

                if no_tracker:
                    # Raw detections
                    labels = [f"{result.names[cls_id]} {conf:.2f}"
                              for cls_id, conf in zip(detections.class_id, detections.confidence)]
                    ellipse_annotator, label_annotator = annotators['team_0']
                    annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    for box in detections.xyxy:
                        all_image_positions.append([(box[0] + box[2]) / 2, box[3]])
                        all_team_ids.append(0)

                elif team_classifier is not None and use_team_tracking:
                    # Team-based tracking
                    if len(players) > 0:
                        player_crops = get_player_crops(frame, players)
                        player_team_ids_arr = team_classifier.predict(player_crops)

                        for team_id in [0, 1]:
                            team_mask_arr = player_team_ids_arr == team_id
                            team_players = players[team_mask_arr]
                            if len(team_players) > 0:
                                tracked = team_track_manager.update_team(team_id, team_players, frame)
                                if len(tracked) > 0:
                                    for tid in tracked.tracker_id:
                                        track_stats['total_tracks'].add(tid)
                                    labels = [f"#{tid}" for tid in tracked.tracker_id]
                                    if show_traces:
                                        trace_ann = annotators.get(f'team_{team_id}_trace')
                                        if trace_ann:
                                            annotated_frame = trace_ann.annotate(scene=annotated_frame, detections=tracked)
                                    ellipse_ann, label_ann = annotators[f'team_{team_id}']
                                    annotated_frame = ellipse_ann.annotate(scene=annotated_frame, detections=tracked)
                                    annotated_frame = label_ann.annotate(scene=annotated_frame, detections=tracked, labels=labels)
                                    for i, box in enumerate(tracked.xyxy):
                                        all_image_positions.append([(box[0] + box[2]) / 2, box[3]])
                                        all_team_ids.append(team_id)
                                        all_tracker_ids.append(tracked.tracker_id[i])

                    # Goalkeepers
                    if len(goalkeepers) > 0 and len(players) > 0:
                        gk_team_ids = resolve_goalkeeper_team_id(players, player_team_ids_arr, goalkeepers)
                        for team_id in [0, 1]:
                            team_mask_arr = gk_team_ids == team_id
                            team_gks = goalkeepers[team_mask_arr]
                            if len(team_gks) > 0:
                                tracked_gks = team_track_manager.update_team(team_id, team_gks, frame)
                                if len(tracked_gks) > 0:
                                    labels = [f"GK#{tid}" for tid in tracked_gks.tracker_id]
                                    ellipse_ann, label_ann = annotators[f'gk_{team_id}']
                                    annotated_frame = ellipse_ann.annotate(scene=annotated_frame, detections=tracked_gks)
                                    annotated_frame = label_ann.annotate(scene=annotated_frame, detections=tracked_gks, labels=labels)
                                    for i, box in enumerate(tracked_gks.xyxy):
                                        all_image_positions.append([(box[0] + box[2]) / 2, box[3]])
                                        all_team_ids.append(team_id)
                                        all_tracker_ids.append(tracked_gks.tracker_id[i])

                    # Referees
                    if len(referees) > 0:
                        tracked_refs = team_track_manager.update_referees(referees, frame)
                        if len(tracked_refs) > 0:
                            labels = [f"REF#{tid}" for tid in tracked_refs.tracker_id]
                            ellipse_ann, label_ann = annotators['referee']
                            annotated_frame = ellipse_ann.annotate(scene=annotated_frame, detections=tracked_refs)
                            annotated_frame = label_ann.annotate(scene=annotated_frame, detections=tracked_refs, labels=labels)

                elif team_classifier is not None:
                    # Team classification without tracking
                    if len(players) > 0:
                        player_crops = get_player_crops(frame, players)
                        player_team_ids_arr = team_classifier.predict(player_crops)
                        annotated_frame = annotate_with_teams(annotated_frame, players, player_team_ids_arr, annotators, show_traces)
                        for i, box in enumerate(players.xyxy):
                            all_image_positions.append([(box[0] + box[2]) / 2, box[3]])
                            all_team_ids.append(player_team_ids_arr[i])

                        if len(goalkeepers) > 0:
                            gk_team_ids = resolve_goalkeeper_team_id(players, player_team_ids_arr, goalkeepers)
                            annotated_frame = annotate_goalkeepers_with_teams(annotated_frame, goalkeepers, gk_team_ids, annotators)
                            for i, box in enumerate(goalkeepers.xyxy):
                                all_image_positions.append([(box[0] + box[2]) / 2, box[3]])
                                all_team_ids.append(gk_team_ids[i])

                    if len(referees) > 0:
                        ellipse_ann, label_ann = annotators['referee']
                        annotated_frame = ellipse_ann.annotate(scene=annotated_frame, detections=referees)
                        annotated_frame = label_ann.annotate(scene=annotated_frame, detections=referees, labels=[f"REF"] * len(referees))

                else:
                    # Single tracker without team classification
                    if player_tracker is not None:
                        dets = detections_to_botsort_format(detections, frame.shape)
                        tracks = player_tracker.update(dets, frame)
                        if fixed_id_manager is not None and len(tracks) > 0:
                            tracks = fixed_id_manager.update(tracks)
                        tracked_detections = botsort_output_to_detections(tracks, detections)
                        if len(tracked_detections) > 0:
                            for tid in tracked_detections.tracker_id:
                                track_stats['total_tracks'].add(tid)
                            labels = [f"#{tid}" for tid in tracked_detections.tracker_id]
                            ellipse_ann, label_ann = annotators['team_0']
                            annotated_frame = ellipse_ann.annotate(scene=annotated_frame, detections=tracked_detections)
                            annotated_frame = label_ann.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
                            for box in tracked_detections.xyxy:
                                all_image_positions.append([(box[0] + box[2]) / 2, box[3]])
                                all_team_ids.append(0)

                # Transform to world coordinates
                if view_transformer is not None and len(all_image_positions) > 0:
                    try:
                        image_positions = np.array(all_image_positions, dtype=np.float32)
                        world_positions = view_transformer.transform_points(image_positions)
                        for i, (world_pt, team_id) in enumerate(zip(world_positions, all_team_ids)):
                            if 0 <= world_pt[0] <= PITCH_CONFIG.length and 0 <= world_pt[1] <= PITCH_CONFIG.width:
                                player_world_positions.append(world_pt)
                                player_team_ids_list.append(team_id)
                                if i < len(all_tracker_ids):
                                    player_tracker_ids_list.append(all_tracker_ids[i])
                                else:
                                    player_tracker_ids_list.append(-1)
                    except Exception:
                        pass

        # Update velocity tracker
        if velocity_tracker is not None and len(player_world_positions) > 0:
            for world_pt, tid in zip(player_world_positions, player_tracker_ids_list):
                if tid != -1:
                    velocity_tracker.update(int(tid), np.array(world_pt), frame_num)
            if frame_num % 30 == 0:
                velocity_tracker.cleanup_old_tracks(frame_num, max_age=90)

        # Pass success visualization
        if (show_pass_success and pass_predictor is not None and view_transformer is not None and
            ball_world_position is not None and len(player_world_positions) > 0):
            try:
                player_world_arr = np.array(player_world_positions)
                team_ids_arr = np.array(player_team_ids_list)
                tracker_ids_arr = np.array(player_tracker_ids_list)
                possessor = find_ball_possessor(ball_world_position, player_world_arr, team_ids_arr, tracker_ids_arr, possession_threshold_cm)
                if possessor is not None:
                    attack_ltr = team_0_attacks_ltr if possessor['team_id'] == 0 else not team_0_attacks_ltr
                    teammate_mask = (team_ids_arr == possessor['team_id']) & (np.arange(len(team_ids_arr)) != possessor['index'])
                    teammate_positions = player_world_arr[teammate_mask]
                    if len(teammate_positions) > 0:
                        annotated_frame = draw_pass_success_lines(annotated_frame, view_transformer, possessor['world_pos'], teammate_positions, pass_predictor, attack_left_to_right=attack_ltr)
            except Exception:
                pass

        # xT visualization
        if (show_pass_xt and pass_predictor is not None and xt_calculator is not None and
            view_transformer is not None and ball_world_position is not None and len(player_world_positions) > 0):
            try:
                player_world_arr = np.array(player_world_positions)
                team_ids_arr = np.array(player_team_ids_list)
                tracker_ids_arr = np.array(player_tracker_ids_list)
                possessor = find_ball_possessor(ball_world_position, player_world_arr, team_ids_arr, tracker_ids_arr, possession_threshold_cm)
                if possessor is not None:
                    attack_ltr = team_0_attacks_ltr if possessor['team_id'] == 0 else not team_0_attacks_ltr
                    teammate_mask = (team_ids_arr == possessor['team_id']) & (np.arange(len(team_ids_arr)) != possessor['index'])
                    teammate_positions = player_world_arr[teammate_mask]
                    if len(teammate_positions) > 0:
                        annotated_frame, _ = draw_pass_xt_lines(annotated_frame, view_transformer, possessor['world_pos'], teammate_positions, pass_predictor, xt_calculator, attack_left_to_right=attack_ltr)
            except Exception:
                pass

        # v11 through-pass visualization
        if (show_pass_xt_v11 and pass_predictor is not None and xt_calculator is not None and
            velocity_tracker is not None and view_transformer is not None and
            ball_world_position is not None and len(player_world_positions) > 0):
            try:
                player_world_arr = np.array(player_world_positions)
                team_ids_arr = np.array(player_team_ids_list)
                tracker_ids_arr = np.array(player_tracker_ids_list)
                possessor = find_ball_possessor(ball_world_position, player_world_arr, team_ids_arr, tracker_ids_arr, possession_threshold_cm)
                if possessor is not None:
                    attack_ltr = team_0_attacks_ltr if possessor['team_id'] == 0 else not team_0_attacks_ltr
                    teammate_mask = (team_ids_arr == possessor['team_id']) & (np.arange(len(team_ids_arr)) != possessor['index'])
                    teammate_positions = player_world_arr[teammate_mask]
                    teammate_ids = tracker_ids_arr[teammate_mask]
                    if len(teammate_positions) > 0:
                        all_velocities = velocity_tracker.get_velocities_for_ids(tracker_ids_arr)
                        pass_options = calculate_all_pass_options(
                            ball_pos=ball_world_position, passer_pos=possessor['world_pos'],
                            teammate_positions=teammate_positions, teammate_ids=teammate_ids,
                            all_positions=player_world_arr, all_velocities=all_velocities,
                            possessor_team_id=possessor['team_id'], all_team_ids=team_ids_arr,
                            velocity_tracker=velocity_tracker, pass_predictor=pass_predictor,
                            xt_calculator=xt_calculator, attack_left_to_right=attack_ltr,
                            pitch_config=PITCH_CONFIG, max_speed=max_player_speed_cms,
                            max_run_distance=max_run_distance_cm, top_n=top_n_passes
                        )
                        if len(pass_options) > 0:
                            annotated_frame = draw_pass_xt_lines_v11(annotated_frame, view_transformer, possessor['world_pos'], pass_options, highlight_zones=highlight_zones, top_n_display=top_n_passes)
            except Exception:
                pass

        # Static Voronoi on frame
        if show_voronoi_frame and view_transformer is not None and len(player_world_positions) > 0:
            player_world_arr = np.array(player_world_positions)
            team_ids_arr = np.array(player_team_ids_list)
            team_0_world = player_world_arr[team_ids_arr == 0]
            team_1_world = player_world_arr[team_ids_arr == 1]
            if len(team_0_world) > 0 and len(team_1_world) > 0:
                annotated_frame = draw_voronoi_on_frame(annotated_frame, view_transformer, team_0_world, team_1_world, PITCH_CONFIG, opacity=voronoi_frame_opacity)

        # Dynamic Voronoi on frame
        if show_dynamic_voronoi and velocity_tracker is not None and view_transformer is not None and len(player_world_positions) > 0:
            try:
                player_world_arr = np.array(player_world_positions)
                team_ids_arr = np.array(player_team_ids_list)
                tracker_ids_arr = np.array(player_tracker_ids_list)
                team_0_mask = team_ids_arr == 0
                team_1_mask = team_ids_arr == 1
                team_0_world = player_world_arr[team_0_mask]
                team_1_world = player_world_arr[team_1_mask]
                team_0_ids = tracker_ids_arr[team_0_mask]
                team_1_ids = tracker_ids_arr[team_1_mask]
                if len(team_0_world) > 0 and len(team_1_world) > 0:
                    team_0_velocities = velocity_tracker.get_velocities_for_ids(team_0_ids)
                    team_1_velocities = velocity_tracker.get_velocities_for_ids(team_1_ids)
                    annotated_frame = draw_dynamic_voronoi_on_frame(
                        annotated_frame, view_transformer, team_0_world, team_0_velocities,
                        team_1_world, team_1_velocities, PITCH_CONFIG, opacity=voronoi_frame_opacity, max_speed=max_player_speed_cms
                    )
            except Exception:
                pass

        # Minimap
        if show_minimap and view_transformer is not None:
            minimap = create_minimap()
            if show_voronoi and len(player_world_positions) > 0:
                player_world_arr = np.array(player_world_positions)
                team_ids_arr = np.array(player_team_ids_list)
                team_0_world = player_world_arr[team_ids_arr == 0]
                team_1_world = player_world_arr[team_ids_arr == 1]
                if len(team_0_world) > 0 and len(team_1_world) > 0:
                    minimap = draw_voronoi_on_minimap(minimap, team_0_world, team_1_world, opacity=voronoi_opacity)
            if len(player_world_positions) > 0:
                minimap = draw_players_on_minimap(minimap, np.array(player_world_positions), np.array(player_team_ids_list))
            if ball_world_position is not None:
                minimap = draw_ball_on_minimap(minimap, ball_world_position)
            annotated_frame = overlay_minimap(annotated_frame, minimap)

        out.write(annotated_frame)
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames ({100*frame_num/total_frames:.1f}%)")

    cap.release()
    out.release()

    print(f"\nComplete! Processed {frame_num} frames")
    print(f"Output saved to: {output_path}")

    if ball_tracker:
        print(f"\n--- Ball Tracking Statistics ---")
        print(f"  Primary detections:  {detection_stats['detected']} frames")
        print(f"  ROI recovered:       {detection_stats['roi_recovered']} frames")
        print(f"  Kalman predicted:    {detection_stats['predicted']} frames")
        print(f"  Lost:                {detection_stats['lost']} frames")

    if track_stats['total_tracks']:
        print(f"\n--- Player Tracking Statistics ---")
        print(f"  Total unique track IDs: {len(track_stats['total_tracks'])}")


def main():
    parser = argparse.ArgumentParser(
        description='Football Video Inference with BoT-SORT tracking and pitch visualization (v11)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output
    parser.add_argument('--input', '-i', type=str, required=True, help='Input video filename')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output video filename')

    # Model options
    parser.add_argument('--ball-model', type=str, default=None, help='Path to ball detection model')
    parser.add_argument('--player-model', type=str, default=None, help='Path to player detection model')
    parser.add_argument('--pitch-model', type=str, default=None, help='Path to pitch detection model')
    parser.add_argument('--ball-only', action='store_true', help='Only use ball detection')
    parser.add_argument('--player-only', action='store_true', help='Only use player detection')
    parser.add_argument('--pitch-only', action='store_true', help='Only use pitch detection')
    parser.add_argument('--no-pitch', action='store_true', help='Disable pitch detection')

    # Confidence
    parser.add_argument('--conf', type=float, default=None, help='Global confidence threshold')
    parser.add_argument('--conf-ball', type=float, default=None, help='Ball detection confidence')
    parser.add_argument('--conf-player', type=float, default=None, help='Player detection confidence')
    parser.add_argument('--conf-pitch', type=float, default=None, help='Pitch detection confidence')

    # Device
    parser.add_argument('--device', type=int, default=0, help='CUDA device')

    # Team classification
    parser.add_argument('--fit-frames', type=int, default=500, help='Frames for team classifier fitting')
    parser.add_argument('--no-team-classification', action='store_true', help='Disable team classification')

    # Visualization
    parser.add_argument('--no-minimap', action='store_true', help='Disable minimap overlay')
    parser.add_argument('--show-keypoints', action='store_true', help='Show detected pitch keypoints')
    parser.add_argument('--voronoi', action='store_true', help='Enable Voronoi diagram on minimap')
    parser.add_argument('--voronoi-opacity', type=float, default=0.3, help='Voronoi minimap opacity')
    parser.add_argument('--voronoi-frame', action='store_true', help='Enable Voronoi overlay on video frame')
    parser.add_argument('--voronoi-frame-opacity', type=float, default=0.25, help='Voronoi frame overlay opacity')

    # Tracking
    parser.add_argument('--cmc-method', type=str, default='ecc', choices=['ecc', 'orb', 'sift', 'sof'], help='CMC method')
    parser.add_argument('--no-cmc', action='store_true', help='Disable Camera Motion Compensation')
    parser.add_argument('--no-reid', action='store_true', help='Disable Re-ID features')
    parser.add_argument('--show-traces', action='store_true', help='Show movement trails')
    parser.add_argument('--no-tracker', action='store_true', help='Disable tracking')
    parser.add_argument('--max-players', type=int, default=22, help='Maximum number of player track IDs')

    # Ball tracking
    parser.add_argument('--robust-ball-tracking', action='store_true', help='Enable robust ball tracking')
    parser.add_argument('--roi-conf', type=float, default=0.10, help='ROI search confidence threshold')
    parser.add_argument('--max-frames-lost', type=int, default=30, help='Max frames to predict ball')
    parser.add_argument('--show-trajectory', action='store_true', help='Draw ball trajectory')

    # Pass visualization
    parser.add_argument('--pass-success', action='store_true', help='Show pass success rate visualization')
    parser.add_argument('--possession-threshold', type=float, default=3.0, help='Ball possession distance (meters)')
    parser.add_argument('--team-0-attacks-left', action='store_true', help='Team 0 attacks left-to-right')
    parser.add_argument('--team-0-attacks-right', action='store_true', help='Team 0 attacks right-to-left')

    # xT visualization
    parser.add_argument('--pass-xt', action='store_true', help='Show pass routes with expected value')
    parser.add_argument('--xt-grid', type=str, default=XT_GRID_PATH, help='Path to xT grid file')

    # v11 features
    parser.add_argument('--dynamic-voronoi', action='store_true', help='Show velocity-weighted dynamic pitch control')
    parser.add_argument('--max-player-speed', type=float, default=7.0, help='Max player speed (m/s)')
    parser.add_argument('--pass-xt-v11', action='store_true', help='Show v11 through-pass visualization')
    parser.add_argument('--top-n-passes', type=int, default=3, help='Number of top pass options to display')
    parser.add_argument('--max-run-distance', type=float, default=15.0, help='Max look-ahead distance (meters)')
    parser.add_argument('--highlight-zones', action='store_true', default=True, help='Highlight target zones')
    parser.add_argument('--no-highlight-zones', action='store_false', dest='highlight_zones')

    args = parser.parse_args()

    # Validate attack direction
    if args.pass_success or args.pass_xt or args.pass_xt_v11 or args.dynamic_voronoi:
        if args.team_0_attacks_left and args.team_0_attacks_right:
            print("Error: Cannot specify both attack directions")
            return
        if not args.team_0_attacks_left and not args.team_0_attacks_right:
            print("Error: Must specify --team-0-attacks-left or --team-0-attacks-right for pass/Voronoi features")
            return

    os.makedirs(INPUT_VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(INPUT_VIDEO_DIR, input_path)

    if not os.path.exists(input_path):
        print(f"Error: Input video not found: {input_path}")
        return

    output_path = None
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(OUTPUT_VIDEO_DIR, output_path)

    # Determine which models to use
    use_ball = not (args.player_only or args.pitch_only)
    use_player = not (args.ball_only or args.pitch_only)
    use_pitch = not (args.ball_only or args.player_only or args.no_pitch)
    if args.pitch_only:
        use_pitch = True

    # Build confidence thresholds
    conf_thresholds = DEFAULT_CONF_THRESHOLDS.copy()
    if args.conf is not None:
        for key in conf_thresholds:
            conf_thresholds[key] = args.conf
    if args.conf_ball is not None:
        conf_thresholds['ball'] = args.conf_ball
    if args.conf_player is not None:
        conf_thresholds['player'] = args.conf_player
    if args.conf_pitch is not None:
        conf_thresholds['pitch'] = args.conf_pitch

    process_video(
        input_path=input_path,
        output_path=output_path,
        ball_model_path=args.ball_model,
        player_model_path=args.player_model,
        pitch_model_path=args.pitch_model,
        use_ball=use_ball,
        use_player=use_player,
        use_pitch=use_pitch,
        conf_thresholds=conf_thresholds,
        device=args.device,
        fit_frames=args.fit_frames,
        show_minimap=not args.no_minimap,
        show_keypoints=args.show_keypoints,
        show_voronoi=args.voronoi,
        voronoi_opacity=args.voronoi_opacity,
        show_voronoi_frame=args.voronoi_frame,
        voronoi_frame_opacity=args.voronoi_frame_opacity,
        robust_ball_tracking=args.robust_ball_tracking,
        roi_conf=args.roi_conf,
        max_frames_lost=args.max_frames_lost,
        show_trajectory=args.show_trajectory,
        no_tracker=args.no_tracker,
        cmc_method=args.cmc_method,
        no_cmc=args.no_cmc,
        with_reid=not args.no_reid,
        show_traces=args.show_traces,
        max_players=args.max_players,
        no_team_classification=args.no_team_classification,
        show_pass_success=args.pass_success,
        possession_threshold=args.possession_threshold,
        team_0_attacks_ltr=args.team_0_attacks_left,
        show_pass_xt=args.pass_xt,
        xt_grid_path=args.xt_grid,
        show_dynamic_voronoi=args.dynamic_voronoi,
        max_player_speed=args.max_player_speed,
        show_pass_xt_v11=args.pass_xt_v11,
        top_n_passes=args.top_n_passes,
        max_run_distance=args.max_run_distance,
        highlight_zones=args.highlight_zones,
    )


if __name__ == '__main__':
    main()
