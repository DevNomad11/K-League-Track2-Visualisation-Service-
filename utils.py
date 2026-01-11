"""
Utilities module for helper functions.

Contains model loading, tracker creation, and data collection utilities.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict

import supervision as sv
from ultralytics import YOLO

from config import (
    BALL_MODEL_PATH,
    PLAYER_MODEL_PATH,
    PITCH_MODEL_PATH,
    REID_MODEL_PATH,
    PLAYER_CLASS_ID,
)

# Check if boxmot is available
try:
    from boxmot import BotSort
    BOTSORT_AVAILABLE = True
except ImportError:
    BOTSORT_AVAILABLE = False
    BotSort = None


def load_models(
    ball_model_path: str = None,
    player_model_path: str = None,
    pitch_model_path: str = None,
    use_ball: bool = True,
    use_player: bool = True,
    use_pitch: bool = True
) -> Dict[str, YOLO]:
    """
    Load YOLO models for inference.

    Args:
        ball_model_path: Path to ball detection model.
        player_model_path: Path to player detection model.
        pitch_model_path: Path to pitch detection model.
        use_ball: Whether to load ball model.
        use_player: Whether to load player model.
        use_pitch: Whether to load pitch model.

    Returns:
        Dictionary of loaded models.
    """
    models = {}

    if use_ball:
        ball_path = ball_model_path or BALL_MODEL_PATH
        if os.path.exists(ball_path):
            models['ball'] = YOLO(ball_path)
            print(f"Loaded ball model: {ball_path}")
        else:
            print(f"Warning: Ball model not found at {ball_path}")

    if use_player:
        player_path = player_model_path or PLAYER_MODEL_PATH
        if os.path.exists(player_path):
            models['player'] = YOLO(player_path)
            print(f"Loaded player model: {player_path}")
        else:
            print(f"Warning: Player model not found at {player_path}")

    if use_pitch:
        pitch_path = pitch_model_path or PITCH_MODEL_PATH
        if os.path.exists(pitch_path):
            models['pitch'] = YOLO(pitch_path)
            print(f"Loaded pitch model: {pitch_path}")
        else:
            print(f"Warning: Pitch model not found at {pitch_path}")

    return models


def create_botsort_tracker(
    device: str = 'cuda:0',
    half: bool = True,
    cmc_method: str = 'ecc',
    reid_weights: str = None,
    track_high_thresh: float = 0.2,
    track_low_thresh: float = 0.05,
    new_track_thresh: float = 0.2,
    track_buffer: int = 90,
    match_thresh: float = 0.7,
    proximity_thresh: float = 0.6,
    appearance_thresh: float = 0.2,
    with_reid: bool = True,
):
    """
    Create and configure a BoT-SORT tracker.

    Args:
        device: Device string (e.g., 'cuda:0' or 'cpu').
        half: Use half precision.
        cmc_method: Camera Motion Compensation method.
        reid_weights: Path to Re-ID model weights.
        track_high_thresh: High confidence threshold for tracking.
        track_low_thresh: Low confidence threshold for tracking.
        new_track_thresh: Threshold for creating new tracks.
        track_buffer: Number of frames to keep lost tracks.
        match_thresh: IoU matching threshold.
        proximity_thresh: Proximity threshold for matching.
        appearance_thresh: Appearance threshold for Re-ID.
        with_reid: Enable Re-ID features.

    Returns:
        Configured BotSort tracker.
    """
    if not BOTSORT_AVAILABLE:
        raise ImportError("boxmot package not installed. Install with: pip install boxmot")

    if device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU for Re-ID")
        device = 'cpu'
        half = False

    torch_device = torch.device(device)

    tracker = BotSort(
        reid_weights=reid_weights or Path(REID_MODEL_PATH),
        device=torch_device,
        half=half,
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        cmc_method=cmc_method,
        proximity_thresh=proximity_thresh,
        appearance_thresh=appearance_thresh,
        with_reid=with_reid,
    )

    return tracker


def collect_player_crops_for_fitting(
    video_path: str,
    player_model: YOLO,
    conf_threshold: float,
    device: int,
    max_frames: int = 500,
    sample_interval: int = 5,
    get_player_crops=None,
) -> list:
    """
    Collect player crops from video frames for team classifier fitting.

    Args:
        video_path: Path to video file.
        player_model: YOLO model for player detection.
        conf_threshold: Confidence threshold for detection.
        device: CUDA device ID.
        max_frames: Maximum frames to sample.
        sample_interval: Sample every Nth frame.
        get_player_crops: Function to extract crops from detections.

    Returns:
        List of player crop images.
    """
    if get_player_crops is None:
        print("Warning: get_player_crops function not provided")
        return []

    print(f"\n--- Collecting player crops for team classifier fitting ---")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_sample = min(max_frames, total_frames // sample_interval)

    print(f"Sampling {frames_to_sample} frames (every {sample_interval}th frame)")

    all_crops = []
    frame_num = 0
    sampled_count = 0

    while sampled_count < frames_to_sample:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_interval == 0:
            results = player_model.predict(
                frame,
                conf=conf_threshold,
                device=device,
                verbose=False
            )

            for result in results:
                detections = sv.Detections.from_ultralytics(result)
                player_mask = detections.class_id == PLAYER_CLASS_ID
                player_detections = detections[player_mask]

                if len(player_detections) > 0:
                    player_detections = player_detections.with_nms(
                        threshold=0.5, class_agnostic=True
                    )
                    crops = get_player_crops(frame, player_detections)
                    all_crops.extend(crops)

            sampled_count += 1
            if sampled_count % 50 == 0:
                print(f"  Sampled {sampled_count}/{frames_to_sample} frames, "
                      f"collected {len(all_crops)} crops")

        frame_num += 1

    cap.release()
    print(f"Collected {len(all_crops)} player crops from {sampled_count} frames")
    return all_crops
