"""
Annotators module for creating supervision annotators.

Contains team-colored annotation setup.
"""

import supervision as sv

from config import (
    TEAM_COLORS,
    GOALKEEPER_COLORS,
    REFEREE_COLOR,
    BALL_COLOR_HEX,
)


def create_team_annotators() -> dict:
    """
    Create supervision annotators for team-classified players.

    Returns:
        Dictionary with annotators for each team, goalkeepers, referees, and ball.
    """
    # Team 0 annotators (pink)
    team_0_ellipse = sv.EllipseAnnotator(
        color=sv.Color.from_hex(TEAM_COLORS[0]),
        thickness=2,
    )
    team_0_label = sv.LabelAnnotator(
        color=sv.Color.from_hex(TEAM_COLORS[0]),
        text_color=sv.Color.from_hex('#FFFFFF'),
        text_position=sv.Position.BOTTOM_CENTER,
        text_padding=5,
        text_thickness=1,
    )
    team_0_trace = sv.TraceAnnotator(
        color=sv.Color.from_hex(TEAM_COLORS[0]),
        position=sv.Position.BOTTOM_CENTER,
        trace_length=30,
        thickness=2,
    )

    # Team 1 annotators (blue)
    team_1_ellipse = sv.EllipseAnnotator(
        color=sv.Color.from_hex(TEAM_COLORS[1]),
        thickness=2,
    )
    team_1_label = sv.LabelAnnotator(
        color=sv.Color.from_hex(TEAM_COLORS[1]),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER,
        text_padding=5,
        text_thickness=1,
    )
    team_1_trace = sv.TraceAnnotator(
        color=sv.Color.from_hex(TEAM_COLORS[1]),
        position=sv.Position.BOTTOM_CENTER,
        trace_length=30,
        thickness=2,
    )

    # Goalkeeper annotators
    gk_0_ellipse = sv.EllipseAnnotator(
        color=sv.Color.from_hex(GOALKEEPER_COLORS[0]),
        thickness=3,
    )
    gk_0_label = sv.LabelAnnotator(
        color=sv.Color.from_hex(GOALKEEPER_COLORS[0]),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER,
        text_padding=5,
        text_thickness=1,
    )

    gk_1_ellipse = sv.EllipseAnnotator(
        color=sv.Color.from_hex(GOALKEEPER_COLORS[1]),
        thickness=3,
    )
    gk_1_label = sv.LabelAnnotator(
        color=sv.Color.from_hex(GOALKEEPER_COLORS[1]),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER,
        text_padding=5,
        text_thickness=1,
    )

    # Referee annotators
    referee_ellipse = sv.EllipseAnnotator(
        color=sv.Color.from_hex(REFEREE_COLOR),
        thickness=2,
    )
    referee_label = sv.LabelAnnotator(
        color=sv.Color.from_hex(REFEREE_COLOR),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER,
        text_padding=5,
        text_thickness=1,
    )

    # Ball annotator
    ball_annotator = sv.BoxCornerAnnotator(
        color=sv.Color.from_hex(BALL_COLOR_HEX),
        thickness=2,
        corner_length=8,
    )

    return {
        'team_0': (team_0_ellipse, team_0_label),
        'team_1': (team_1_ellipse, team_1_label),
        'team_0_trace': team_0_trace,
        'team_1_trace': team_1_trace,
        'gk_0': (gk_0_ellipse, gk_0_label),
        'gk_1': (gk_1_ellipse, gk_1_label),
        'referee': (referee_ellipse, referee_label),
        'ball': ball_annotator,
    }


def annotate_with_teams(frame, detections, team_ids, annotators, show_traces: bool = False):
    """
    Annotate frame with team-colored player annotations.

    Args:
        frame: Video frame to annotate.
        detections: Player detections.
        team_ids: Team ID for each detection.
        annotators: Dictionary of annotators.
        show_traces: Whether to show movement trails.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()

    for team_id in [0, 1]:
        team_mask = team_ids == team_id
        team_detections = detections[team_mask]

        if len(team_detections) == 0:
            continue

        ellipse_annotator, label_annotator = annotators[f'team_{team_id}']

        if show_traces and team_detections.tracker_id is not None:
            trace_annotator = annotators.get(f'team_{team_id}_trace')
            if trace_annotator:
                annotated = trace_annotator.annotate(scene=annotated, detections=team_detections)

        if team_detections.tracker_id is not None:
            labels = [f"#{tid}" for tid in team_detections.tracker_id]
        else:
            labels = [f"T{team_id}" for _ in range(len(team_detections))]

        annotated = ellipse_annotator.annotate(scene=annotated, detections=team_detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=team_detections, labels=labels
        )

    return annotated


def annotate_goalkeepers_with_teams(frame, detections, team_ids, annotators):
    """
    Annotate goalkeepers with team-colored annotations.

    Args:
        frame: Video frame to annotate.
        detections: Goalkeeper detections.
        team_ids: Team ID for each goalkeeper.
        annotators: Dictionary of annotators.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()

    for team_id in [0, 1]:
        team_mask = team_ids == team_id
        gk_detections = detections[team_mask]

        if len(gk_detections) == 0:
            continue

        ellipse_annotator, label_annotator = annotators[f'gk_{team_id}']

        if gk_detections.tracker_id is not None:
            labels = [f"GK#{tid}" for tid in gk_detections.tracker_id]
        else:
            labels = [f"GK{team_id}" for _ in range(len(gk_detections))]

        annotated = ellipse_annotator.annotate(scene=annotated, detections=gk_detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=gk_detections, labels=labels
        )

    return annotated
