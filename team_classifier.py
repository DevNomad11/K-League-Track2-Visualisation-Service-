"""
Team Classifier Module

Uses SigLIP for visual feature extraction, UMAP for dimensionality reduction,
and KMeans for clustering players into teams.

Adapted from sports/sports/common/team.py
"""

from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence: The input sequence to be batched.
        batch_size: The size of each batch.

    Yields:
        Batches of the input sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering players into teams.

    Usage:
        1. Collect player crops from video frames
        2. Call fit() with the crops to train the classifier
        3. Call predict() on new crops to get team IDs (0 or 1)
    """

    def __init__(self, device: str = 'cuda', batch_size: int = 32):
        """
        Initialize the TeamClassifier.

        Args:
            device: The device to run the model on ('cpu' or 'cuda').
            batch_size: The batch size for processing images.
        """
        self.device = device
        self.batch_size = batch_size

        print(f"Loading SigLIP model: {SIGLIP_MODEL_PATH}")
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH
        ).to(device)
        self.processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_PATH)

        # UMAP for dimensionality reduction (embedding dim -> 3)
        self.reducer = umap.UMAP(n_components=3, random_state=42)

        # KMeans for clustering into 2 teams
        self.cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)

        self._is_fitted = False
        print("TeamClassifier initialized!")

    def extract_features(self, crops: List[np.ndarray], show_progress: bool = True) -> np.ndarray:
        """
        Extract features from a list of image crops using SigLIP.

        Args:
            crops: List of image crops (BGR numpy arrays from OpenCV).
            show_progress: Whether to show progress bar.

        Returns:
            Extracted features as a numpy array of shape (n_crops, embedding_dim).
        """
        if len(crops) == 0:
            return np.array([])

        # Convert OpenCV BGR images to PIL RGB
        pil_crops = [sv.cv2_to_pillow(crop) for crop in crops]

        batches = list(create_batches(pil_crops, self.batch_size))
        data = []

        iterator = tqdm(batches, desc='Extracting embeddings') if show_progress else batches

        with torch.no_grad():
            for batch in iterator:
                inputs = self.processor(
                    images=batch, return_tensors="pt"
                ).to(self.device)

                outputs = self.features_model(**inputs)

                # Global average pooling over spatial dimensions
                # last_hidden_state shape: (batch, seq_len, hidden_dim)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier on a list of player image crops.

        This extracts features, reduces dimensionality with UMAP,
        and fits KMeans to cluster into 2 teams.

        Args:
            crops: List of player image crops (BGR numpy arrays).
        """
        if len(crops) < 2:
            raise ValueError("Need at least 2 crops to fit the classifier")

        print(f"Fitting TeamClassifier on {len(crops)} crops...")

        # Extract features
        data = self.extract_features(crops)

        # Reduce dimensionality
        print("Fitting UMAP...")
        projections = self.reducer.fit_transform(data)

        # Cluster
        print("Fitting KMeans...")
        self.cluster_model.fit(projections)

        self._is_fitted = True
        print("TeamClassifier fitted successfully!")

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team IDs for a list of player image crops.

        Args:
            crops: List of player image crops (BGR numpy arrays).

        Returns:
            Array of team IDs (0 or 1) for each crop.
        """
        if not self._is_fitted:
            raise RuntimeError("TeamClassifier must be fitted before prediction. Call fit() first.")

        if len(crops) == 0:
            return np.array([], dtype=np.int32)

        # Extract features (no progress bar for single-frame inference)
        data = self.extract_features(crops, show_progress=False)

        # Transform with fitted UMAP
        projections = self.reducer.transform(data)

        # Predict cluster labels
        return self.cluster_model.predict(projections)

    @property
    def is_fitted(self) -> bool:
        """Check if the classifier has been fitted."""
        return self._is_fitted


def get_player_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract image crops from a frame based on detection bounding boxes.

    Args:
        frame: The video frame (BGR numpy array).
        detections: Supervision Detections object with bounding boxes.

    Returns:
        List of cropped images (BGR numpy arrays).
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeeper_team_id(
    players: sv.Detections,
    player_team_ids: np.ndarray,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Assign goalkeepers to teams based on proximity to team centroids.

    Since goalkeepers often wear different colors than their team,
    we assign them based on which team's players they are closest to.

    Args:
        players: Detections of field players.
        player_team_ids: Team IDs (0 or 1) for each player.
        goalkeepers: Detections of goalkeepers.

    Returns:
        Array of team IDs for each goalkeeper.
    """
    if len(goalkeepers) == 0:
        return np.array([], dtype=np.int32)

    if len(players) == 0 or len(player_team_ids) == 0:
        # No players to reference, assign all to team 0
        return np.zeros(len(goalkeepers), dtype=np.int32)

    # Get bottom-center positions (feet position on pitch)
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Calculate centroid of each team
    team_0_mask = player_team_ids == 0
    team_1_mask = player_team_ids == 1

    if not np.any(team_0_mask) or not np.any(team_1_mask):
        # One team has no players, assign goalkeepers to the team with players
        if np.any(team_0_mask):
            return np.zeros(len(goalkeepers), dtype=np.int32)
        else:
            return np.ones(len(goalkeepers), dtype=np.int32)

    team_0_centroid = players_xy[team_0_mask].mean(axis=0)
    team_1_centroid = players_xy[team_1_mask].mean(axis=0)

    # Assign each goalkeeper to the nearest team
    goalkeeper_team_ids = []
    for gk_xy in goalkeepers_xy:
        dist_to_team_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_to_team_1 = np.linalg.norm(gk_xy - team_1_centroid)
        goalkeeper_team_ids.append(0 if dist_to_team_0 < dist_to_team_1 else 1)

    return np.array(goalkeeper_team_ids, dtype=np.int32)
