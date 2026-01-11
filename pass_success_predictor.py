"""
Pass Success Predictor - Inference utility for pass success prediction.

This module provides an easy-to-use interface for predicting pass success
probability, designed for integration with xT/PCF calculations.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Union, List, Tuple


class PassSuccessPredictor:
    """
    Predicts pass success probability based on passer and receiver coordinates.

    Usage:
        predictor = PassSuccessPredictor()
        prob = predictor.predict(start_x=50, start_y=34, end_x=70, end_y=40)
        print(f"Pass success probability: {prob:.2%}")
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model. If None, uses default path.
        """
        if model_path is None:
            # Use config path if available, otherwise fallback to relative path
            try:
                from config import PASS_SUCCESS_MODEL_PATH
                model_path = PASS_SUCCESS_MODEL_PATH
            except ImportError:
                model_path = Path(__file__).parent / "models" / "pass_success_model.pkl"

        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please run train_pass_success_model.py first."
            )

        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def _create_features(
        self,
        start_x: Union[float, np.ndarray],
        start_y: Union[float, np.ndarray],
        end_x: Union[float, np.ndarray],
        end_y: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        """
        Create feature DataFrame from coordinates.

        Coordinates should be in standard pitch dimensions (105m x 68m).
        """
        start_x = np.atleast_1d(start_x)
        start_y = np.atleast_1d(start_y)
        end_x = np.atleast_1d(end_x)
        end_y = np.atleast_1d(end_y)

        # Calculate derived features
        pass_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        pass_angle = np.arctan2(end_y - start_y, end_x - start_x)

        # Create DataFrame with proper feature names
        features = pd.DataFrame({
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'pass_distance': pass_distance,
            'pass_angle': pass_angle
        })

        return features

    def predict(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float
    ) -> float:
        """
        Predict pass success probability for a single pass.

        Args:
            start_x: Passer x-coordinate (0-105)
            start_y: Passer y-coordinate (0-68)
            end_x: Receiver x-coordinate (0-105)
            end_y: Receiver y-coordinate (0-68)

        Returns:
            Success probability between 0.0 and 1.0
        """
        features = self._create_features(start_x, start_y, end_x, end_y)
        prob = self.model.predict_proba(features)[0, 1]
        return float(prob)

    def predict_batch(
        self,
        start_x: np.ndarray,
        start_y: np.ndarray,
        end_x: np.ndarray,
        end_y: np.ndarray
    ) -> np.ndarray:
        """
        Predict pass success probability for multiple passes.

        Args:
            start_x: Array of passer x-coordinates
            start_y: Array of passer y-coordinates
            end_x: Array of receiver x-coordinates
            end_y: Array of receiver y-coordinates

        Returns:
            Array of success probabilities
        """
        features = self._create_features(start_x, start_y, end_x, end_y)
        probs = self.model.predict_proba(features)[:, 1]
        return probs

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict pass success probability from a DataFrame.

        Args:
            df: DataFrame with columns 'start_x', 'start_y', 'end_x', 'end_y'

        Returns:
            Series of success probabilities
        """
        probs = self.predict_batch(
            df['start_x'].values,
            df['start_y'].values,
            df['end_x'].values,
            df['end_y'].values
        )
        return pd.Series(probs, index=df.index, name='pass_success_prob')

    def find_best_passing_option(
        self,
        passer_x: float,
        passer_y: float,
        receiver_options: List[Tuple[float, float]],
        expected_values: List[float] = None
    ) -> dict:
        """
        Find the best passing option among multiple receivers.

        Args:
            passer_x: Passer x-coordinate
            passer_y: Passer y-coordinate
            receiver_options: List of (x, y) tuples for potential receivers
            expected_values: Optional list of expected values (xT, PCF) for each option

        Returns:
            Dictionary with best option details
        """
        n_options = len(receiver_options)
        receiver_x = np.array([r[0] for r in receiver_options])
        receiver_y = np.array([r[1] for r in receiver_options])

        # Predict success probabilities
        success_probs = self.predict_batch(
            np.full(n_options, passer_x),
            np.full(n_options, passer_y),
            receiver_x,
            receiver_y
        )

        # Calculate combined scores if expected values provided
        if expected_values is not None:
            expected_values = np.array(expected_values)
            combined_scores = success_probs * expected_values
        else:
            combined_scores = success_probs

        # Find best option
        best_idx = np.argmax(combined_scores)

        result = {
            'best_index': int(best_idx),
            'best_receiver': receiver_options[best_idx],
            'success_probability': float(success_probs[best_idx]),
            'all_probabilities': success_probs.tolist(),
        }

        if expected_values is not None:
            result['best_expected_value'] = float(expected_values[best_idx])
            result['best_combined_score'] = float(combined_scores[best_idx])
            result['all_combined_scores'] = combined_scores.tolist()

        return result


# Example usage and testing
if __name__ == "__main__":
    print("Testing PassSuccessPredictor...")
    print("="*50)

    try:
        predictor = PassSuccessPredictor()

        # Test single prediction
        prob = predictor.predict(
            start_x=50, start_y=34,  # Center of pitch
            end_x=70, end_y=40       # Forward pass to the right
        )
        print(f"\nSingle pass prediction:")
        print(f"  From (50, 34) to (70, 40)")
        print(f"  Success probability: {prob:.2%}")

        # Test batch prediction
        start_x = np.array([50, 30, 80])
        start_y = np.array([34, 20, 50])
        end_x = np.array([70, 50, 90])
        end_y = np.array([40, 34, 55])

        probs = predictor.predict_batch(start_x, start_y, end_x, end_y)
        print(f"\nBatch prediction:")
        for i in range(len(probs)):
            print(f"  Pass {i+1}: ({start_x[i]}, {start_y[i]}) -> ({end_x[i]}, {end_y[i]}) = {probs[i]:.2%}")

        # Test best option finder
        print(f"\nBest passing option test:")
        passer = (50, 34)
        receivers = [(60, 40), (70, 34), (55, 50), (80, 30)]
        xT_values = [0.02, 0.05, 0.01, 0.08]  # Example xT values

        result = predictor.find_best_passing_option(
            passer[0], passer[1],
            receivers,
            expected_values=xT_values
        )

        print(f"  Passer at: {passer}")
        print(f"  Receiver options: {receivers}")
        print(f"  xT values: {xT_values}")
        print(f"  Success probabilities: {[f'{p:.2%}' for p in result['all_probabilities']]}")
        print(f"  Combined scores: {[f'{s:.4f}' for s in result['all_combined_scores']]}")
        print(f"  Best option: Receiver {result['best_index']+1} at {result['best_receiver']}")
        print(f"    - Success prob: {result['success_probability']:.2%}")
        print(f"    - xT value: {result['best_expected_value']:.4f}")
        print(f"    - Combined score: {result['best_combined_score']:.4f}")

        print("\n" + "="*50)
        print("All tests passed!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo use the predictor, first run:")
        print("  1. python extract_pass_data.py")
        print("  2. python train_pass_success_model.py")
