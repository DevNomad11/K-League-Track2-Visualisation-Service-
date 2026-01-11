"""
Expected Threat (xT) Grid Calculator for K-League Data.

This module calculates xT values for each zone on the pitch based on
event data. xT measures the probability of scoring from each location.

Algorithm:
    xT(z) = s(z) * g(z) + (1 - s(z)) * sum[m(z,z') * xT(z')]

Where:
    s(z) = probability of shooting from zone z
    g(z) = probability of scoring given a shot from zone z
    m(z,z') = probability of moving the ball from zone z to zone z'
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from pathlib import Path
from typing import Tuple, Optional


class XTCalculator:
    """
    Calculates and provides xT (Expected Threat) values for pitch zones.

    Usage:
        # Calculate xT from data
        calculator = XTCalculator()
        calculator.calculate_from_data(raw_data_path)
        calculator.save_grid(output_path)

        # Or load pre-computed grid
        calculator = XTCalculator()
        calculator.load_grid(grid_path)

        # Get xT values
        xt_value = calculator.get_xt(x=80, y=34)
        xt_added = calculator.get_xt_added(start_x=50, start_y=34, end_x=80, end_y=34)
    """

    def __init__(self, n_x: int = 16, n_y: int = 12,
                 pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Initialize the xT calculator.

        Args:
            n_x: Number of zones along the x-axis (length)
            n_y: Number of zones along the y-axis (width)
            pitch_length: Pitch length in meters (default: 105)
            pitch_width: Pitch width in meters (default: 68)
        """
        self.n_x = n_x
        self.n_y = n_y
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Zone dimensions
        self.zone_width = pitch_length / n_x
        self.zone_height = pitch_width / n_y

        # xT grid (will be calculated or loaded)
        self.xt_grid: Optional[np.ndarray] = None

        # Intermediate data for analysis
        self.shot_rate: Optional[np.ndarray] = None
        self.goal_rate: Optional[np.ndarray] = None
        self.move_matrix: Optional[np.ndarray] = None
        self.statistics: dict = {}

    def _coord_to_zone(self, x: float, y: float) -> Tuple[int, int]:
        """Convert pitch coordinates to zone indices."""
        x_zone = int(np.clip(x / self.zone_width, 0, self.n_x - 1))
        y_zone = int(np.clip(y / self.zone_height, 0, self.n_y - 1))
        return x_zone, y_zone

    def _coord_to_zone_vectorized(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert arrays of coordinates to zone indices."""
        x_zones = np.clip((x / self.zone_width).astype(int), 0, self.n_x - 1)
        y_zones = np.clip((y / self.zone_height).astype(int), 0, self.n_y - 1)
        return x_zones, y_zones

    def calculate_from_data(self, data_path: str, convergence_threshold: float = 1e-6,
                            max_iterations: int = 100) -> np.ndarray:
        """
        Calculate xT grid from raw event data.

        Args:
            data_path: Path to raw_data.csv
            convergence_threshold: Stop iteration when max change < threshold
            max_iterations: Maximum number of iterations

        Returns:
            xT grid as numpy array (n_x, n_y)
        """
        print("Loading data...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} events")

        # Filter and preprocess data
        df = self._preprocess_data(df)

        # Calculate zone probabilities
        print("\nCalculating zone probabilities...")
        self._calculate_shot_rate(df)
        self._calculate_goal_rate(df)
        self._calculate_move_matrix(df)

        # Iterate to find xT values
        print("\nIterating to find xT values...")
        self.xt_grid = self._iterate_xt(convergence_threshold, max_iterations)

        return self.xt_grid

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess event data for xT calculation."""
        # Define action types we care about
        move_types = ['Pass', 'Pass_Freekick', 'Pass_Corner', 'Cross', 'Carry',
                      'Throw-In', 'Dribble']
        shot_types = ['Shot', 'Shot_Freekick']

        # Filter to relevant events
        relevant_types = move_types + shot_types
        df = df[df['type_name'].isin(relevant_types)].copy()

        # Remove invalid coordinates
        df = df.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])

        # Ensure coordinates are within pitch bounds
        df = df[
            (df['start_x'] >= 0) & (df['start_x'] <= self.pitch_length) &
            (df['start_y'] >= 0) & (df['start_y'] <= self.pitch_width) &
            (df['end_x'] >= 0) & (df['end_x'] <= self.pitch_length) &
            (df['end_y'] >= 0) & (df['end_y'] <= self.pitch_width)
        ]

        # Create binary columns
        df['is_shot'] = df['type_name'].isin(shot_types).astype(int)
        df['is_goal'] = (df['result_name'] == 'Goal').astype(int)
        # Only count SUCCESSFUL moves for the transition matrix
        df['is_move'] = (
            df['type_name'].isin(move_types) &
            df['result_name'].isin(['Successful', 'Goal'])
        ).astype(int)

        # Add zone indices
        df['start_zone_x'], df['start_zone_y'] = self._coord_to_zone_vectorized(
            df['start_x'].values, df['start_y'].values
        )
        df['end_zone_x'], df['end_zone_y'] = self._coord_to_zone_vectorized(
            df['end_x'].values, df['end_y'].values
        )

        print(f"Preprocessed {len(df):,} relevant events")
        print(f"  - Shots: {df['is_shot'].sum():,}")
        print(f"  - Goals: {df['is_goal'].sum():,}")
        print(f"  - Moves: {df['is_move'].sum():,}")

        self.statistics['total_events'] = len(df)
        self.statistics['total_shots'] = int(df['is_shot'].sum())
        self.statistics['total_goals'] = int(df['is_goal'].sum())
        self.statistics['total_moves'] = int(df['is_move'].sum())

        return df

    def _calculate_shot_rate(self, df: pd.DataFrame):
        """Calculate probability of shooting from each zone."""
        self.shot_rate = np.zeros((self.n_x, self.n_y))
        self.move_rate = np.zeros((self.n_x, self.n_y))
        self.turnover_rate = np.zeros((self.n_x, self.n_y))

        for x in range(self.n_x):
            for y in range(self.n_y):
                zone_mask = (df['start_zone_x'] == x) & (df['start_zone_y'] == y)
                total_actions = zone_mask.sum()
                shots = (zone_mask & (df['is_shot'] == 1)).sum()
                successful_moves = (zone_mask & (df['is_move'] == 1)).sum()
                turnovers = total_actions - shots - successful_moves

                if total_actions > 0:
                    self.shot_rate[x, y] = shots / total_actions
                    self.move_rate[x, y] = successful_moves / total_actions
                    self.turnover_rate[x, y] = turnovers / total_actions

        print(f"Shot rate range: {self.shot_rate.min():.4f} - {self.shot_rate.max():.4f}")
        print(f"Move rate range: {self.move_rate.min():.4f} - {self.move_rate.max():.4f}")
        print(f"Turnover rate range: {self.turnover_rate.min():.4f} - {self.turnover_rate.max():.4f}")

    def _calculate_goal_rate(self, df: pd.DataFrame):
        """Calculate probability of scoring given a shot from each zone."""
        self.goal_rate = np.zeros((self.n_x, self.n_y))
        shots_df = df[df['is_shot'] == 1]

        for x in range(self.n_x):
            for y in range(self.n_y):
                zone_mask = (shots_df['start_zone_x'] == x) & (shots_df['start_zone_y'] == y)
                total_shots = zone_mask.sum()
                goals = (zone_mask & (shots_df['is_goal'] == 1)).sum()

                if total_shots > 0:
                    self.goal_rate[x, y] = goals / total_shots
                else:
                    zone_center_x = (x + 0.5) * self.zone_width
                    zone_center_y = (y + 0.5) * self.zone_height
                    dist_to_goal = np.sqrt(
                        (self.pitch_length - zone_center_x)**2 +
                        (self.pitch_width/2 - zone_center_y)**2
                    )
                    self.goal_rate[x, y] = max(0, 0.3 - 0.01 * dist_to_goal)

        print(f"Goal rate range: {self.goal_rate.min():.4f} - {self.goal_rate.max():.4f}")

    def _calculate_move_matrix(self, df: pd.DataFrame):
        """Calculate transition probabilities between zones."""
        self.move_matrix = np.zeros((self.n_x, self.n_y, self.n_x, self.n_y))
        moves_df = df[df['is_move'] == 1].copy()

        transition_counts = np.zeros((self.n_x, self.n_y, self.n_x, self.n_y))
        zone_totals = np.zeros((self.n_x, self.n_y))

        for _, row in moves_df.iterrows():
            x1, y1 = int(row['start_zone_x']), int(row['start_zone_y'])
            x2, y2 = int(row['end_zone_x']), int(row['end_zone_y'])
            transition_counts[x1, y1, x2, y2] += 1
            zone_totals[x1, y1] += 1

        for x1 in range(self.n_x):
            for y1 in range(self.n_y):
                if zone_totals[x1, y1] > 0:
                    self.move_matrix[x1, y1, :, :] = (
                        transition_counts[x1, y1, :, :] / zone_totals[x1, y1]
                    )

        non_zero = np.sum(self.move_matrix > 0)
        total = self.n_x * self.n_y * self.n_x * self.n_y
        print(f"Move matrix sparsity: {non_zero}/{total} ({100*non_zero/total:.2f}% non-zero)")

    def _iterate_xt(self, threshold: float, max_iter: int) -> np.ndarray:
        """Iterate to find xT values using value iteration."""
        xt = self.shot_rate * self.goal_rate

        for iteration in range(max_iter):
            xt_new = np.zeros_like(xt)

            for x in range(self.n_x):
                for y in range(self.n_y):
                    shoot_value = self.shot_rate[x, y] * self.goal_rate[x, y]
                    move_value = 0.0
                    for x2 in range(self.n_x):
                        for y2 in range(self.n_y):
                            move_value += self.move_matrix[x, y, x2, y2] * xt[x2, y2]
                    xt_new[x, y] = shoot_value + self.move_rate[x, y] * move_value

            max_change = np.max(np.abs(xt_new - xt))
            xt = xt_new

            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: max change = {max_change:.8f}")

            if max_change < threshold:
                print(f"  Converged at iteration {iteration + 1}")
                self.statistics['iterations'] = iteration + 1
                self.statistics['final_max_change'] = float(max_change)
                break
        else:
            print(f"  Warning: Did not converge after {max_iter} iterations")
            self.statistics['iterations'] = max_iter
            self.statistics['final_max_change'] = float(max_change)

        return xt

    def get_xt(self, x: float, y: float) -> float:
        """
        Get xT value at given coordinates.

        Args:
            x: X coordinate (0-105)
            y: Y coordinate (0-68)

        Returns:
            xT value at the location
        """
        if self.xt_grid is None:
            raise ValueError("xT grid not calculated. Call calculate_from_data() or load_grid() first.")

        x_zone, y_zone = self._coord_to_zone(x, y)
        return float(self.xt_grid[x_zone, y_zone])

    def get_xt_added(self, start_x: float, start_y: float,
                     end_x: float, end_y: float) -> float:
        """
        Calculate xT gained or lost by moving from start to end.

        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates

        Returns:
            xT difference (positive = gained threat, negative = lost threat)
        """
        return self.get_xt(end_x, end_y) - self.get_xt(start_x, start_y)

    def get_xt_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get xT values for arrays of coordinates."""
        if self.xt_grid is None:
            raise ValueError("xT grid not calculated.")

        x_zones, y_zones = self._coord_to_zone_vectorized(x, y)
        return self.xt_grid[x_zones, y_zones]

    def save_grid(self, output_dir: str):
        """Save xT grid and statistics to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "xt_grid.npy", self.xt_grid)
        print(f"Saved: {output_dir / 'xt_grid.npy'}")

        df = pd.DataFrame(
            self.xt_grid.T,
            columns=[f"x{i}" for i in range(self.n_x)],
            index=[f"y{i}" for i in range(self.n_y)]
        )
        df.to_csv(output_dir / "xt_grid.csv")
        print(f"Saved: {output_dir / 'xt_grid.csv'}")

        self.statistics['grid_shape'] = [self.n_x, self.n_y]
        self.statistics['zone_size'] = [self.zone_width, self.zone_height]
        self.statistics['xt_range'] = [float(self.xt_grid.min()), float(self.xt_grid.max())]
        self.statistics['xt_mean'] = float(self.xt_grid.mean())

        with open(output_dir / "xt_statistics.json", 'w') as f:
            json.dump(self.statistics, f, indent=2)
        print(f"Saved: {output_dir / 'xt_statistics.json'}")

    def load_grid(self, grid_path: str):
        """Load pre-computed xT grid from file."""
        grid_path = Path(grid_path)

        if grid_path.suffix == '.npy':
            self.xt_grid = np.load(grid_path)
        elif grid_path.suffix == '.csv':
            df = pd.read_csv(grid_path, index_col=0)
            self.xt_grid = df.values.T
        else:
            raise ValueError(f"Unsupported file format: {grid_path.suffix}")

        self.n_x, self.n_y = self.xt_grid.shape
        self.zone_width = self.pitch_length / self.n_x
        self.zone_height = self.pitch_width / self.n_y

        print(f"Loaded xT grid: {self.n_x}x{self.n_y}")


def draw_pitch(ax, pitch_length=105, pitch_width=68):
    """Draw a football pitch on the given axes."""
    ax.plot([0, 0], [0, pitch_width], color='white', linewidth=2)
    ax.plot([0, pitch_length], [pitch_width, pitch_width], color='white', linewidth=2)
    ax.plot([pitch_length, pitch_length], [pitch_width, 0], color='white', linewidth=2)
    ax.plot([pitch_length, 0], [0, 0], color='white', linewidth=2)

    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color='white', linewidth=2)

    center_circle = Circle((pitch_length/2, pitch_width/2), 9.15,
                           fill=False, color='white', linewidth=2)
    ax.add_patch(center_circle)
    ax.plot(pitch_length/2, pitch_width/2, 'o', color='white', markersize=3)

    ax.plot([0, 16.5], [pitch_width/2 + 20.15, pitch_width/2 + 20.15], color='white', linewidth=2)
    ax.plot([16.5, 16.5], [pitch_width/2 + 20.15, pitch_width/2 - 20.15], color='white', linewidth=2)
    ax.plot([16.5, 0], [pitch_width/2 - 20.15, pitch_width/2 - 20.15], color='white', linewidth=2)

    ax.plot([pitch_length, pitch_length - 16.5], [pitch_width/2 + 20.15, pitch_width/2 + 20.15], color='white', linewidth=2)
    ax.plot([pitch_length - 16.5, pitch_length - 16.5], [pitch_width/2 + 20.15, pitch_width/2 - 20.15], color='white', linewidth=2)
    ax.plot([pitch_length - 16.5, pitch_length], [pitch_width/2 - 20.15, pitch_width/2 - 20.15], color='white', linewidth=2)

    ax.plot([0, 5.5], [pitch_width/2 + 9.15, pitch_width/2 + 9.15], color='white', linewidth=2)
    ax.plot([5.5, 5.5], [pitch_width/2 + 9.15, pitch_width/2 - 9.15], color='white', linewidth=2)
    ax.plot([5.5, 0], [pitch_width/2 - 9.15, pitch_width/2 - 9.15], color='white', linewidth=2)

    ax.plot([pitch_length, pitch_length - 5.5], [pitch_width/2 + 9.15, pitch_width/2 + 9.15], color='white', linewidth=2)
    ax.plot([pitch_length - 5.5, pitch_length - 5.5], [pitch_width/2 + 9.15, pitch_width/2 - 9.15], color='white', linewidth=2)
    ax.plot([pitch_length - 5.5, pitch_length], [pitch_width/2 - 9.15, pitch_width/2 - 9.15], color='white', linewidth=2)

    ax.plot(11, pitch_width/2, 'o', color='white', markersize=3)
    ax.plot(pitch_length - 11, pitch_width/2, 'o', color='white', markersize=3)

    left_arc = Arc((11, pitch_width/2), 18.3, 18.3, angle=0, theta1=308, theta2=52, color='white', linewidth=2)
    right_arc = Arc((pitch_length - 11, pitch_width/2), 18.3, 18.3, angle=0, theta1=128, theta2=232, color='white', linewidth=2)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    ax.set_xlim(-2, pitch_length + 2)
    ax.set_ylim(-2, pitch_width + 2)
    ax.set_aspect('equal')
    ax.axis('off')


def create_xt_heatmap(calculator: XTCalculator, output_path: str, title: str = "Expected Threat (xT) Grid"):
    """Create a heatmap visualization of the xT grid."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor('#1a472a')

    extent = [0, calculator.pitch_length, 0, calculator.pitch_width]
    im = ax.imshow(
        calculator.xt_grid.T,
        extent=extent,
        origin='lower',
        cmap='RdYlGn',
        alpha=0.8,
        aspect='auto'
    )

    draw_pitch(ax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Expected Threat (xT)', fontsize=12)

    for x in range(calculator.n_x):
        for y in range(calculator.n_y):
            value = calculator.xt_grid[x, y]
            center_x = (x + 0.5) * calculator.zone_width
            center_y = (y + 0.5) * calculator.zone_height

            if value > 0.01:
                text_color = 'black' if value < 0.15 else 'white'
                ax.text(center_x, center_y, f'{value:.2f}',
                       ha='center', va='center', fontsize=7,
                       color=text_color, fontweight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a472a', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calculate Expected Threat (xT) grid from K-League data')
    parser.add_argument('--data', type=str, default=None, help='Path to raw_data.csv')
    parser.add_argument('--output', type=str, default=None, help='Output directory for xT files')
    parser.add_argument('--grid-x', type=int, default=16, help='Number of zones along x-axis')
    parser.add_argument('--grid-y', type=int, default=12, help='Number of zones along y-axis')
    parser.add_argument('--threshold', type=float, default=1e-6, help='Convergence threshold')
    parser.add_argument('--max-iter', type=int, default=500, help='Maximum iterations')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    data_path = args.data or (base_dir / "dacon_data" / "raw_data.csv")
    output_dir = args.output or (base_dir / "dacon_data")

    print("=" * 60)
    print("K-League Expected Threat (xT) Calculator")
    print("=" * 60)
    print(f"\nGrid size: {args.grid_x} x {args.grid_y}")
    print(f"Zone size: {105/args.grid_x:.2f}m x {68/args.grid_y:.2f}m")
    print(f"Data path: {data_path}")
    print(f"Output dir: {output_dir}")

    calculator = XTCalculator(n_x=args.grid_x, n_y=args.grid_y)
    calculator.calculate_from_data(
        data_path,
        convergence_threshold=args.threshold,
        max_iterations=args.max_iter
    )

    print("\nSaving results...")
    calculator.save_grid(output_dir)

    print("\nGenerating heatmap...")
    create_xt_heatmap(
        calculator,
        output_dir / "xt_heatmap.png",
        title=f"K-League Expected Threat (xT) - {args.grid_x}x{args.grid_y} Grid"
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"xT range: {calculator.xt_grid.min():.4f} - {calculator.xt_grid.max():.4f}")
    print(f"xT mean: {calculator.xt_grid.mean():.4f}")
    print(f"Highest xT zones (attacking third):")

    for x in range(calculator.n_x - 4, calculator.n_x):
        for y in range(calculator.n_y):
            if calculator.xt_grid[x, y] > 0.05:
                center_x = (x + 0.5) * calculator.zone_width
                center_y = (y + 0.5) * calculator.zone_height
                print(f"  Zone ({x},{y}) at ({center_x:.1f}m, {center_y:.1f}m): {calculator.xt_grid[x, y]:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
