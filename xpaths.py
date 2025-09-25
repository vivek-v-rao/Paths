#!/usr/bin/env python3
"""Simulate price paths from log returns and report extrema statistics."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Simulation parameters (edit these values to change the scenario)
NUM_PATHS = 10**6
NUM_STEPS = 252
LOG_RETURN_MEAN = 0.0
LOG_RETURN_STD = 0.01
INITIAL_PRICE = 100.0
RNG_SEED = None  # Set to an integer for reproducibility
BATCH_SIZE = 100_000  # Number of paths to process per vectorized batch
TERMINAL_THRESHOLDS = [100.0, 104.0]

def simulate_extrema(
    num_paths: int,
    num_steps: int,
    mean: float,
    std: float,
    initial_price: float,
    seed: int | None,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if num_paths <= 0:
        raise ValueError("num_paths must be positive")

    rng = np.random.default_rng(seed)

    maxima = np.empty(num_paths, dtype=np.float64)
    minima = np.empty(num_paths, dtype=np.float64)
    terminals = np.empty(num_paths, dtype=np.float64)
    initial_is_max = np.empty(num_paths, dtype=bool)
    initial_is_min = np.empty(num_paths, dtype=bool)

    batch_size = max(1, min(batch_size, num_paths))

    processed = 0
    while processed < num_paths:
        current_batch = min(batch_size, num_paths - processed)
        log_returns = rng.normal(mean, std, size=(current_batch, num_steps))

        if num_steps == 0:
            batch_prices = np.empty((current_batch, 0), dtype=np.float64)
        else:
            cumulative_log_returns = np.cumsum(log_returns, axis=1)
            batch_prices = initial_price * np.exp(cumulative_log_returns)

        if num_steps == 0:
            batch_max = np.full(current_batch, initial_price, dtype=np.float64)
            batch_min = np.full(current_batch, initial_price, dtype=np.float64)
            batch_terminal = np.full(current_batch, initial_price, dtype=np.float64)
        else:
            batch_max = np.maximum(initial_price, batch_prices.max(axis=1))
            batch_min = np.minimum(initial_price, batch_prices.min(axis=1))
            batch_terminal = batch_prices[:, -1]

        start = processed
        end = processed + current_batch
        maxima[start:end] = batch_max
        minima[start:end] = batch_min
        terminals[start:end] = batch_terminal
        initial_is_max[start:end] = np.isclose(batch_max, initial_price)
        initial_is_min[start:end] = np.isclose(batch_min, initial_price)

        processed = end

    return maxima, minima, terminals, initial_is_max, initial_is_min


def main() -> None:
    print("Simulation parameters:")
    print(f"  Paths:          {NUM_PATHS}")
    print(f"  Steps per path: {NUM_STEPS}")
    print(f"  Log-return mean: {LOG_RETURN_MEAN:.6f}")
    print(f"  Log-return std:  {LOG_RETURN_STD:.6f}")
    print(f"  Initial price:   {INITIAL_PRICE:.4f}")
    print(f"  RNG seed:        {RNG_SEED if RNG_SEED is not None else 'None'}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print()

    maxima, minima, terminals, initial_is_max, initial_is_min = simulate_extrema(
        num_paths=NUM_PATHS,
        num_steps=NUM_STEPS,
        mean=LOG_RETURN_MEAN,
        std=LOG_RETURN_STD,
        initial_price=INITIAL_PRICE,
        seed=RNG_SEED,
        batch_size=BATCH_SIZE,
    )

    print("Simulated {paths} paths with {steps} steps per path.".format(
        paths=NUM_PATHS,
        steps=NUM_STEPS,
    ))
    print()

    price_paths = pd.DataFrame(
        {
            "Maximum": maxima,
            "Minimum": minima,
            "Terminal": terminals,
        }
    )

    summary = pd.DataFrame(
        {
            "mean": price_paths.mean(axis=0),
            "median": price_paths.median(axis=0),
            "std": price_paths.std(axis=0, ddof=0),
            "min": price_paths.min(axis=0),
            "q1": price_paths.quantile(0.25, axis=0),
            "q3": price_paths.quantile(0.75, axis=0),
            "max": price_paths.max(axis=0),
        }
    )
    summary = summary[["mean", "median", "std", "min", "q1", "q3", "max"]]

    prob_initial_max = initial_is_max.mean()
    prob_initial_min = initial_is_min.mean()
    prob_initial_min_zd = 1 / np.sqrt(np.pi * NUM_STEPS)
    threshold_probs = {
        threshold: (terminals > threshold).mean()
        for threshold in TERMINAL_THRESHOLDS
    }

    print("Summary statistics (Pandas dataframe):")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Probability initial price is path maximum: {prob_initial_max:.4%}")
    print(f"Probability initial price is path minimum: {prob_initial_min:.4%}")
    print(f"Zero-drift probability that initial price is path minimum: {prob_initial_min_zd:.4%}")
    print()
    print("Probability terminal price exceeds thresholds:")
    for threshold, probability in threshold_probs.items():
        print(f"  > {threshold:.2f}: {probability:.4%}")

if __name__ == "__main__":
    main()