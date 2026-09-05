#!/usr/bin/env python3
"""Benchmark runner utility for NanoLocz v1.0."""

import time
import json
import csv
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    dataset_size: str
    hardware: str
    backend: str  # 'cpu' or 'gpu'
    duration_seconds: float
    throughput: float  # frames/sec or particles/sec
    memory_mb: float
    timestamp: str


def time_function(func, *args, **kwargs) -> tuple[Any, float]:
    """Time a function execution and return (result, duration)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def save_results_csv(results: list[BenchmarkResult], filepath: Path):
    """Save benchmark results to CSV file."""
    if not results:
        return
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def save_results_json(results: list[BenchmarkResult], filepath: Path):
    """Save benchmark results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def generate_dataset(width: int, height: int, frames: int, seed: int = 42):
    """Generate synthetic AFM movie dataset for benchmarking."""
    import numpy as np
    
    rng = np.random.default_rng(seed)
    # Generate realistic AFM-like data with particles
    base = np.zeros((frames, height, width), dtype=np.float64)
    
    # Add some Gaussian peaks to simulate particles
    n_particles = min(50, width * height // 100)
    for _ in range(n_particles):
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        sigma = rng.uniform(2, 5)
        amplitude = rng.uniform(100, 500)
        
        yy, xx = np.ogrid[:height, :width]
        gaussian = amplitude * np.exp(-((xx - x)**2 + **(yy - y)2) / (2 * sigma**2))
        base += gaussian
    
    # Add noise
    noise = rng.normal(0, 10, base.shape)
    return base + noise


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 0.001:
        return f"{seconds*1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def print_results_table(results: list[BenchmarkResult]):
    """Print benchmark results as formatted table."""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    print(f"{'Test':<30} {'Size':<15} {'Backend':<10} {'Duration':<12} {'Throughput':<15} {'Memory':<10}")
    print("-"*100)
    
    for r in results:
        throughput_str = f"{r.throughput:.1f}/s" if r.throughput > 0 else "N/A"
        memory_str = f"{r.memory_mb:.1f} MB" if r.memory_mb > 0 else "N/A"
        print(f"{r.name:<30} {r.dataset_size:<15} {r.backend:<10} {format_duration(r.duration_seconds):<12} {throughput_str:<15} {memory_str:<10}")
    
    print("="*100 + "\n")
