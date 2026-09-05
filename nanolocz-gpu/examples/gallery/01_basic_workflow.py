#!/usr/bin/env python3
"""
NanoLocz v1.0 - Basic AFM Analysis Workflow

This example demonstrates the complete NanoLocz workflow:
1. Load AFM image/movie
2. Preprocess (leveling and filtering)
3. Detect particles
4. Track particles across frames
5. Generate LAFM reconstruction
6. Export results

Expected runtime: < 30 seconds for test data
"""

import numpy as np
from pathlib import Path

# Import NanoLocz core modules
from nanolocz.core.leveling import level_image
from nanolocz.core.filters import gaussian_filter
from nanolocz.core.detection import detect_particles
from nanolocz.core.types import LocalizedParticle
from nanolocz.core.tracking import track_particles, TrackParams
from nanolocz.gpu.lafm import splat_localizations_gpu


def main():
    print("="*60)
    print("NanoLocz v1.0 - Basic AFM Analysis Workflow")
    print("="*60)
    
    # Step 1: Create synthetic test data (in real use, load from file)
    print("\n[1/6] Creating synthetic AFM movie...")
    rng = np.random.default_rng(42)
    frames, height, width = 10, 256, 256
    
    # Generate movie with Gaussian peaks simulating particles
    movie = np.zeros((frames, height, width), dtype=np.float64)
    for f in range(frames):
        for _ in range(20):  # 20 particles per frame
            x = rng.integers(20, width-20)
            y = rng.integers(20, height-20)
            # Add slight motion between frames
            x += rng.integers(-3, 4) * f
            y += rng.integers(-3, 4) * f
            sigma = 3.0
            amplitude = 200 + rng.uniform(-50, 50)
            
            yy, xx = np.ogrid[:height, :width]
            gaussian = amplitude * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            movie[f] += gaussian
        
        # Add noise
        movie[f] += rng.normal(0, 15, (height, width))
    
    print(f"    Created {frames} frames of {height}x{width} pixels")
    print(f"    ~{20 * frames} particle detections expected")
    
    # Step 2: Preprocess - leveling
    print("\n[2/6] Preprocessing - plane leveling...")
    leveled_movie = np.zeros_like(movie)
    for f in range(frames):
        leveled_frame, meta = level_image(movie[f], method='plane')
        leveled_movie[f] = leveled_frame
    print("    Plane leveling applied to all frames")
    
    # Step 3: Preprocess - filtering
    print("\n[3/6] Preprocessing - Gaussian filter (sigma=1.5)...")
    filtered_movie = np.zeros_like(leveled_movie)
    for f in range(frames):
        filtered_movie[f] = gaussian_filter(leveled_movie[f], sigma=1.5)
    print("    Gaussian smoothing applied")
    
    # Step 4: Particle detection
    print("\n[4/6] Detecting particles (threshold=3.0, min_distance=5)...")
    all_localizations_by_frame = []  # List of lists for tracking API
    all_localizations_flat = []  # Flat list for LAFM
    
    for f in range(frames):
        result = detect_particles(
            filtered_movie[f],
            threshold=3.0,
            min_distance=5
        )
        
        # Convert to LocalizedParticle objects for tracking API
        frame_localizations = []
        n_particles = len(result.coordinates)
        for i in range(n_particles):
            particle = LocalizedParticle(
                x=float(result.coordinates[i, 0]),
                y=float(result.coordinates[i, 1]),
                intensity=float(result.intensities[i]),
                frame=f
            )
            frame_localizations.append(particle)
            all_localizations_flat.append({
                'frame': f,
                'x': float(result.coordinates[i, 0]),
                'y': float(result.coordinates[i, 1]),
                'intensity': float(result.intensities[i])
            })
        
        all_localizations_by_frame.append(frame_localizations)
        print(f"    Frame {f+1}/{frames}: {n_particles} particles detected")
    
    print(f"    Total: {len(all_localizations_flat)} localizations")
    
    # Step 5: Particle tracking
    print("\n[5/6] Tracking particles...")
    if len(all_localizations_flat) > 0:
        params = TrackParams(
            max_displacement=10.0,
            memory=2,
            gap_closing=2
        )
        tracks = track_particles(all_localizations_by_frame, params)
        print(f"    {len(tracks)} tracks identified")
        
        # Print track statistics
        track_lengths = [len(t.particles) for t in tracks]
        print(f"    Track length: min={min(track_lengths)}, max={max(track_lengths)}, mean={np.mean(track_lengths):.1f}")
    else:
        print("    No particles to track")
        tracks = []
    
    # Step 6: LAFM reconstruction
    print("\n[6/6] Generating LAFM reconstruction...")
    if len(all_localizations_flat) > 0:
        # Convert localizations to arrays for LAFM
        coords = np.array([[loc['x'], loc['y']] for loc in all_localizations_flat])
        intensities = np.array([loc['intensity'] for loc in all_localizations_flat])
        
        # Splat Gaussians
        lafm_image = splat_localizations_gpu(
            coords,
            intensities,
            output_shape=(height, width),
            sigma=2.0,
            use_gpu=False  # Use CPU for this example
        )
        print(f"    LAFM image generated: {lafb_image.shape}")
        print(f"    Intensity range: [{lafb_image.min():.1f}, {lafb_image.max():.1f}]")
    else:
        print("    No localizations for LAFM")
        lafm_image = None
    
    # Summary
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print(f"Input: {frames} frames, {height}x{width} pixels")
    print(f"Output: {len(all_localizations_flat)} localizations, {len(tracks)} tracks")
    if lafm_image is not None:
        print(f"LAFM: {lafb_image.shape} reconstruction")
    print("\nTo export results, use:")
    print("  from nanolocz.io.store import NanoLoczStore")
    print("  with NanoLoczStore('results.zarr', mode='w') as store:")
    print("      store.save_movie(movie, metadata={})")
    print("      store.save_localizations(all_localizations_flat)")
    print("      store.save_tracks(tracks)")
    print("="*60)
    
    return {
        'movie': movie,
        'leveled': leveled_movie,
        'filtered': filtered_movie,
        'localizations': all_localizations_flat,
        'tracks': tracks,
        'lafm': lafm_image
    }


if __name__ == '__main__':
    results = main()
