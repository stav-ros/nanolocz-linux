"""Detect subcommand for NanoLocz CLI."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def cmd_detect(args: "Namespace") -> int:
    """Execute the detect command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Import here to avoid circular imports
        from nanolocz.io.store import NanoLoczStore
        from nanolocz.core.detection import detect_particles
        from nanolocz.gpu.backend import get_backend_context, Backend
        
        if args.verbose:
            print(f"Detecting particles in: {args.input}")
        
        # Load input data
        with NanoLoczStore.open(args.input, mode="r") as store:
            movie = store.load_movie()
            
            if args.verbose:
                print(f"Loaded movie with {len(movie.data)} frames")
        
        # Setup backend
        backend_type = Backend.CUDA if args.gpu else Backend.CPU
        ctx = get_backend_context(backend=backend_type, precision=args.precision)
        
        # Load optional mask
        mask = None
        if args.mask is not None:
            import numpy as np
            if args.mask.suffix == ".npy":
                mask = np.load(args.mask)
            else:
                # Try to load from zarr
                with NanoLoczStore.open(args.mask, mode="r") as mask_store:
                    mask = mask_store.load_mask()
        
        # Detect particles
        if args.verbose:
            print(f"Detecting with threshold={args.threshold}, min_distance={args.min_distance}...")
        
        results = []
        for i, frame_data in enumerate(movie.data):
            if args.verbose and (i + 1) % 10 == 0:
                print(f"  Processing frame {i + 1}/{len(movie.data)}")
            
            # Apply detection to this frame
            from nanolocz.core.detection import find_local_maxima, calculate_prominence
            from nanolocz.core.types import DetectionResult
            
            # Simple detection per frame
            frame_result = detect_particles(
                frame_data,
                threshold=args.threshold,
                min_distance=args.min_distance,
                mask=mask,
            )
            results.append(frame_result)
        
        # Combine results across frames
        from nanolocz.core.types import Localization
        import numpy as np
        
        all_localizations = []
        for frame_idx, result in enumerate(results):
            if hasattr(result, 'positions') and result.positions is not None:
                n_particles = len(result.positions)
                for i in range(n_particles):
                    loc = Localization(
                        x=result.positions[i, 1] if result.positions.shape[1] > 1 else result.positions[i, 0],
                        y=result.positions[i, 0] if result.positions.shape[1] > 1 else 0,
                        frame=frame_idx,
                        intensity=result.intensities[i] if hasattr(result, 'intensities') and result.intensities is not None else 0.0,
                        sigma=result.sigmas[i] if hasattr(result, 'sigmas') and result.sigmas is not None else 1.0,
                    )
                    all_localizations.append(loc)
        
        if args.verbose:
            print(f"Detected {len(all_localizations)} total localizations")
        
        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with NanoLoczStore.open(args.output, mode="w") as store:
            # Save original movie
            store.save_movie(movie)
            # Save localizations
            store.save_localizations(all_localizations)
        
        if args.verbose:
            print(f"Saved detections to {args.output}")
        
        return 0
        
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error during detection: {e}", file=sys.stderr)
        return 1
