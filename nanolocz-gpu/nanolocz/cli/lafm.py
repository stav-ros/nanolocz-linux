"""LAFM subcommand for NanoLocz CLI."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def cmd_lafm(args: "Namespace") -> int:
    """Execute the LAFM reconstruction command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Import here to avoid circular imports
        from nanolocz.io.store import NanoLoczStore
        from nanolocz.gpu.lafm import splat_localizations_gpu, compute_frc_gpu, frc_resolution
        from nanolocz.gpu.backend import get_backend_context, Backend
        
        if args.verbose:
            print(f"Performing LAFM reconstruction on: {args.input}")
        
        # Load input data
        with NanoLoczStore.open(args.input, mode="r") as store:
            localizations = store.load_localizations()
            
            if args.verbose:
                print(f"Loaded {len(localizations)} localizations")
        
        # Setup backend
        backend_type = Backend.CUDA if args.gpu else Backend.CPU
        ctx = get_backend_context(backend=backend_type, precision=args.precision)
        
        # Calculate output grid size based on pixel size and localization bounds
        import numpy as np
        
        xs = [loc.x for loc in localizations]
        ys = [loc.y for loc in localizations]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add some padding
        padding = args.sigma * 5
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        
        # Calculate grid dimensions
        width_px = int(np.ceil((x_max - x_min) / args.pixel_size))
        height_px = int(np.ceil((y_max - y_min) / args.pixel_size))
        
        if args.verbose:
            print(f"Reconstruction grid: {width_px} x {height_px} pixels")
            print(f"Pixel size: {args.pixel_size} nm, Sigma: {args.sigma} nm")
        
        # Perform LAFM splatting
        if args.verbose:
            print("Splatting localizations...")
        
        reconstructed_map = splat_localizations_gpu(
            localizations=localizations,
            pixel_size=args.pixel_size,
            sigma=args.sigma,
            shape=(height_px, width_px),
            offset=(x_min, y_min),
            backend_ctx=ctx,
        )
        
        # Convert to CPU for saving
        reconstructed_map_cpu = ctx.to_cpu(reconstructed_map)
        
        # Calculate FRC if requested
        frc_result = None
        resolution = None
        if args.frc:
            if args.verbose:
                print("Calculating FRC resolution estimate...")
            
            # Split localizations into two halves for FRC
            n_half = len(localizations) // 2
            half1 = localizations[:n_half]
            half2 = localizations[n_half:]
            
            # Splat both halves
            map1 = splat_localizations_gpu(
                localizations=half1,
                pixel_size=args.pixel_size,
                sigma=args.sigma,
                shape=(height_px, width_px),
                offset=(x_min, y_min),
                backend_ctx=ctx,
            )
            map2 = splat_localizations_gpu(
                localizations=half2,
                pixel_size=args.pixel_size,
                sigma=args.sigma,
                shape=(height_px, width_px),
                offset=(x_min, y_min),
                backend_ctx=ctx,
            )
            
            # Compute FRC
            frc_curve, frequencies = compute_frc_gpu(map1, map2, backend_ctx=ctx)
            resolution = frc_resolution(frc_curve, frequencies, threshold=0.5)
            
            frc_result = {
                "curve": ctx.to_cpu(frc_curve).tolist(),
                "frequencies": ctx.to_cpu(frequencies).tolist(),
                "resolution_nm": float(resolution) if resolution is not None else None,
            }
            
            if args.verbose:
                if resolution is not None:
                    print(f"FRC resolution at 0.5 threshold: {resolution:.3f} nm")
                else:
                    print("FRC resolution could not be estimated")
        
        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with NanoLoczStore.open(args.output, mode="w") as store:
            # Save original localizations
            store.save_localizations(localizations)
            # Save reconstructed map
            store.save_reconstruction(reconstructed_map_cpu, pixel_size=args.pixel_size)
            # Save FRC results if available
            if frc_result is not None:
                import json
                frc_path = args.output.with_suffix(".frc.json")
                with open(frc_path, "w") as f:
                    json.dump(frc_result, f, indent=2)
                if args.verbose:
                    print(f"Saved FRC results to {frc_path}")
        
        if args.verbose:
            print(f"Saved LAFM reconstruction to {args.output}")
        
        return 0
        
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error during LAFM reconstruction: {e}", file=sys.stderr)
        return 1
