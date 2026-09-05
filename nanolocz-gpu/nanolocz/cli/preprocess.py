"""Preprocess subcommand for NanoLocz CLI."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def cmd_preprocess(args: "Namespace") -> int:
    """Execute the preprocess command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Import here to avoid circular imports
        from nanolocz.io import open_nanolocz
        from nanolocz.core.leveling import batch_level_movie
        from nanolocz.core.filters import gaussian_filter, median_filter, uniform_filter
        from nanolocz.core.filters import remove_scars as remove_scars_func
        from nanolocz.gpu.backend import get_backend_context, Backend
        from nanolocz.io.store import NanoLoczStore
        
        if args.verbose:
            print(f"Preprocessing: {args.input}")
        
        # Load input file
        frame, meta = open_nanolocz(args.input)
        
        if args.verbose:
            print(f"Loaded {frame.data.shape} data with {len(frame.data)} frames")
        
        # Setup backend
        backend_type = Backend.CUDA if args.gpu else Backend.CPU
        ctx = get_backend_context(backend=backend_type, precision=args.precision)
        
        data = frame.data
        
        # Apply leveling
        if args.leveling != "none":
            if args.verbose:
                print(f"Applying {args.leveling} leveling...")
            data, _ = batch_level_movie(data, method=args.leveling)
        
        # Apply filter
        if args.filter != "none":
            if args.verbose:
                print(f"Applying {args.filter} filter (sigma={args.filter_sigma})...")
            if args.filter == "gaussian":
                data = gaussian_filter(data, sigma=args.filter_sigma)
            elif args.filter == "median":
                data = median_filter(data, size=int(args.filter_sigma))
            elif args.filter == "uniform":
                data = uniform_filter(data, size=int(args.filter_sigma))
        
        # Remove scars
        if args.remove_scars:
            if args.verbose:
                print("Removing scan line scars...")
            data = remove_scars_func(data)
        
        # Update frame with processed data
        from nanolocz.core.types import Frame
        processed_frame = Frame(data=data, meta=meta)
        
        # Save to output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with NanoLoczStore.open(args.output, mode="w") as store:
            store.save_movie(processed_frame)
        
        if args.verbose:
            print(f"Saved preprocessed data to {args.output}")
        
        return 0
        
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error during preprocessing: {e}", file=sys.stderr)
        return 1
