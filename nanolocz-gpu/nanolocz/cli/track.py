"""Track subcommand for NanoLocz CLI."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def cmd_track(args: "Namespace") -> int:
    """Execute the track command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Import here to avoid circular imports
        from nanolocz.io.store import NanoLoczStore
        from nanolocz.core.tracking import track_particles
        from nanolocz.gpu.backend import get_backendContext, Backend
        
        if args.verbose:
            print(f"Tracking particles in: {args.input}")
        
        # Load input data
        with NanoLoczStore.open(args.input, mode="r") as store:
            localizations = store.load_localizations()
            
            if args.verbose:
                print(f"Loaded {len(localizations)} localizations")
        
        # Setup backend (tracking is CPU-based)
        backend_type = Backend.CPU  # Tracking is typically CPU
        ctx = get_backend_context(backend=backend_type, precision=args.precision)
        
        # Track particles
        if args.verbose:
            print(f"Tracking with max_displacement={args.max_displacement}, gap_closing={args.gap_closing}...")
        
        tracks = track_particles(
            localizations,
            max_displacement=args.max_displacement,
            gap_closing=args.gap_closing,
            memory=args.memory,
        )
        
        if args.verbose:
            print(f"Generated {len(tracks)} tracks")
        
        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with NanoLoczStore.open(args.output, mode="w") as store:
            # Save original localizations
            store.save_localizations(localizations)
            # Save tracks
            store.save_tracks(tracks)
        
        if args.verbose:
            print(f"Saved tracks to {args.output}")
        
        return 0
        
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error during tracking: {e}", file=sys.stderr)
        return 1
