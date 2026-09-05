"""Main CLI entry point for NanoLocz."""

import argparse
import sys
from pathlib import Path

from nanolocz.cli.preprocess import cmd_preprocess
from nanolocz.cli.detect import cmd_detect
from nanolocz.cli.track import cmd_track
from nanolocz.cli.lafm import cmd_lafm
from nanolocz.cli.batch import cmd_batch


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="nanolocz",
        description="NanoLocz: GPU-accelerated AFM analysis pipeline",
        epilog="Use '%(prog)s <command> --help' for more information on a specific command.",
    )
    
    # Add common arguments
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        required=True,
    )
    
    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess AFM movies (leveling, filtering, scar removal)",
        description="Apply preprocessing operations to AFM movie data.",
    )
    preprocess_parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input file path (supported formats: .gwy, .h5-jpk, .spm, .jpk, .ibw, .asd, .tiff)",
    )
    preprocess_parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output file path (.zarr format)",
    )
    preprocess_parser.add_argument(
        "--leveling",
        choices=["none", "line", "plane", "weighted"],
        default="plane",
        help="Leveling method (default: plane)",
    )
    preprocess_parser.add_argument(
        "--filter",
        choices=["none", "gaussian", "median", "uniform"],
        default="gaussian",
        help="Filter type (default: gaussian)",
    )
    preprocess_parser.add_argument(
        "--filter-sigma",
        type=float,
        default=1.0,
        help="Filter sigma/size (default: 1.0)",
    )
    preprocess_parser.add_argument(
        "--remove-scars",
        action="store_true",
        help="Enable scar removal",
    )
    preprocess_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    preprocess_parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default="float64",
        help="Numerical precision (default: float64)",
    )
    preprocess_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    preprocess_parser.set_defaults(func=cmd_preprocess)
    
    # Detect subcommand
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect particles in preprocessed data",
        description="Detect particles using local maxima and prominence filtering.",
    )
    detect_parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input .zarr file from preprocessing step",
    )
    detect_parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output .zarr file with detections",
    )
    detect_parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Prominence threshold in standard deviations (default: 3.0)",
    )
    detect_parser.add_argument(
        "--min-distance",
        type=int,
        default=5,
        help="Minimum distance between particles in pixels (default: 5)",
    )
    detect_parser.add_argument(
        "--mask",
        type=Path,
        help="Optional mask file to restrict detection area",
    )
    detect_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    detect_parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default="float64",
        help="Numerical precision (default: float64)",
    )
    detect_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    detect_parser.set_defaults(func=cmd_detect)
    
    # Track subcommand
    track_parser = subparsers.add_parser(
        "track",
        help="Track particles across frames",
        description="Perform single-particle tracking with gap reconnection.",
    )
    track_parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input .zarr file from detection step",
    )
    track_parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output .zarr file with tracks",
    )
    track_parser.add_argument(
        "--max-displacement",
        type=float,
        default=10.0,
        help="Maximum displacement between frames in pixels (default: 10.0)",
    )
    track_parser.add_argument(
        "--gap-closing",
        type=int,
        default=2,
        help="Maximum number of frames to close gaps (default: 2)",
    )
    track_parser.add_argument(
        "--memory",
        type=int,
        default=3,
        help="Memory for track linking (default: 3)",
    )
    track_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    track_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    track_parser.set_defaults(func=cmd_track)
    
    # LAFM subcommand
    lafm_parser = subparsers.add_parser(
        "lafm",
        help="LAFM reconstruction and FRC calculation",
        description="Perform Localization-based AFM super-resolution reconstruction.",
    )
    lafm_parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input .zarr file with tracks or localizations",
    )
    lafm_parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output .zarr file with reconstructed map",
    )
    lafm_parser.add_argument(
        "--pixel-size",
        type=float,
        required=True,
        help="Pixel size in nanometers (required)",
    )
    lafm_parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Gaussian sigma for splatting in nm (default: 0.5)",
    )
    lafm_parser.add_argument(
        "--frc",
        action="store_true",
        help="Calculate FRC resolution estimate",
    )
    lafm_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    lafm_parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default="float64",
        help="Numerical precision (default: float64)",
    )
    lafm_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    lafm_parser.set_defaults(func=cmd_lafm)
    
    # Batch subcommand
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process multiple files",
        description="Process multiple input files with a configurable pipeline.",
    )
    batch_parser.add_argument(
        "-i", "--input",
        required=True,
        nargs="+",
        type=Path,
        help="Input file paths or directory (glob patterns supported)",
    )
    batch_parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output directory for results",
    )
    batch_parser.add_argument(
        "-c", "--config",
        type=Path,
        help="JSON configuration file with pipeline parameters",
    )
    batch_parser.add_argument(
        "--pipeline",
        choices=["preprocess", "detect", "track", "lafm", "full"],
        default="full",
        help="Pipeline stages to run (default: full)",
    )
    batch_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)",
    )
    batch_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining files if one fails",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without executing",
    )
    batch_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    batch_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    batch_parser.set_defaults(func=cmd_batch)
    
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Call the appropriate command function
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
