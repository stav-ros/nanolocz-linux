"""Shared utilities for CLI commands."""

import json
from pathlib import Path
from typing import Any

from nanolocz.core.config import PipelineConfig


def load_config(config_path: Path | None, cli_args: Any) -> PipelineConfig:
    """Load and merge configuration from file and CLI arguments.

    CLI arguments take precedence over config file values.

    Args:
        config_path: Path to JSON config file (optional)
        cli_args: Parsed CLI arguments

    Returns:
        Merged PipelineConfig
    """
    # Start with defaults
    config = PipelineConfig()

    # Load from file if provided
    if config_path is not None:
        file_config = PipelineConfig.from_json(config_path)
        # Merge file config into default
        for field_name in config.__dataclass_fields__:
            file_value = getattr(file_config, field_name)
            if file_value is not None:
                setattr(config, field_name, file_value)

    # Override with CLI arguments
    cli_mapping = {
        "leveling": "leveling",
        "filter_type": "filter",
        "filter_sigma": "filter_sigma",
        "remove_scars": "remove_scars",
        "threshold": "threshold",
        "min_distance": "min_distance",
        "max_displacement": "max_displacement",
        "gap_closing": "gap_closing",
        "memory": "memory",
        "pixel_size": "pixel_size",
        "sigma": "sigma",
        "frc": "frc",
        "gpu": "gpu",
        "precision": "precision",
    }

    for config_field, cli_attr in cli_mapping.items():
        cli_value = getattr(cli_args, cli_attr, None)
        if cli_value is not None:
            setattr(config, config_field, cli_value)

    return config


def expand_input_paths(input_paths: list[Path]) -> list[Path]:
    """Expand input paths including glob patterns and directories.

    Args:
        input_paths: List of input paths (may include globs or directories)

    Returns:
        Expanded list of file paths
    """
    expanded = []

    for path in input_paths:
        # Convert to string for glob
        path_str = str(path)

        # Check if it's a directory
        if path.is_dir():
            # Add all supported files in directory
            supported_extensions = [".gwy", ".h5-jpk", ".spm", ".jpk", ".ibw", ".asd", ".tiff", ".tif"]
            for ext in supported_extensions:
                expanded.extend(path.glob(f"*{ext}"))
                expanded.extend(path.glob(f"*{ext.upper()}"))
        # Check if it's a glob pattern
        elif "*" in path_str or "?" in path_str:
            expanded.extend(Path(".").glob(path_str))
        else:
            # Regular file
            if path.exists():
                expanded.append(path)

    return sorted(set(expanded))


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def print_summary(results: dict[str, Any]) -> None:
    """Print processing summary."""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    if "files_processed" in results:
        print(f"Files processed: {results['files_processed']}")
    if "files_failed" in results:
        print(f"Files failed: {results['files_failed']}")
    if "total_time" in results:
        print(f"Total time: {results['total_time']:.2f}s")
    if "output_dir" in results:
        print(f"Output directory: {results['output_dir']}")

    print("=" * 60)
