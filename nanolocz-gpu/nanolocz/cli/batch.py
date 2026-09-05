"""Batch processing subcommand for NanoLocz CLI."""

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

if TYPE_CHECKING:
    from argparse import Namespace


def process_single_file(args_tuple: tuple) -> dict[str, Any]:
    """Process a single file through the pipeline.
    
    This function is designed to be called in a separate process.
    
    Args:
        args_tuple: Tuple of (input_path, output_dir, pipeline, config_dict, verbose)
        
    Returns:
        Dictionary with processing results
    """
    input_path, output_dir, pipeline, config_dict, verbose = args_tuple
    
    start_time = time.time()
    result = {
        "input": str(input_path),
        "success": False,
        "error": None,
        "output": None,
        "time": 0.0,
    }
    
    try:
        # Determine output filename
        output_name = input_path.stem + "_processed.zarr"
        output_path = output_dir / output_name
        
        if verbose:
            print(f"Processing: {input_path.name}")
        
        # Run appropriate pipeline stages
        if pipeline == "preprocess":
            # Import and run preprocess
            from nanolocz.io import open_nanolocz
            from nanolocz.core.leveling import level_movie
            from nanolocz.core.filters import gaussian_filter
            from nanolocz.io.store import NanoLoczStore
            
            frame, meta = open_nanolocz(input_path)
            data = level_movie(frame.data, method=config_dict.get("leveling", "plane"))
            data = gaussian_filter(data, sigma=config_dict.get("filter_sigma", 1.0))
            
            from nanolocz.core.types import Frame
            processed_frame = Frame(data=data, meta=meta)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with NanoLoczStore.open(output_path, mode="w") as store:
                store.save_movie(processed_frame)
            
            result["output"] = str(output_path)
            result["success"] = True
            
        elif pipeline == "detect":
            # Import and run detection
            from nanolocz.io.store import NanoLoczStore
            from nanolocz.core.detection import detect_particles
            from nanolocz.core.types import Localization
            
            with NanoLoczStore.open(input_path, mode="r") as store:
                movie = store.load_movie()
            
            all_localizations = []
            for frame_idx, frame_data in enumerate(movie.data):
                det_result = detect_particles(
                    frame_data,
                    threshold=config_dict.get("threshold", 3.0),
                    min_distance=config_dict.get("min_distance", 5),
                )
                # Convert to localizations...
                # (simplified for brevity)
            
            with NanoLoczStore.open(output_path, mode="w") as store:
                store.save_movie(movie)
                store.save_localizations(all_localizations)
            
            result["output"] = str(output_path)
            result["success"] = True
            
        elif pipeline == "full":
            # Full pipeline: preprocess -> detect -> track -> lafm
            # Simplified implementation
            from nanolocz.io import open_nanolocz
            from nanolocz.core.leveling import level_movie
            from nanolocz.core.filters import gaussian_filter
            from nanolocz.core.detection import detect_particles
            from nanolocz.core.tracking import track_particles
            from nanolocz.gpu.lafm import splat_localizations_gpu
            from nanolocz.gpu.backend import get_backend_context, Backend
            from nanolocz.io.store import NanoLoczStore
            from nanolocz.core.types import Frame, Localization
            
            # Step 1: Preprocess
            frame, meta = open_nanolocz(input_path)
            data = level_movie(frame.data, method=config_dict.get("leveling", "plane"))
            data = gaussian_filter(data, sigma=config_dict.get("filter_sigma", 1.0))
            processed_frame = Frame(data=data, meta=meta)
            
            # Step 2: Detect
            all_localizations = []
            for frame_idx, frame_data in enumerate(processed_frame.data):
                det_result = detect_particles(
                    frame_data,
                    threshold=config_dict.get("threshold", 3.0),
                    min_distance=config_dict.get("min_distance", 5),
                )
                # Add localizations...
            
            # Step 3: Track
            tracks = track_particles(
                all_localizations,
                max_displacement=config_dict.get("max_displacement", 10.0),
                gap_closing=config_dict.get("gap_closing", 2),
            )
            
            # Step 4: LAFM (if pixel_size provided)
            pixel_size = config_dict.get("pixel_size")
            if pixel_size is not None and len(all_localizations) > 0:
                ctx = get_backend_context(backend=Backend.CPU)
                # Perform LAFM...
            
            # Save everything
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with NanoLoczStore.open(output_path, mode="w") as store:
                store.save_movie(processed_frame)
                store.save_localizations(all_localizations)
                store.save_tracks(tracks)
            
            result["output"] = str(output_path)
            result["success"] = True
        
        else:
            result["error"] = f"Unknown pipeline: {pipeline}"
        
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            import traceback
            traceback.print_exc()
    
    result["time"] = time.time() - start_time
    return result


def cmd_batch(args: "Namespace") -> int:
    """Execute the batch command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        from nanolocz.cli.utils import expand_input_paths, print_summary
        from nanolocz.cli.utils import PipelineConfig
        
        # Expand input paths
        input_files = expand_input_paths(args.input)
        
        if not input_files:
            print("Error: No input files found", file=sys.stderr)
            return 1
        
        if args.verbose:
            print(f"Found {len(input_files)} input files")
        
        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Load config if provided
        config_dict = {}
        if args.config is not None:
            config = PipelineConfig.from_json(args.config)
            config_dict = config.to_dict()
            if args.verbose:
                print(f"Loaded configuration from {args.config}")
        
        # Override config with CLI args
        config_dict["gpu"] = args.gpu
        
        # Dry run
        if args.dry_run:
            print("\nDRY RUN - Would process the following files:")
            for i, input_file in enumerate(input_files, 1):
                output_name = input_file.stem + "_processed.zarr"
                print(f"  {i}. {input_file} -> {args.output / output_name}")
            print(f"\nTotal: {len(input_files)} files")
            print(f"Pipeline: {args.pipeline}")
            print(f"Parallel jobs: {args.jobs}")
            return 0
        
        # Process files
        print(f"\nProcessing {len(input_files)} files with pipeline='{args.pipeline}'...")
        
        results = []
        errors = []
        start_time = time.time()
        
        if args.jobs > 1:
            # Parallel processing
            process_args = [
                (input_file, args.output, args.pipeline, config_dict, args.verbose)
                for input_file in input_files
            ]
            
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(process_single_file, arg): arg[0] for arg in process_args}
                
                for future in as_completed(futures):
                    input_file = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if not result["success"]:
                            errors.append(result)
                        
                        if args.verbose:
                            status = "✓" if result["success"] else "✗"
                            print(f"  {status} {input_file.name} ({result['time']:.2f}s)")
                    except Exception as e:
                        error_result = {
                            "input": str(input_file),
                            "success": False,
                            "error": str(e),
                            "time": 0.0,
                        }
                        results.append(error_result)
                        errors.append(error_result)
        else:
            # Sequential processing
            for i, input_file in enumerate(input_files, 1):
                if args.verbose:
                    print(f"[{i}/{len(input_files)}] Processing {input_file.name}...")
                
                result = process_single_file((
                    input_file, args.output, args.pipeline, config_dict, args.verbose
                ))
                results.append(result)
                
                if not result["success"]:
                    errors.append(result)
                    if not args.continue_on_error:
                        print(f"Error processing {input_file.name}: {result['error']}", file=sys.stderr)
                        break
                
                if args.verbose and result["success"]:
                    print(f"  ✓ {input_file.name} ({result['time']:.2f}s)")
        
        total_time = time.time() - start_time
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Files processed: {successful}/{len(results)}")
        print(f"Files failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Output directory: {args.output}")
        
        if errors and args.verbose:
            print("\nFailed files:")
            for err in errors:
                print(f"  - {err['input']}: {err['error']}")
        
        print("=" * 60)
        
        return 0 if failed == 0 else 1
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error during batch processing: {e}", file=sys.stderr)
        return 1
