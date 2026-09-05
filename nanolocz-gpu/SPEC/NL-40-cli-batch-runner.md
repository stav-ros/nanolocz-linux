# NL-40 — Headless CLI and batch runner

**Phase:** P4 — Interface and ship  
**Depends:** NL-16 (Detection), NL-23 (LAFM splat/FRC)  
**State:** not_started → in_progress  
**Agent budget:** 1 session  

---

## Acceptance criteria

1. **CLI entry point**: `python -m nanolocz` or `nanolocz` command available after install
2. **Subcommands implemented**:
   - `preprocess`: Leveling, filtering, scar removal on AFM movies
   - `detect`: Particle detection with configurable parameters
   - `track`: Single-particle tracking
   - `lafm`: LAFM reconstruction and FRC calculation
   - `batch`: Process multiple files with a job queue
3. **Common options** across all commands:
   - `--input` / `-i`: Input file(s) or directory
   - `--output` / `-o`: Output path (.zarr by default)
   - `--config` / `-c`: Optional JSON/YAML config file
   - `--verbose` / `-v`: Verbose output
   - `--gpu` / `--no-gpu`: Force GPU/CPU backend
   - `--precision`: float32 or float64
4. **Batch processing**:
   - Queue management for multiple input files
   - Progress reporting with ETA
   - Error handling with continue-on-failure option
   - Summary report at completion
5. **Configuration files**: Support JSON format for saving/loading parameter presets
6. **Tests**: 15+ tests covering CLI parsing, subcommand execution, batch processing
7. **Documentation**: Usage examples in README, help text for all commands

---

## Implementation plan

### 1. Create CLI module structure
```
nanolocz/
  cli/
    __init__.py       # Entry point
    main.py           # Argparse setup
    preprocess.py     # Preprocessing subcommand
    detect.py         # Detection subcommand
    track.py          # Tracking subcommand
    lafm.py           # LAFM subcommand
    batch.py          # Batch processing
    utils.py          # Shared utilities
```

### 2. Implement argparse structure
- Main parser with subparsers
- Each subcommand has its own argument parser
- Shared argument groups for common options

### 3. Implement subcommands
Each subcommand:
- Parses arguments
- Loads input data using existing openers
- Calls appropriate core functions
- Saves results to .zarr or other formats
- Returns exit codes (0=success, 1=error)

### 4. Batch processor
- JobQueue class for managing multiple files
- Worker function for processing single file
- Progress bar with tqdm
- Error collection and summary

### 5. Configuration system
- Config dataclass with all parameters
- Load/save JSON functions
- Merge config file with CLI args (CLI wins)

### 6. Tests
- Test CLI argument parsing
- Test each subcommand with sample data
- Test batch processing with multiple files
- Test config file loading
- Test error handling

---

## File changes

**New files:**
- `nanolocz/cli/__init__.py`
- `nanolocz/cli/main.py`
- `nanolocz/cli/preprocess.py`
- `nanolocz/cli/detect.py`
- `nanolocz/cli/track.py`
- `nanolocz/cli/lafm.py`
- `nanolocz/cli/batch.py`
- `nanolocz/cli/utils.py`
- `SPEC/NL-40-cli-batch-runner.md` (this file)
- `tests/test_cli_nl40.py`

**Modified files:**
- `pyproject.toml` (add console_scripts entry point)
- `nanolocz/__init__.py` (export CLI if needed)
- `README.md` (add CLI usage section)
- `STATUS.md` (mark NL-40 done)

---

## Evidence of completion

- [ ] All 15+ tests passing
- [ ] `python -m nanolocz --help` works
- [ ] Each subcommand `--help` shows proper documentation
- [ ] Batch processing demonstrated on example data
- [ ] Config file example in `examples/cli_config.json`
- [ ] SESSIONS/2026-09-XX-NL-40.md handoff created
- [ ] STATUS.md updated with NL-40 done

---

## Example usage

```bash
# Preprocess a movie
nanolocz preprocess -i movie.gwy -o processed.zarr --leveling plane --filter gaussian

# Detect particles
nanolocz detect -i processed.zarr -o detected.zarr --threshold 3.5 --min-distance 5

# Track particles
nanolocz track -i detected.zarr -o tracked.zarr --max-displacement 10

# LAFM reconstruction
nanolocz lafm -i tracked.zarr -o lafm.zarr --pixel-size 0.5

# Full pipeline in one command
nanolocz preprocess -i movie.gwy -o final.zarr --pipeline full

# Batch process multiple files
nanolocz batch -i ./data/*.gwy -o ./results/ --config my_pipeline.json --jobs 4
```

---

## Notes

- Keep CLI stateless: no hidden state between runs
- All operations should be reproducible with same inputs and config
- Exit codes matter for scripting integration
- Consider adding `--dry-run` option for batch jobs
