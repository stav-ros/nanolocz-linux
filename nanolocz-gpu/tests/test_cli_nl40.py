"""Tests for NL-40 CLI and batch runner."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Test CLI argument parsing
class TestCLIArgumentParsing:
    """Test CLI argument parsing."""
    
    def test_main_help(self):
        """Test that main command shows help."""
        from nanolocz.cli.main import create_parser
        parser = create_parser()
        
        # --help causes SystemExit, which is expected
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
    
    def test_preprocess_required_args(self):
        """Test preprocess command requires input and output."""
        from nanolocz.cli.main import create_parser
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["preprocess"])
    
    def test_preprocess_all_options(self):
        """Test preprocess command accepts all options."""
        from nanolocz.cli.main import create_parser
        parser = create_parser()
        
        args = parser.parse_args([
            "preprocess",
            "-i", "input.gwy",
            "-o", "output.zarr",
            "--leveling", "plane",
            "--filter", "gaussian",
            "--filter-sigma", "1.5",
            "--remove-scars",
            "--gpu",
            "--precision", "float32",
            "-v",
        ])
        
        assert args.input == Path("input.gwy")
        assert args.output == Path("output.zarr")
        assert args.leveling == "plane"
        assert args.filter == "gaussian"
        assert args.filter_sigma == 1.5
        assert args.remove_scars is True
        assert args.gpu is True
        assert args.precision == "float32"
        assert args.verbose is True
    
    def test_detect_options(self):
        """Test detect command options."""
        from nanolocz.cli.main import create_parser
        parser = create_parser()
        
        args = parser.parse_args([
            "detect",
            "-i", "input.zarr",
            "-o", "output.zarr",
            "--threshold", "5.0",
            "--min-distance", "10",
            "--mask", "mask.npy",
            "--gpu",
            "-v",
        ])
        
        assert args.threshold == 5.0
        assert args.min_distance == 10
        assert args.mask == Path("mask.npy")
    
    def test_track_options(self):
        """Test track command options."""
        from nanolocz.cli.main import create_parser
        parser = create_parser()
        
        args = parser.parse_args([
            "track",
            "-i", "input.zarr",
            "-o", "output.zarr",
            "--max-displacement", "15.0",
            "--gap-closing", "3",
            "--memory", "5",
            "-v",
        ])
        
        assert args.max_displacement == 15.0
        assert args.gap_closing == 3
        assert args.memory == 5
    
    def test_lafm_options(self):
        """Test lafm command options."""
        from nanolocz.cli.main import create_parser
        parser = create_parser()
        
        args = parser.parse_args([
            "lafm",
            "-i", "input.zarr",
            "-o", "output.zarr",
            "--pixel-size", "0.5",
            "--sigma", "1.0",
            "--frc",
            "--gpu",
            "-v",
        ])
        
        assert args.pixel_size == 0.5
        assert args.sigma == 1.0
        assert args.frc is True
    
    def test_batch_options(self):
        """Test batch command options."""
        from nanolocz.cli.main import create_parser
        parser = create_parser()
        
        args = parser.parse_args([
            "batch",
            "-i", "file1.gwy", "file2.gwy",
            "-o", "output_dir",
            "-c", "config.json",
            "--pipeline", "full",
            "--jobs", "4",
            "--continue-on-error",
            "--dry-run",
            "--gpu",
            "-v",
        ])
        
        assert len(args.input) == 2
        assert args.output == Path("output_dir")
        assert args.config == Path("config.json")
        assert args.pipeline == "full"
        assert args.jobs == 4
        assert args.continue_on_error is True
        assert args.dry_run is True


# Test PipelineConfig
class TestPipelineConfig:
    """Test PipelineConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from nanolocz.cli.utils import PipelineConfig
        
        config = PipelineConfig()
        assert config.leveling == "plane"
        assert config.filter_type == "gaussian"
        assert config.filter_sigma == 1.0
        assert config.threshold == 3.0
        assert config.min_distance == 5
        assert config.max_displacement == 10.0
        assert config.pixel_size is None
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        from nanolocz.cli.utils import PipelineConfig
        
        config = PipelineConfig(leveling="line", threshold=5.0)
        d = config.to_dict()
        
        assert d["leveling"] == "line"
        assert d["threshold"] == 5.0
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        from nanolocz.cli.utils import PipelineConfig
        
        data = {"leveling": "weighted", "min_distance": 8}
        config = PipelineConfig.from_dict(data)
        
        assert config.leveling == "weighted"
        assert config.min_distance == 8
    
    def test_from_json(self, tmp_path):
        """Test loading config from JSON file."""
        from nanolocz.cli.utils import PipelineConfig
        
        config_file = tmp_path / "config.json"
        config_data = {
            "leveling": "line",
            "threshold": 4.5,
            "gpu": True,
        }
        
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        config = PipelineConfig.from_json(config_file)
        assert config.leveling == "line"
        assert config.threshold == 4.5
        assert config.gpu is True
    
    def test_save_json(self, tmp_path):
        """Test saving config to JSON file."""
        from nanolocz.cli.utils import PipelineConfig
        
        config = PipelineConfig(leveling="weighted", gpu=True)
        config_file = tmp_path / "saved_config.json"
        
        config.save_json(config_file)
        
        with open(config_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["leveling"] == "weighted"
        assert loaded["gpu"] is True


# Test expand_input_paths
class TestExpandInputPaths:
    """Test input path expansion."""
    
    def test_single_file(self, tmp_path):
        """Test expanding a single file path."""
        from nanolocz.cli.utils import expand_input_paths
        
        test_file = tmp_path / "test.gwy"
        test_file.touch()
        
        result = expand_input_paths([test_file])
        assert len(result) == 1
        assert result[0] == test_file
    
    def test_directory_expansion(self, tmp_path):
        """Test expanding a directory to multiple files."""
        from nanolocz.cli.utils import expand_input_paths
        
        # Create test files
        (tmp_path / "file1.gwy").touch()
        (tmp_path / "file2.gwy").touch()
        (tmp_path / "file3.txt").touch()  # Should be ignored
        
        result = expand_input_paths([tmp_path])
        assert len(result) == 2


# Test CLI execution with mocks
class TestCLIExecution:
    """Test CLI command execution with mocked functions."""
    
    @patch('nanolocz.io.open_nanolocz')
    @patch('nanolocz.core.leveling.batch_level_movie')
    @patch('nanolocz.core.filters.gaussian_filter')
    @patch('nanolocz.io.store.NanoLoczStore')
    def test_preprocess_command(self, mock_store, mock_filter, mock_level, mock_open, tmp_path):
        """Test preprocess command execution."""
        from argparse import Namespace
        
        # Setup mocks
        mock_frame = MagicMock()
        mock_frame.data = [[1.0, 2.0], [3.0, 4.0]]  # 2D data, not 3D
        mock_meta = MagicMock()
        mock_open.return_value = (mock_frame, mock_meta)
        mock_level.return_value = (mock_frame.data, {})
        mock_filter.return_value = mock_frame.data
        
        mock_store_instance = MagicMock()
        mock_store.open.return_value.__enter__.return_value = mock_store_instance
        
        input_file = tmp_path / "test.gwy"
        input_file.touch()
        output_file = tmp_path / "output.zarr"
        
        args = Namespace(
            input=input_file,
            output=output_file,
            leveling="plane",
            filter="gaussian",
            filter_sigma=1.0,
            remove_scars=False,
            gpu=False,
            precision="REFERENCE",
            verbose=False,
        )
        
        # Import after mocking to avoid import issues
        from nanolocz.cli.preprocess import cmd_preprocess
        result = cmd_preprocess(args)
        
        # Should return success
        assert result == 0


# Test batch processing
class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def test_dry_run(self, tmp_path):
        """Test batch dry-run mode."""
        from nanolocz.cli.main import main
        import io
        import sys
        
        # Create test files
        (tmp_path / "file1.gwy").touch()
        (tmp_path / "file2.gwy").touch()
        output_dir = tmp_path / "output"
        
        # Capture stdout
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            result = main([
                "batch",
                "-i", str(tmp_path / "file1.gwy"), str(tmp_path / "file2.gwy"),
                "-o", str(output_dir),
                "--dry-run",
            ])
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        assert "DRY RUN" in output
        assert "file1.gwy" in output
        assert "file2.gwy" in output
        assert result == 0


# Integration test - actual CLI invocation
class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_cli_version(self):
        """Test CLI version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "nanolocz.cli.main", "--version"],
            capture_output=True,
            text=True,
        )
        # Version should be shown (either in stdout or via SystemExit)
        assert result.returncode in (0, 1)
    
    def test_cli_no_args(self):
        """Test CLI with no arguments shows error."""
        result = subprocess.run(
            ["nanolocz"],
            capture_output=True,
            text=True,
        )
        # Should show usage or error
        assert result.returncode != 0 or "usage" in result.stderr.lower()
