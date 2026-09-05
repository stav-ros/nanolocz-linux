"""NanoLocz Napari Plugin - Main dock widget implementation.

This module provides the main dock widget for the NanoLocz plugin, offering
a complete AFM analysis workflow from preprocessing through LAFM reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Points, Tracks
from napari.utils.notifications import show_info, show_error, show_warning
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
)

if TYPE_CHECKING:
    from napari import Viewer


@dataclass
class PipelineConfig:
    """Shared configuration for CLI and Napari plugin."""
    
    # Preprocessing
    leveling_method: str = "plane"
    filter_name: str = "gaussian"
    filter_sigma: float = 1.0
    
    # Detection
    detection_threshold: float = 3.5
    min_distance: float = 5.0
    prominence: float = 0.0
    
    # Tracking
    max_displacement: float = 10.0
    memory: int = 2
    gap_closing: bool = True
    
    # LAFM
    use_gpu: bool = False
    precision: str = "mixed"
    splat_sigma: float = 2.0
    
    # I/O
    input_path: str = ""
    output_path: str = ""


class NanoLoczWidget(QWidget):
    """Main NanoLocz dock widget for Napari."""
    
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        self.config = PipelineConfig()
        self.current_image_layer: Image | None = None
        self.particles_layer: Points | None = None
        self.tracks_layer: Tracks | None = None
        self.lafm_layer: Image | None = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Project section
        project_group = self._create_project_group()
        layout.addWidget(project_group)
        
        # Preprocess section
        preprocess_group = self._create_preprocess_group()
        layout.addWidget(preprocess_group)
        
        # Detect section
        detect_group = self._create_detect_group()
        layout.addWidget(detect_group)
        
        # Track section
        track_group = self._create_track_group()
        layout.addWidget(track_group)
        
        # Results section
        results_group = self._create_results_group()
        layout.addWidget(results_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def _create_project_group(self) -> QGroupBox:
        """Create the project management group."""
        group = QGroupBox("Project")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # Open file button
        self.open_btn = QPushButton("📁 Open File")
        self.open_btn.clicked.connect(self._open_file)
        layout.addWidget(self.open_btn)
        
        # Recent files label
        self.recent_label = QLabel("Recent: None")
        self.recent_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.recent_label)
        
        # Save config button
        self.save_config_btn = QPushButton("💾 Save Config")
        self.save_config_btn.clicked.connect(self._save_config)
        layout.addWidget(self.save_config_btn)
        
        # Export results button
        self.export_btn = QPushButton("📤 Export Results")
        self.export_btn.clicked.connect(self._export_results)
        layout.addWidget(self.export_btn)
        
        return group
    
    def _create_preprocess_group(self) -> QGroupBox:
        """Create the preprocessing controls group."""
        group = QGroupBox("Preprocess")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # Leveling method
        leveling_layout = QHBoxLayout() if hasattr(__builtins__, 'QHBoxLayout') else QVBoxLayout()
        leveling_label = QLabel("Leveling:")
        self.leveling_combo = QComboBox()
        self.leveling_combo.addItems(["none", "plane", "median"])
        self.leveling_combo.currentTextChanged.connect(
            lambda v: setattr(self.config, 'leveling_method', v)
        )
        layout.addWidget(leveling_label)
        layout.addWidget(self.leveling_combo)
        
        # Filter type
        filter_layout = QVBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["none", "gaussian", "median", "bilateral"])
        self.filter_combo.currentTextChanged.connect(
            lambda v: setattr(self.config, 'filter_name', v)
        )
        layout.addWidget(filter_label)
        layout.addWidget(self.filter_combo)
        
        # Filter sigma
        sigma_layout = QVBoxLayout()
        sigma_label = QLabel("Filter Sigma:")
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setValue(1.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.valueChanged.connect(
            lambda v: setattr(self.config, 'filter_sigma', v)
        )
        layout.addWidget(sigma_label)
        layout.addWidget(self.sigma_spin)
        
        # Apply button
        self.apply_preprocess_btn = QPushButton("▶ Apply Preprocessing")
        self.apply_preprocess_btn.clicked.connect(self._apply_preprocessing)
        layout.addWidget(self.apply_preprocess_btn)
        
        return group
    
    def _create_detect_group(self) -> QGroupBox:
        """Create the particle detection controls group."""
        group = QGroupBox("Detect")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # Threshold
        threshold_layout = QVBoxLayout()
        threshold_label = QLabel("Detection Threshold:")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 20.0)
        self.threshold_spin.setValue(3.5)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.valueChanged.connect(
            lambda v: setattr(self.config, 'detection_threshold', v)
        )
        layout.addWidget(threshold_label)
        layout.addWidget(self.threshold_spin)
        
        # Minimum distance
        distance_layout = QVBoxLayout()
        distance_label = QLabel("Min Distance (pixels):")
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1.0, 50.0)
        self.distance_spin.setValue(5.0)
        self.distance_spin.setSingleStep(1.0)
        self.distance_spin.valueChanged.connect(
            lambda v: setattr(self.config, 'min_distance', v)
        )
        layout.addWidget(distance_label)
        layout.addWidget(self.distance_spin)
        
        # Prominence
        prominence_layout = QVBoxLayout()
        prominence_label = QLabel("Prominence:")
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.0, 10.0)
        self.prominence_spin.setValue(0.0)
        self.prominence_spin.setSingleStep(0.1)
        self.prominence_spin.valueChanged.connect(
            lambda v: setattr(self.config, 'prominence', v)
        )
        layout.addWidget(prominence_label)
        layout.addWidget(self.prominence_spin)
        
        # Detect button
        self.detect_btn = QPushButton("🔍 Detect Particles")
        self.detect_btn.clicked.connect(self._detect_particles)
        layout.addWidget(self.detect_btn)
        
        return group
    
    def _create_track_group(self) -> QGroupBox:
        """Create the particle tracking controls group."""
        group = QGroupBox("Track")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # Max displacement
        displacement_layout = QVBoxLayout()
        displacement_label = QLabel("Max Displacement:")
        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(1.0, 100.0)
        self.displacement_spin.setValue(10.0)
        self.displacement_spin.setSingleStep(1.0)
        self.displacement_spin.valueChanged.connect(
            lambda v: setattr(self.config, 'max_displacement', v)
        )
        layout.addWidget(displacement_label)
        layout.addWidget(self.displacement_spin)
        
        # Memory
        memory_layout = QVBoxLayout()
        memory_label = QLabel("Memory (frames):")
        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(0, 10)
        self.memory_spin.setValue(2)
        self.memory_spin.valueChanged.connect(
            lambda v: setattr(self.config, 'memory', v)
        )
        layout.addWidget(memory_label)
        layout.addWidget(self.memory_spin)
        
        # Gap closing
        self.gap_check = QCheckBox("Gap Closing")
        self.gap_check.setChecked(True)
        self.gap_check.stateChanged.connect(
            lambda v: setattr(self.config, 'gap_closing', v == 2)  # Qt.Checked = 2
        )
        layout.addWidget(self.gap_check)
        
        # Track button
        self.track_btn = QPushButton("🔗 Track Particles")
        self.track_btn.clicked.connect(self._track_particles)
        layout.addWidget(self.track_btn)
        
        return group
    
    def _create_results_group(self) -> QGroupBox:
        """Create the results and analysis group."""
        group = QGroupBox("Results")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # Statistics label
        self.stats_label = QLabel("Particles: 0 | Tracks: 0")
        layout.addWidget(self.stats_label)
        
        # LAFM reconstruction button
        self.lafm_btn = QPushButton("🎨 LAFM Reconstruction")
        self.lafm_btn.clicked.connect(self._run_lafm)
        layout.addWidget(self.lafm_btn)
        
        # FRC resolution button
        self.frc_btn = QPushButton("📊 FRC Resolution")
        self.frc_btn.clicked.connect(self._compute_frc)
        layout.addWidget(self.frc_btn)
        
        # Export layer button
        self.export_layer_btn = QPushButton("💾 Export Layer")
        self.export_layer_btn.clicked.connect(self._export_layer)
        layout.addWidget(self.export_layer_btn)
        
        return group
    
    def _open_file(self) -> None:
        """Open a file dialog to load AFM data."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open AFM File",
            "",
            "AFM Files (*.tiff *.tif *.zarr *.h5 *.gwy);;All Files (*)"
        )
        
        if file_path:
            self._load_file(file_path)
    
    def _load_file(self, file_path: str) -> None:
        """Load an AFM file into the viewer."""
        try:
            from nanolocz.io.read import read_afm
            
            data = read_afm(Path(file_path))
            
            # Update recent files
            self.recent_label.setText(f"Recent: {Path(file_path).name}")
            
            # Add image layer
            self.current_image_layer = self.viewer.add_image(
                data,
                name=Path(file_path).stem,
                colormap='gray',
                contrast_limits=(np.percentile(data, 1), np.percentile(data, 99))
            )
            
            self.config.input_path = file_path
            show_info(f"Loaded: {Path(file_path).name}")
            
        except Exception as e:
            show_error(f"Failed to load file: {e}")
    
    def _apply_preprocessing(self) -> None:
        """Apply preprocessing to the current image."""
        if self.current_image_layer is None:
            show_warning("No image loaded")
            return
        
        try:
            from nanolocz.core.preprocess import level_image, apply_filter
            
            data = self.current_image_layer.data
            
            # Apply leveling
            if self.config.leveling_method != "none":
                data = level_image(data, method=self.config.leveling_method)
            
            # Apply filter
            if self.config.filter_name != "none":
                data = apply_filter(
                    data,
                    method=self.config.filter_name,
                    sigma=self.config.filter_sigma
                )
            
            # Update image layer
            self.viewer.layers.remove(self.current_image_layer)
            self.current_image_layer = self.viewer.add_image(
                data,
                name=f"{self.current_image_layer.name}_processed",
                colormap='gray'
            )
            
            show_info("Preprocessing applied")
            
        except Exception as e:
            show_error(f"Preprocessing failed: {e}")
    
    def _detect_particles(self) -> None:
        """Detect particles in the current image."""
        if self.current_image_layer is None:
            show_warning("No image loaded")
            return
        
        try:
            from nanolocz.core.detection import detect_peaks
            
            data = self.current_image_layer.data
            
            # Detect peaks
            peaks = detect_peaks(
                data,
                threshold=self.config.detection_threshold,
                min_distance=self.config.min_distance,
                prominence=self.config.prominence
            )
            
            # Remove existing particles layer
            if self.particles_layer is not None:
                self.viewer.layers.remove(self.particles_layer)
            
            # Add points layer
            self.particles_layer = self.viewer.add_points(
                peaks,
                name="particles",
                size=5,
                face_color='red',
                edge_color='yellow'
            )
            
            # Update stats
            self.stats_label.setText(f"Particles: {len(peaks)} | Tracks: 0")
            show_info(f"Detected {len(peaks)} particles")
            
        except Exception as e:
            show_error(f"Detection failed: {e}")
    
    def _track_particles(self) -> None:
        """Track particles across frames."""
        if self.particles_layer is None:
            show_warning("No particles detected")
            return
        
        if self.current_image_layer is None:
            show_warning("No image loaded")
            return
        
        try:
            from nanolocz.core.tracking import track_particles
            
            # Get particle data
            particles = self.particles_layer.data
            
            # Handle both 2D and 3D cases
            if particles.ndim == 2 and particles.shape[1] == 2:
                # Single frame - create dummy tracks
                tracks = np.column_stack([
                    np.zeros(len(particles)),
                    particles[:, 0],
                    particles[:, 1],
                    np.arange(len(particles))
                ])
            else:
                # Multi-frame tracking
                tracks = track_particles(
                    particles,
                    max_displacement=self.config.max_displacement,
                    memory=self.config.memory,
                    gap_closing=self.config.gap_closing
                )
            
            # Remove existing tracks layer
            if self.tracks_layer is not None:
                self.viewer.layers.remove(self.tracks_layer)
            
            # Add tracks layer
            self.tracks_layer = self.viewer.add_tracks(
                tracks,
                name="tracks",
                tail_width=2,
                tail_length=10,
                color_by='track_id'
            )
            
            # Update stats
            n_tracks = len(np.unique(tracks[:, 0]))
            self.stats_label.setText(f"Particles: {len(particles)} | Tracks: {n_tracks}")
            show_info(f"Created {n_tracks} tracks")
            
        except Exception as e:
            show_error(f"Tracking failed: {e}")
    
    def _run_lafm(self) -> None:
        """Run LAFM reconstruction."""
        if self.current_image_layer is None or self.particles_layer is None:
            show_warning("Need image and particles for LAFM")
            return
        
        try:
            from nanolocz.gpu.lafm import lafm_splat
            
            data = self.current_image_layer.data
            particles = self.particles_layer.data
            
            # Run LAFM splat
            lafm_result = lafm_splat(
                data,
                particles,
                sigma=self.config.splat_sigma,
                use_gpu=self.config.use_gpu
            )
            
            # Remove existing LAFM layer
            if self.lafm_layer is not None:
                self.viewer.layers.remove(self.lafm_layer)
            
            # Add LAFM layer
            self.lafm_layer = self.viewer.add_image(
                lafm_result,
                name="LAFM",
                colormap='viridis',
                blending='additive'
            )
            
            show_info("LAFM reconstruction complete")
            
        except Exception as e:
            show_error(f"LAFM failed: {e}")
    
    def _compute_frc(self) -> None:
        """Compute FRC resolution."""
        if self.lafm_layer is None:
            show_warning("Need LAFM reconstruction first")
            return
        
        try:
            from nanolocz.gpu.lafm import compute_frc
            
            data = self.lafm_layer.data
            
            # Compute FRC
            frc_result = compute_frc(data)
            
            show_info(f"FRC resolution: {frc_result['resolution']:.2f} px")
            
        except Exception as e:
            show_error(f"FRC computation failed: {e}")
    
    def _save_config(self) -> None:
        """Save current configuration to JSON."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                import json
                
                with open(file_path, 'w') as f:
                    json.dump(self.config.__dict__, f, indent=2)
                
                show_info(f"Config saved to {file_path}")
                
            except Exception as e:
                show_error(f"Failed to save config: {e}")
    
    def _export_results(self) -> None:
        """Export all results."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Export Results Directory"
        )
        
        if dir_path:
            try:
                import json
                from pathlib import Path
                
                output_dir = Path(dir_path)
                
                # Export config
                config_path = output_dir / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(self.config.__dict__, f, indent=2)
                
                # Export particles
                if self.particles_layer is not None:
                    particles_path = output_dir / "particles.csv"
                    np.savetxt(particles_path, self.particles_layer.data, delimiter=',')
                
                # Export tracks
                if self.tracks_layer is not None:
                    tracks_path = output_dir / "tracks.csv"
                    np.savetxt(tracks_path, self.tracks_layer.data, delimiter=',')
                
                # Export LAFM
                if self.lafm_layer is not None:
                    lafm_path = output_dir / "lafm.tif"
                    from tifffile import imwrite
                    imwrite(lafm_path, self.lafm_layer.data.astype(np.float32))
                
                show_info(f"Results exported to {dir_path}")
                
            except Exception as e:
                show_error(f"Export failed: {e}")
    
    def _export_layer(self) -> None:
        """Export the currently selected layer."""
        if self.viewer.layers.selection.active is None:
            show_warning("No layer selected")
            return
        
        layer = self.viewer.layers.selection.active
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {layer.name}",
            "",
            "TIFF Files (*.tif *.tiff);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    np.savetxt(file_path, layer.data, delimiter=',')
                else:
                    from tifffile import imwrite
                    imwrite(file_path, layer.data.astype(np.float32))
                
                show_info(f"Exported {layer.name} to {file_path}")
                
            except Exception as e:
                show_error(f"Export failed: {e}")


# Import CheckBox for track group
from qtpy.QtWidgets import QCheckBox, QHBoxLayout
