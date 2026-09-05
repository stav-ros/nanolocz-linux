"""NanoLocz Napari Plugin - Main dock widget implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image, Points, Tracks
from napari.utils.notifications import show_info, show_error, show_warning
from qtpy.QtWidgets import (
    QVBoxLayout, QWidget, QPushButton, QFileDialog,
    QGroupBox, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QHBoxLayout,
)

if TYPE_CHECKING:
    from napari import Viewer

from nanolocz.core.config import PipelineConfig


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
        self.recent_files: list[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._create_project_group())
        layout.addWidget(self._create_preprocess_group())
        layout.addWidget(self._create_detect_group())
        layout.addWidget(self._create_track_group())
        layout.addWidget(self._create_results_group())
        self.stats_label = QLabel("Particles: 0 | Tracks: 0")
        layout.addWidget(self.stats_label)
        self.recent_label = QLabel("Recent: None")
        layout.addWidget(self.recent_label)

    def _create_project_group(self) -> QGroupBox:
        group = QGroupBox("Project")
        layout = QVBoxLayout()
        open_btn = QPushButton("Open File...")
        open_btn.clicked.connect(self._open_file_dialog)
        layout.addWidget(open_btn)
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self._save_config)
        layout.addWidget(save_btn)
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self._export_results)
        layout.addWidget(export_btn)
        group.setLayout(layout)
        return group

    def _create_preprocess_group(self) -> QGroupBox:
        group = QGroupBox("Preprocess")
        layout = QVBoxLayout()
        
        leveling_layout = QHBoxLayout()
        leveling_layout.addWidget(QLabel("Leveling:"))
        self.leveling_combo = QComboBox()
        self.leveling_combo.addItems(["plane", "line", "weighted", "none"])
        self.leveling_combo.setCurrentText("plane")
        self.leveling_combo.currentTextChanged.connect(lambda t: setattr(self.config, "leveling", t))
        leveling_layout.addWidget(self.leveling_combo)
        layout.addLayout(leveling_layout)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["gaussian", "median", "uniform", "none"])
        self.filter_combo.setCurrentText("gaussian")
        self.filter_combo.currentTextChanged.connect(lambda t: setattr(self.config, "filter_type", t))
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)
        
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setValue(1.0)
        self.sigma_spin.valueChanged.connect(lambda v: setattr(self.config, "filter_sigma", v))
        sigma_layout.addWidget(self.sigma_spin)
        layout.addLayout(sigma_layout)
        
        apply_btn = QPushButton("Apply Preprocessing")
        apply_btn.clicked.connect(self._apply_preprocessing)
        layout.addWidget(apply_btn)
        
        group.setLayout(layout)
        return group

    def _create_detect_group(self) -> QGroupBox:
        group = QGroupBox("Detect")
        layout = QVBoxLayout()
        
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.1, 10.0)
        self.thresh_spin.setValue(3.0)
        self.thresh_spin.valueChanged.connect(lambda v: setattr(self.config, "threshold", v))
        thresh_layout.addWidget(self.thresh_spin)
        layout.addLayout(thresh_layout)
        
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Min Distance:"))
        self.dist_spin = QSpinBox()
        self.dist_spin.setRange(1, 50)
        self.dist_spin.setValue(5)
        self.dist_spin.valueChanged.connect(lambda v: setattr(self.config, "min_distance", v))
        dist_layout.addWidget(self.dist_spin)
        layout.addLayout(dist_layout)
        
        detect_btn = QPushButton("Detect Particles")
        detect_btn.clicked.connect(self._detect_particles)
        layout.addWidget(detect_btn)
        
        group.setLayout(layout)
        return group

    def _create_track_group(self) -> QGroupBox:
        group = QGroupBox("Track")
        layout = QVBoxLayout()
        
        disp_layout = QHBoxLayout()
        disp_layout.addWidget(QLabel("Max Displacement:"))
        self.disp_spin = QDoubleSpinBox()
        self.disp_spin.setRange(1.0, 100.0)
        self.disp_spin.setValue(10.0)
        self.disp_spin.valueChanged.connect(lambda v: setattr(self.config, "max_displacement", v))
        disp_layout.addWidget(self.disp_spin)
        layout.addLayout(disp_layout)
        
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Gap Closing:"))
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 10)
        self.gap_spin.setValue(2)
        self.gap_spin.valueChanged.connect(lambda v: setattr(self.config, "gap_closing", v))
        gap_layout.addWidget(self.gap_spin)
        layout.addLayout(gap_layout)
        
        track_btn = QPushButton("Track Particles")
        track_btn.clicked.connect(self._track_particles)
        layout.addWidget(track_btn)
        
        group.setLayout(layout)
        return group

    def _create_results_group(self) -> QGroupBox:
        group = QGroupBox("Results")
        layout = QVBoxLayout()
        lafm_btn = QPushButton("Run LAFM")
        lafm_btn.clicked.connect(self._run_lafm)
        layout.addWidget(lafm_btn)
        frc_btn = QPushButton("Compute FRC")
        frc_btn.clicked.connect(self._compute_frc)
        layout.addWidget(frc_btn)
        group.setLayout(layout)
        return group

    def _open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open AFM File", "",
            "AFM Files (*.tiff *.tif *.zarr *.h5 *.gwy *.spm *.jpk);;All Files (*)"
        )
        if file_path:
            self._load_file(file_path)

    def _load_file(self, file_path: str) -> None:
        try:
            from nanolocz.io import open_nanolocz
            dataset = open_nanolocz(Path(file_path))
            if hasattr(dataset, 'data'):
                data = dataset.data
            elif hasattr(dataset, 'images'):
                data = dataset.images
            else:
                data = np.asarray(dataset)
            
            self.recent_files.insert(0, file_path)
            self.recent_files = self.recent_files[:5]
            self.recent_label.setText(f"Recent: {Path(file_path).name}")
            
            self.current_image_layer = self.viewer.add_image(
                data, name=Path(file_path).stem, colormap='gray',
                contrast_limits=(float(np.percentile(data, 1)), float(np.percentile(data, 99)))
            )
            self.config.input_path = file_path
            show_info(f"Loaded: {Path(file_path).name}")
        except Exception as e:
            show_error(f"Failed to load file: {e}")

    def _apply_preprocessing(self) -> None:
        if self.current_image_layer is None:
            show_warning("No image loaded")
            return
        try:
            from nanolocz.core.leveling import level_image
            from nanolocz.core.filters import gaussian_blur, median_blur, uniform_blur
            data = self.current_image_layer.data.copy()
            if self.config.leveling != "none":
                data = level_image(data, method=self.config.leveling)
            if self.config.filter_type != "none":
                if self.config.filter_type == "gaussian":
                    data = gaussian_blur(data, sigma=self.config.filter_sigma)
                elif self.config.filter_type == "median":
                    data = median_blur(data, size=int(self.config.filter_sigma * 2 + 1))
                elif self.config.filter_type == "uniform":
                    data = uniform_blur(data, size=int(self.config.filter_sigma * 2 + 1))
            self.viewer.layers.remove(self.current_image_layer)
            self.current_image_layer = self.viewer.add_image(
                data, name=f"{self.current_image_layer.name}_processed", colormap='gray'
            )
            show_info("Preprocessing applied")
        except Exception as e:
            show_error(f"Preprocessing failed: {e}")

    def _detect_particles(self) -> None:
        if self.current_image_layer is None:
            show_warning("No image loaded")
            return
        try:
            from nanolocz.core.detection import detect_particles
            data = self.current_image_layer.data
            result = detect_particles(
                data, threshold=self.config.threshold,
                min_distance=self.config.min_distance,
                prominence=self.config.prominence if self.config.prominence > 0 else None
            )
            peaks = result['coordinates'] if isinstance(result, dict) else result
            if self.particles_layer is not None:
                self.viewer.layers.remove(self.particles_layer)
            self.particles_layer = self.viewer.add_points(
                peaks, name="particles", size=5, face_color='red', edge_color='yellow'
            )
            self.stats_label.setText(f"Particles: {len(peaks)} | Tracks: 0")
            show_info(f"Detected {len(peaks)} particles")
        except Exception as e:
            show_error(f"Detection failed: {e}")

    def _track_particles(self) -> None:
        if self.particles_layer is None:
            show_warning("No particles detected")
            return
        if self.current_image_layer is None:
            show_warning("No image loaded")
            return
        try:
            from nanolocz.core.tracking import track_particles
            particles = self.particles_layer.data
            if particles.ndim == 2 and particles.shape[1] == 2:
                tracks = np.column_stack([
                    np.zeros(len(particles)), particles[:, 0],
                    particles[:, 1], np.arange(len(particles))
                ])
            else:
                tracks = track_particles(
                    particles, max_displacement=self.config.max_displacement,
                    memory=self.config.memory, gap_closing=self.config.gap_closing
                )
            if self.tracks_layer is not None:
                self.viewer.layers.remove(self.tracks_layer)
            self.tracks_layer = self.viewer.add_tracks(
                tracks, name="tracks", tail_width=2, tail_length=10, color_by='track_id'
            )
            n_tracks = len(np.unique(tracks[:, 0]))
            self.stats_label.setText(f"Particles: {len(particles)} | Tracks: {n_tracks}")
            show_info(f"Created {n_tracks} tracks")
        except Exception as e:
            show_error(f"Tracking failed: {e}")

    def _run_lafm(self) -> None:
        if self.current_image_layer is None or self.particles_layer is None:
            show_warning("Need image and particles for LAFM")
            return
        try:
            from nanolocz.gpu.lafm import splat_localizations_gpu, BackendContext
            data = self.current_image_layer.data
            particles = self.particles_layer.data
            ctx = BackendContext(use_gpu=False)
            lafm_result = splat_localizations_gpu(data, particles, sigma=self.config.sigma, ctx=ctx)
            if self.lafm_layer is not None:
                self.viewer.layers.remove(self.lafm_layer)
            self.lafm_layer = self.viewer.add_image(
                lafm_result, name="LAFM", colormap='viridis', blending='additive'
            )
            show_info("LAFM reconstruction complete")
        except Exception as e:
            show_error(f"LAFM failed: {e}")

    def _compute_frc(self) -> None:
        if self.lafm_layer is None:
            show_warning("Need LAFM reconstruction first")
            return
        try:
            from nanolocz.gpu.lafm import compute_frc_gpu, BackendContext
            data = self.lafm_layer.data
            ctx = BackendContext(use_gpu=False)
            frc_result = compute_frc_gpu(data, ctx=ctx)
            resolution = frc_result.get('resolution', 'N/A')
            show_info(f"FRC resolution: {resolution}")
        except Exception as e:
            show_error(f"FRC computation failed: {e}")

    def _save_config(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json)")
        if file_path:
            try:
                self.config.save_json(Path(file_path))
                show_info(f"Config saved to {file_path}")
            except Exception as e:
                show_error(f"Failed to save config: {e}")

    def _export_results(self) -> None:
        if self.current_image_layer is None:
            show_warning("No data to export")
            return
        try:
            output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if not output_dir:
                return
            output_path = Path(output_dir)
            if self.current_image_layer is not None:
                from tifffile import imwrite
                imwrite(output_path / f"{self.current_image_layer.name}.tiff", self.current_image_layer.data)
            if self.particles_layer is not None:
                np.savetxt(output_path / "particles.csv", self.particles_layer.data, delimiter=',', header='y,x')
            if self.tracks_layer is not None:
                np.savetxt(output_path / "tracks.csv", self.tracks_layer.data, delimiter=',', header='track_id,t,y,x')
            if self.lafm_layer is not None:
                from tifffile import imwrite
                imwrite(output_path / "lafm.tiff", self.lafm_layer.data)
            show_info(f"Results exported to {output_path}")
        except Exception as e:
            show_error(f"Export failed: {e}")


def make_nanolocz_widget(viewer: Viewer) -> QWidget:
    """Create and return a NanoLocz widget instance."""
    return NanoLoczWidget(viewer)
