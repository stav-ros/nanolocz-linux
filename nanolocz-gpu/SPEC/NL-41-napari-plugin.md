# NL-41 — Napari plugin v1

**Phase:** P4 — Interface and ship  
**Depends:** NL-40 (Headless CLI and batch runner) ✓  
**State:** not_started → in_progress  
**Agent budget:** 2 sessions (NL-41a: minimum usable plugin, NL-41b: full workflow)  

---

## Scope split

### NL-41a — Minimum usable plugin (this card)
Focus on core functionality for immediate testing:

1. **Napari plugin discovery** — Plugin manifest and loading
2. **File readers** — Open `.tiff`, `.zarr`, `.gwy`, `.h5-jpk` formats
3. **Image/movie display** — Image layer with frame slider
4. **Basic leveling controls** — Line/plane leveling selection
5. **Detection threshold control** — Threshold slider, detect button
6. **Detection points layer** — Show detected particles
7. **Track display** — Display trajectories from tracking
8. **Export processed result** — Save to .zarr, export images
9. **Worker threads** — Non-blocking processing with progress
10. **Tests** — 10-15 plugin tests

**Acceptance:** User can open an AFM file, adjust parameters, run detection/tracking, see results as layers, and export.

### NL-41b — Full analysis workflow (future card)
Advanced features after NL-41a is stable:

- Additional format support (.spm, .jpk, .ibw, .asd)
- LAFM reconstruction visualization
- FRC resolution overlay
- 3D reconstruction viewer (NL-37 integration)
- ROI measurements and shapes layer
- Batch processing from GUI
- GPU selection controls
- Advanced filtering parameters
- Session/config persistence
- Multiple synchronized viewers

---

## Acceptance criteria (NL-41a)

1. **Napari plugin structure**: Proper napari plugin package with `napari.yaml` manifest
2. **Core viewer features**:
   - Load `.tiff`, `.zarr`, `.gwy`, `.h5-jpk` files
   - Display movie frames with frame slider
   - Show detected particles as points layer
   - Display tracks as paths/trajectories
3. **Processing integration**: Call CLI backend functions directly from GUI (`nanolocz.cli.detect`, `nanolocz.cli.track`)
4. **Interactive controls**:
   - Threshold adjustment for detection (slider + input)
   - Leveling method selection (dropdown: none/line/plane)
   - Apply preprocessing and detection buttons
5. **Export functionality**: Save results to .zarr, export current view as PNG
6. **Worker threads**: Processing runs in background with progress bar, cancellation support
7. **Tests**: 10+ tests for plugin loading, layer creation, parameter updates
8. **Documentation**: Plugin usage in README, installation instructions

---

## Implementation plan (NL-41a)

### 1. Create plugin package structure
```
nanolocz/
  plugins/
    __init__.py           # Plugin entry point with napari hook
    napari.yaml           # Plugin manifest
    widget.py             # Main dock widget
    config.py             # Shared PipelineConfig dataclass
```

### 2. Implement napari.yaml manifest
```yaml
name: nanolocz
display_name: NanoLocz
contributions:
  commands:
    - id: nanolocz.make_qwidget
      python_name: nanolocz.plugins.widget:NanoLoczWidget
      title: NanoLocz Controller
  widgets:
    - command: nanolocz.make_qwidget
      display_name: NanoLocz
```

### 3. Build main dock widget (5 sections)
```
NanoLocz Widget
├── Project
│   ├── Open file (.tiff, .zarr, .gwy, .h5-jpk)
│   ├── Recent files list
│   ├── Save project/config
│   └── Export results
│
├── Preprocess
│   ├── Leveling method (none/line/plane)
│   ├── [Apply Leveling] button
│   └── Status indicator
│
├── Detect
│   ├── Threshold slider (0.0-10.0, default 3.5)
│   ├── Min distance slider (1-20, default 5)
│   └── [Detect Particles] button
│
├── Track
│   ├── Max displacement slider (1-50, default 10)
│   ├── Memory slider (0-5, default 2)
│   └── [Track Particles] button
│
└── Results
    ├── Particle count display
    ├── Track count display
    ├── [Export Zarr] button
    └── [Export PNG] button
```

### 4. Layer management
- **Image layer**: AFM frames (grayscale), contrast limits adjustable
- **Points layer**: Particle detections (red markers, labeled)
- **Tracks layer**: Trajectories (colored by track ID)
- **Progress indicator**: In-widget status or napari notification

### 5. Backend integration
```python
# Import from existing CLI modules
from nanolocz.cli.preprocess import run_preprocessing
from nanolocz.cli.detect import run_detection
from nanolocz.cli.track import run_tracking
from nanolocz.core.types import PipelineConfig

# Use worker threads
from napari.utils.threading import thread_worker

@thread_worker
def process_detection(image, config):
    return run_detection(image, config)
```

### 6. Shared configuration model
```python
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """Shared between CLI and Napari plugin"""
    leveling: str = "plane"
    filter_name: str = "gaussian"
    filter_sigma: float = 1.0
    detection_threshold: float = 3.5
    min_distance: float = 5.0
    max_displacement: float = 10.0
    memory: int = 2
    use_gpu: bool = False
    precision: str = "mixed"
```

### 7. Tests
- Plugin discovery (`napari --info`)
- Widget instantiation
- File loading with test fixtures
- Layer creation after detection
- Parameter update callbacks
- Export functionality
- Worker thread execution

---

## File changes

**New files:**
- `nanolocz/plugins/__init__.py`
- `nanolocz/plugins/napari.yaml`
- `nanolocz/plugins/widget.py`
- `nanolocz/plugins/config.py`
- `tests/test_napari_plugin_nl41a.py`
- `SESSIONS/2026-09-XX-NL-41a.md`

**Modified files:**
- `pyproject.toml` (add napari plugin entry point) ✓ already done
- `README.md` (add plugin usage section)
- `STATUS.md` (mark NL-41a done)
- `SPEC/NL-41-napari-plugin.md` (this file — updated for NL-41a/b split)

---

## Evidence of completion (NL-41a)

- [ ] Napari discovers plugin (`napari --info` shows nanolocz)
- [ ] `napari.yaml` manifest valid
- [ ] Dock widget appears in napari (Plugins → NanoLocz → NanoLocz Controller)
- [ ] Can load `.tiff`, `.zarr`, `.gwy`, `.h5-jpk` files
- [ ] Image layer displays with frame slider for movies
- [ ] Detection creates points layer with correct particle positions
- [ ] Tracking displays trajectory paths
- [ ] Export saves .zarr file with results
- [ ] Processing runs in worker thread (UI remains responsive)
- [ ] 10+ tests passing
- [ ] SESSIONS handoff created
- [ ] STATUS.md updated

---

## Example usage

```python
# In napari Python console or script:
import napari

viewer = napari.Viewer()
# Via menu: Plugins → NanoLocz → NanoLocz Controller

# Or programmatically:
from nanolocz.plugins.widget import NanoLoczWidget
widget = NanoLoczWidget(viewer)
viewer.window.add_dock_widget(widget, area='right')
```

### User workflow (NL-41a):
1. Install: `pip install -e ".[napari]"`
2. Launch: `napari`
3. Open plugin: Plugins → NanoLocz → NanoLocz Controller
4. Click "Open File" → select `.gwy` or `.h5-jpk` or `.tiff` or `.zarr`
5. Adjust leveling method → Apply Leveling
6. Adjust detection threshold → Detect Particles
7. View particles as red points overlay
8. Set tracking parameters → Track Particles
9. View trajectories as colored paths
10. Export results to .zarr or PNG

---

## Notes

- **Keep plugin lightweight**: Delegate all computation to existing `nanolocz/cli/` and `nanolocz/core/` modules
- **Use magicgui**: Automatic widget generation from typed function signatures where appropriate
- **Worker threads required**: Detection, tracking must not block UI; use `@thread_worker` decorator
- **Shared config**: CLI and plugin use identical `PipelineConfig` dataclass
- **Clear boundaries**: React dashboard = documentation/control plane; Napari = scientific interface; CLI/core = execution engine
- **Future extension**: NL-41b will add LAFM, FRC, 3D reconstruction, ROI tools, batch GUI processing
