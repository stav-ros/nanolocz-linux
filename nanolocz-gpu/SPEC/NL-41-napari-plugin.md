# NL-41 — Napari plugin v1

**Phase:** P4 — Interface and ship  
**Depends:** NL-40 (Headless CLI and batch runner) ✓  
**State:** not_started → in_progress  
**Agent budget:** 1-2 sessions  

---

## Acceptance criteria

1. **Napari plugin structure**: Proper napari plugin package with manifest
2. **Core viewer features**:
   - Load AFM files (.gwy, .h5-jpk, .spm, .jpk, .ibw, .asd, .tiff, .zarr)
   - Display movie frames with frame slider
   - Show detected particles as points layer
   - Display tracks as tracks layer or labeled paths
   - LAFM splat reconstruction visualization
   - FRC resolution overlay
3. **Processing integration**: Call CLI backend functions directly from GUI
4. **Interactive controls**:
   - Threshold adjustment for detection
   - Leveling method selection
   - Filter parameters (sigma, type)
   - Tracking parameters (max displacement, memory)
5. **Export functionality**: Save results to .zarr, export figures (PNG, TIFF)
6. **Tests**: 10+ tests for plugin loading, layer creation, parameter updates
7. **Documentation**: Plugin usage in README, example screenshots

---

## Implementation plan

### 1. Create plugin package structure
```
nanolocz/
  plugins/
    __init__.py           # Plugin entry point with napari hook
    napari.yaml           # Plugin manifest
    widget.py             # Main dock widget
    layers.py             # Custom layer types (optional)
```

### 2. Implement napari.yaml manifest
Define:
- Plugin name, version, display name
- Dock widgets with commands
- Sample data readers for AFM formats

### 3. Build main dock widget
Tabs or collapsible sections for:
- **File I/O**: Open file dialog, recent files, save/export
- **Preprocessing**: Leveling, filtering, scar removal controls
- **Detection**: Threshold, min-distance, prominence sliders
- **Tracking**: Max displacement, memory controls
- **LAFM**: Pixel size, sigma, FRC threshold
- **Results**: Statistics panel, export buttons

### 4. Layer management
- Image layer for AFM frames
- Points layer for particle detections
- Tracks layer or Paths for trajectories
- Image layer for LAFM reconstruction
- Labels layer for regions/masks

### 5. Backend integration
- Import functions from `nanolocz.cli.*` modules
- Use same PipelineConfig as CLI
- Run processing in worker threads (napari.utils.threading)
- Progress reporting via napari notification system

### 6. Tests
- Plugin discovery and loading
- Widget creation and visibility
- File loading with sample data
- Layer creation after processing
- Parameter update callbacks
- Export functionality

### 7. Documentation
- Add plugin section to README.md
- Screenshot examples
- Link to napari plugin hub

---

## File changes

**New files:**
- `nanolocz/plugins/__init__.py`
- `nanolocz/plugins/napari.yaml`
- `nanolocz/plugins/widget.py`
- `nanolocz/plugins/layers.py` (optional)
- `tests/test_napari_plugin_nl41.py`
- `SPEC/NL-41-napari-plugin.md` (this file)

**Modified files:**
- `pyproject.toml` (add napari plugin entry point)
- `README.md` (add plugin usage section)
- `STATUS.md` (mark NL-41 done)
- `examples/` (add sample screenshots)

---

## Evidence of completion

- [ ] Napari discovers and loads plugin (`napari --info` shows nanolocz)
- [ ] Dock widget appears in napari interface
- [ ] Can load AFM file and display as image layer
- [ ] Detection creates points layer
- [ ] Tracking displays trajectories
- [ ] LAFM reconstruction visualized
- [ ] 10+ tests passing
- [ ] SESSIONS/2026-09-XX-NL-41.md handoff created
- [ ] STATUS.md updated with NL-41 done

---

## Example usage

```python
# In napari Python console or script:
import napari
from nanolocz.plugins import NanoLoczWidget

viewer = napari.Viewer()
widget = NanoLoczWidget(viewer)
viewer.window.add_dock_widget(widget, area='right')

# Or via napari menu: Plugins > NanoLocz > NanoLocz Controller
```

### User workflow:
1. Open napari
2. Plugins → NanoLocz → NanoLocz Controller
3. Click "Open File" → select .gwy/.h5-jpk/etc.
4. Adjust preprocessing parameters → Apply
5. Adjust detection threshold → Detect Particles
6. Set tracking parameters → Track
7. View LAFM reconstruction → Export Results

---

## Notes

- Keep plugin lightweight; delegate heavy computation to existing core modules
- Use napari's magicgui for automatic widget generation from typed functions
- Support both synchronous and async processing (worker threads for long operations)
- Follow napari plugin best practices for compatibility
- Consider future extension to 3D volume visualization (for NL-37 reconstruction)
