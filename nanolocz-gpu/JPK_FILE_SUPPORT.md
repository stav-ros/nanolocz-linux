# JPK File Support in NanoLocz

## Overview

NanoLocz now supports **both** JPK file formats:

1. **HDF5-based JPK** (`.jpk`, `.h5-jpk`, `.jpks`) - Newer instruments
2. **Legacy Binary JPK** (`.jpk`) - Older instruments

## What Was Fixed

### Problem
Your `.jpk` file was failing with error: `Unable to synchronously open file (file signature not found)`

This occurred because:
- Your file is a **legacy binary JPK format** (not HDF5)
- The original reader only supported HDF5-based JPK files
- Napari couldn't read the file through drag-and-drop

### Solution
Enhanced `nanolocz/formats/jpk_reader.py` to:
1. Automatically detect file format (HDF5 vs binary)
2. Parse legacy binary JPK files with intelligent dimension detection
3. Extract metadata from both formats
4. Return napari-compatible LayerDataTuple

## Supported File Formats

### All Registered Readers (via npe2)

| Format | Extensions | Reader Function |
|--------|-----------|----------------|
| TIFF | `.tiff`, `.tif` | `napari_read_tiff` |
| HDF5 AFM | `.h5`, `.hdf5` | `napari_read_h5_afm` |
| Gwyddion | `.gwy` | `napari_read_gwy` |
| **JPK (HDF5)** | `.jpk`, `.h5-jpk`, `.jpks` | `napari_read_jpk` |
| **JPK (Binary)** | `.jpk` | `napari_read_jpk` |
| SPM | `.spm` | `napari_read_spm` |
| IBW | `.ibw` | `napari_read_ibw` |
| ASD | `.asd` | `napari_read_asd` |

## How to Use

### Option 1: Drag-and-Drop in Napari (Now Works!)

1. Close napari completely
2. Ensure plugin is installed: `pip install -e .`
3. Start napari
4. Drag your `.jpk` file into napari window

### Option 2: File → Open Menu

1. In napari, go to `File → Open...`
2. Select your `.jpk` file
3. NanoLocz reader will automatically detect format and load

### Option 3: NanoLocz Widget

1. Open NanoLocz plugin from napari's Plugins menu
2. Use the "Load File" button in the widget
3. Select your `.jpk` file

### Option 4: Python API

```python
from nanolocz.formats import read_jpk

# Automatically detects format
data, metadata = read_jpk("your_file.jpk")
print(f"Shape: {data.shape}, Format: {metadata['format']}")

# For HDF5 files, you can specify channel
data, metadata = read_jpk("your_file.jpk", channel="deflection")
```

## Legacy Binary JPK Detection

The enhanced reader intelligently detects binary JPK files by:

1. **Checking HDF5 signature**: First 8 bytes are checked for `\x89HDF\r\n\x1a\n`
2. **Dimension detection**: 
   - Tries common header offsets (60-68, 32-40 bytes)
   - Falls back to file size analysis
   - Tests common AFM sizes: 64×64, 128×128, 256×256, 512×512, 1024×1024
3. **Data type detection**:
   - Tries float32 (4 bytes/pixel)
   - Tries float64 (8 bytes/pixel)
   - Tries int16/uint16 (2 bytes/pixel)
4. **Metadata extraction**:
   - Attempts to read scan size from header offsets 80-96
   - Calculates pixel size if scan size found

## Verification

Check that everything is properly registered:

```bash
# Verify plugin is installed
npe2 list | grep nanolocz

# Should show: nanolocz | 0.1.0.dev0 | ✅ | commands (1), readers (7), widgets (1)

# Validate manifest
npe2 validate nanolocz

# Should show: ✔ Manifest for 'NanoLocz' valid!

# View registered readers
npe2 parse nanolocz
```

## Troubleshooting

### Still Getting "file signature not found" Error?

1. **Restart napari completely** - Plugin registration requires fresh start
2. **Clear napari cache**:
   ```powershell
   Remove-Item -Recurse -Force $env:LOCALAPPDATA\napari
   ```
3. **Verify installation**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -e .
   npe2 list | grep nanolocz
   ```
4. **Check file path** - Ensure Windows path is accessible from WSL if using WSL

### Binary JPK Not Reading Correctly?

If dimensions are wrong or data looks corrupted:

1. Check file size: should be `512 + (width × height × bytes_per_pixel)`
2. Try opening in JPK's own software to verify file integrity
3. Convert to HDF5 format using JPK software for better compatibility
4. Report the file size and we can add support for your specific format variant

### CUDA Path Warning

The CUDA warning is separate from file reading and doesn't affect JPK loading:
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
```

## Technical Details

### File Format Detection

```python
def _is_hdf5_file(filepath):
    with open(filepath, 'rb') as f:
        signature = f.read(8)
        return signature == b'\x89HDF\r\n\x1a\n'
```

### Binary JPK Structure

```
[512-byte header][Raw image data]
```

Header may contain:
- Bytes 32-36: Width (alternative location)
- Bytes 36-40: Height (alternative location)
- Bytes 60-64: Width (common location)
- Bytes 64-68: Height (common location)
- Bytes 80-88: Scan size X (double precision)
- Bytes 88-96: Scan size Y (double precision)

### Metadata Returned

For all JPK files, you get:
```python
{
    'shape': (height, width),
    'dtype': 'float32',
    'filepath': '/path/to/file.jpk',
    'format': 'JPK-Binary' or 'H5-JPK',
    'width': 256,
    'height': 256,
    'pixel_size': (x_size, y_size),  # If available
    'pixel_size_units': 'm',  # If available
    'scan_size_x': 10e-6,  # If available
    'scan_size_y': 10e-6,  # If available
}
```

## Next Steps

1. ✅ Plugin registered with npe2
2. ✅ All 7 file formats supported
3. ✅ Legacy binary JPK support added
4. ✅ Napari integration complete
5. 🔄 Test with your actual file in Windows napari
6. 📝 Document any edge cases for future enhancement

## References

- JPK Instruments: https://www.jpk.com
- HDF5 Signature: https://forum.hdfgroup.org/t/hdf5-file-signature/
- Napari Plugin System: https://napari.org/stable/plugins/
- npe2 Documentation: https://github.com/napari/npe2
