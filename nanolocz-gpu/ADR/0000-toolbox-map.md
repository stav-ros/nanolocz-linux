# ADR-0000 — upstream toolbox map

Status: audit complete; implementation not started
Audited: 2026-08-28
Upstream: `https://github.com/George-R-Heath/NanoLocz`
Audited revision: `e41575c9c0f40d4ca5d9d4b1dd0b092e792d5892`

## Scope and method

This audit covers the 60 `.m` files in upstream `NanoLocz-lib/`, the externalized
processing library introduced in NanoLocz v1.42. `GUI/` (13 files) and
`Mat_SimAFM/` (19 files) were scanned separately for dependency notes but are not
part of the first core-port inventory. The source was cloned at the revision above,
then searched for executable MATLAB calls (comments are marked where relevant).

The map is an inventory, not a porting decision. Each row needs a golden fixture
when output behavior is numerical, image-based, format-dependent, or sensitive to
MATLAB defaults.

## Required MATLAB products from upstream documentation

The upstream README lists these MATLAB app requirements:

- MATLAB 2020a or newer
- Curve Fitting Toolbox
- Image Processing Toolbox
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Bioinformatics Toolbox
- Computer Vision Toolbox

The static audit confirms substantial Image Processing, Signal Processing, and
Curve/Optimization usage. Bioinformatics, Computer Vision, and Statistics/ML need
a second semantic audit because their calls are less obvious from names and may be
wrapped in local helper functions or GUI paths.

## Core processing calls

| MATLAB function/family | Product/classification | Source locations (file:line) | Proposed Python replacement | Golden fixture | Notes / open questions |
|---|---|---|---|---|---|
| `imresize` | Image Processing Toolbox | `Detector.m:43-44`; `FindCenterPositions.m:76`; `align_movie.m:63,83`; `align_rot.m:116`; `align_trans.m:58,83`; `localize.m:54,125` | `skimage.transform.resize` or `cupyx.scipy.ndimage.zoom` | Yes | Match bilinear/bicubic, shape rounding, and anti-alias behavior. |
| `imgaussfilt` | Image Processing Toolbox | `Detector.m:55,87,109`; `LAFM_renderer.m:63,93`; `LAFM_Movie_renderer.m:71,110`; `align_movie.m:41`; `filter_movie.m:58,64,144`; `measureFRC.m:73`; `ref_selector.m:15`; `thresholder.m:63,89,108,139,157,216,244` | `scipy.ndimage.gaussian_filter` / CuPy equivalent | Yes | Freeze sigma, boundary mode, and dimensional filtering. |
| `imrotate` | Image Processing Toolbox | `ConstructParticleStack.m:74,78`; `Detector.m:83`; `FindCenterPositions.m:40`; `align_iterate.m:79`; `align_rot.m:70`; `localize.m` comments; `rotation_sym.m:15` | `skimage.transform.rotate` or custom CUDA transform | Yes | `'crop'`, interpolation, center convention, and fill value must match. |
| `imtranslate` | Image Processing Toolbox | `ConstructParticleStack.m:75,79`; `align_iterate.m:62`; `FindCenterPositions.m:65` comment; `ref_selector.m:29` | `scipy.ndimage.shift` / CuPy kernel | Yes | Pixel-center and sign convention are high-risk. |
| `imwarp` | Image Processing Toolbox | `ConstructParticleStack.m:72` | `skimage.transform.warp` | Yes | Used for affine alignment; preserve output-view semantics. |
| `imfilter` | Image Processing Toolbox | `filter_movie.m:91,105,144`; `thresholder.m:65-66` | `scipy.ndimage.convolve` / `cupyx.scipy.ndimage` | Yes | Replicate convolution/correlation orientation and `'replicate'` padding. |
| `medfilt2` | Image Processing Toolbox | `scar_fill.m:14,24` | `scipy.ndimage.median_filter` | Yes | Directional `[1,n]` windows are part of scar-removal behavior. |
| `imcrop` | Image Processing Toolbox | `LineShift.m:101`; `ref_selector.m:52` | Array slicing plus explicit bounds helper | Yes | MATLAB rectangle coordinates are inclusive and 1-based. |
| `imdilate`, `imerode`, `imclose`, `imfill` | Image Processing Toolbox | `thresholder.m:81-83,101,119,130,218,221-227` | `scipy.ndimage` morphology / `skimage.morphology` | Yes | Structuring-element geometry and hole connectivity must match. |
| `strel` | Image Processing Toolbox | `thresholder.m:80,95,100,118,129,217,220,222` | `skimage.morphology.disk`, `diamond`, `rectangle`, `line` | Yes | Disk decomposition/version behavior may differ. |
| `bwareaopen` | Image Processing Toolbox | `scar_fill.m:23`; `thresholder.m:78,79,94,116-120,127-130,219,252` | `skimage.morphology.remove_small_objects` | Yes | Connectivity and 2D/3D handling need explicit contract. |
| `bwmorph` | Image Processing Toolbox | `thresholder.m:84,94,115,117,126-129,149-150,169-170` | `skimage.morphology` equivalents or custom binary morphology | Yes | Operations include `remove`, `bridge`, `thin`, `spur`, and `clean`; no single drop-in. |
| `bwconncomp`, `bwlabeln` | Image Processing Toolbox | `level_weighted.m:33`; `thresholder.m:145` | `scipy.ndimage.label` / `skimage.measure.label` | Yes | MATLAB component ordering and connectivity affect deterministic IDs. |
| `bwskel` | Image Processing Toolbox | `thresholder.m:148,167` | `skimage.morphology.skeletonize` / `skeletonize_3d` | Yes | `MinBranchLength` needs a compatible pruning pass. |
| `regionprops` | Image Processing Toolbox | `AnalyzeAreas.m:51,76,121` | `skimage.measure.regionprops_table` | Yes | Property names, weighted intensity, and table column ordering matter. |
| `regionfill`, `poly2mask` | Image Processing Toolbox | `scar_fill.m:26`; `GUI/draw_calc.m:94`, `GUI/draw_finish.m:59` | `skimage.restoration.inpaint` or `scipy`; `skimage.draw.polygon2mask` | Yes | `regionfill` interpolation is a numerical parity item; GUI calls are secondary scope. |
| `normxcorr2` | Image Processing Toolbox | `Detector.m:84,106`; `FindCenterPositions.m:66`; `align_movie.m:37`; `align_rot.m:98`; `align_trans.m:34` | FFT-based normalized cross-correlation in NumPy/CuPy | Yes | Define valid/full output shape, mask behavior, and normalization explicitly. |
| `improfile`, `improfile_thick` | Image Processing Toolbox / local helper | `Fast_peaks2D.m:60`; `peaks2D.m:58`; `improfile_thick.m:58` | `skimage.measure.profile_line` plus local thick-profile implementation | Yes | Sampling coordinates and interpolation need a fixture. |
| `edge` | Image Processing Toolbox | `thresholder.m:216` | `skimage.feature.canny` or Sobel gradient implementation | Yes | Current call uses Sobel; confirm threshold defaults. |
| `adaptthresh` | Image Processing Toolbox | `thresholder.m:215` (commented example) | `skimage.filters.threshold_local` | No unless activated | Latent dependency, not currently executable in the audited path. |
| `mat2gray`, `rgb2gray`, `imnlmfilt` | Image Processing Toolbox | `ref_selector.m:18`; `ReadAFMFile.m` RGB path; `filter_movie.m:77` | `skimage.exposure.rescale_intensity`, `skimage.color.rgb2gray`, `skimage.restoration.denoise_nl_means` | Yes for active paths | Confirm whether RGB conversion and non-local means are core or UI-only workflows. |
| `islocalmax`, `findpeaks`, `findchangepts`, `xcorr` | Signal Processing Toolbox | `LineShift.m:43,48,51,81`; `Lineprofiler.m:94,107,119,133`; `fft_line_analysis.m:42`; `thresholder.m:181` | `scipy.signal.find_peaks`, correlation, and change-point implementation | Yes | `MinProminence`, width reference, and change-point statistic need matching tests. |
| `fft`, `fft2`, `ifft2`, `fftshift`, `ifftshift` | MATLAB base numerical functions | `align_trans.m:26,99,197,206`; `filter_movie.m:179,191`; `measureFRC.m:102-103`; several helpers | NumPy FFT / CuPy cuFFT | Yes | MATLAB normalization and axis conventions must be frozen before GPU work. |
| `fit`, `fitoptions`, `fittype` | Curve Fitting Toolbox | `LineShift.m:77,81`; `level.m:149,155`; `level_auto.m:119,122,150,153,166` | `scipy.optimize.curve_fit` / `lmfit` | Yes | Gaussian and custom log model parameter bounds/initialization are important. |
| `lsqcurvefit`, `optimoptions` | Optimization Toolbox | `localize.m:204-205` | `scipy.optimize.least_squares` | Yes | Bounds, termination, and parameter scaling must match. |
| `fminsearch` | MATLAB base numerical optimization | `align_rot.m:76` | `scipy.optimize.minimize(method="Nelder-Mead")` | Yes | Match simplex initialization and stopping tolerances where reproducibility matters. |
| `readtable`, `writetable`, `table2array`, `struct2table` | MATLAB table functionality (base MATLAB plus dependent producer functions) | `ReadAFMFile.m:532-533`; `exporter.m:61`; `AnalyzeAreas.m:53,55,79,81,121`; `open_NHF.m:27` | `pandas.read_csv/to_csv`, typed records, and explicit column schemas | Yes | Do not let pandas infer units or column ordering silently. |
| `h5info`, `h5read`, `h5readatt`, `h5create`, `h5write`, `H5F.open/close` | MATLAB HDF5 interface; documented built-in | `open_ARIS.m:7-8,16,33,36,87`; `open_NHF.m:15,43,47-50`; `open_h5.m:19,27,32,39`; `open_h5jpk.m:39,128`; `write_h5.m:44,48,54,58` | `h5py` | Yes | Preserve transposes, attributes, group paths, and missing-channel behavior. |
| `imread`, `imfinfo`, `imwrite` | MATLAB image I/O; documented built-in | `ReadAFMFile.m` and exporter helpers (scan all call sites during NL-10) | `tifffile`, Pillow, imageio | Yes | TIFF dtype, orientation, and multi-page behavior need fixtures. |
| `Tiff` class | MATLAB built-in TIFF class | `tiff_exporter.m:22-54`; `Mat_SimAFM/export_tiffs.m:8-32` | `tifffile.TiffWriter` | Yes | Floating-point, planar configuration, compression, and append semantics. |
| `uiputfile` | MATLAB base GUI | `exporter.m:34`; `Mat_SimAFM/export_tiffs.m:4`; GUI save helpers | CLI output path; napari file dialog later | No for core | Must not leak GUI concerns into the headless core. |
| `arrayfun`, `cellfun`, `struct2cell`, `regexp`, string helpers | MATLAB base language/runtime | `align_rot.m:76` and scattered parser/helpers | Python loops, comprehensions, `re`, dataclasses | Usually no | Treat as implementation details unless they affect ordering or parsing. |

## Non-core and external helper findings

| Finding | Evidence | Porting implication |
|---|---|---|
| Igor reader provenance | `NanoLocz-lib/open_IBW.m:1-9` says it is a modified `IBWread.m` | Preserve the upstream attribution/license of the embedded reader; verify whether its original terms permit redistribution. |
| Parallel Computing Toolbox | `Mat_SimAFM/parMat_SimAFM.m:24` uses `parfor` | The externalized core does not use `parfor`; replace batch parallelism with Dask/CUDA streams later. |
| GUI-only MATLAB APIs | `GUI/*.m` uses `uifigure`, `uilistbox`, `jsondecode`, drag/drop JavaScript bridge, and drawing controls | Exclude from P0/P1 core port. Napari replacement belongs to NL-41. |
| Simulation code duplication | `Mat_SimAFM/` duplicates many core functions | Audit once against `NanoLocz-lib/`; do not create two Python implementations. |
| README capability list | Upstream supports `.spm`, `.asd`, `.jpk`, `.h5-jpk`, `.ibw`, `.ARIS`, TIFF, NHF, GWY and exports TIFF/GIF/AVI/PNG/JPEG/PDF/TXT/CSV/XLS/H5 | NL-10 through NL-13 must separate required scientific inputs from presentation/export formats. |

## Licensing and attribution requirements

The upstream repository is marked GPLv3 and includes a GPL-3.0 license. A
redistributed derivative port must:

1. retain the GPL-3.0 license;
2. preserve applicable source headers and identify modifications;
3. retain NanoLocz attribution and cite the NanoLocz paper;
4. preserve attribution/license terms for the modified `IBWread` code and any other
   third-party code discovered during the full audit; and
5. document Python dependencies and their licenses before release.

Scientific citations requested by upstream:

- Heath, Micklethwaite & Storer, “NanoLocz: Image analysis platform for AFM,
  high-speed AFM and localization AFM,” *Small Methods* (2024),
  DOI: `10.1002/smtd.202301766`.
- Heath et al., “Localization atomic force microscopy,” *Nature* 594 (2021),
  DOI: `10.1038/s41586-021-03551-x`.

The simulation bridge additionally cites Amyot & Flechsig, *PLOS Computational
Biology* (2020), DOI: `10.1371/journal.pcbi.1008444`, when that module begins.

## Audit limitations and follow-ups

- This is a static source audit, not an execution trace. A later pass must run the
  MATLAB/Octave test data and capture actual code paths.
- The upstream README's toolbox list includes Statistics/ML, Bioinformatics, and
  Computer Vision, but this scan did not yet prove every listed product's use in
  `NanoLocz-lib`. Search the GUI and release/test data before declaring those
  dependencies droppable.
- No MATLAB or Octave runtime was available in this session, so no golden outputs
  were generated. NL-02 must establish the capture route before parity claims.
- The audit clone is intentionally not vendored into this project. Re-run the audit
  against the recorded upstream URL and commit revision if the source changes.
