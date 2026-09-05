# NanoLocz Conda Build Instructions

## Prerequisites

Install conda-build and conda-verify:

```bash
conda install -c conda-forge conda-build conda-verify
```

Or with mamba:

```bash
mamba install -c conda-forge conda-build conda-verify
```

## Build the Package

From the `nanolocz-gpu` directory:

```bash
conda-build conda.recipe --output-folder dist/conda
```

This will:
1. Create a source tarball
2. Build the noarch Python package
3. Run tests in an isolated environment
4. Output conda packages to `dist/conda/`

## Test the Package Locally

```bash
conda install --use-local -c dist/conda nanolocz
```

Or create a test environment:

```bash
conda create -n test-nanolocz -c dist/conda nanolocz
conda activate test-nanolocz
nanolocz --help
```

## Upload to Anaconda.org

```bash
anaconda login
anaconda upload dist/conda/noarch/nanolocz-0.1.0.dev0-py_0.tar.bz2
```

## GPU Variant (Optional)

For GPU support with CuPy, create a separate recipe or add variants:

```yaml
# conda.recipe/variants.yaml
cupy:
  - cupy-cuda11x
  - cupy-cuda12x
```

Then build with:

```bash
conda-build conda.recipe --variant-config-file conda.recipe/variants.yaml
```

## Notes

- The package is built as `noarch: python` for cross-platform compatibility
- CUDA/CuPy dependencies are optional and not included in the base recipe
- Tests require pytest and are run in an isolated conda environment
- The recipe uses the local source tree (path: ..)
