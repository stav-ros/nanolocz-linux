#!/bin/bash
# Run all NanoLocz v1.0 benchmarks

set -e

echo "=========================================="
echo "NanoLocz v1.0 Benchmark Suite"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Run individual benchmarks
echo "Running I/O benchmark..."
python "${SCRIPT_DIR}/benchmark_io.py" || echo "I/O benchmark failed"

echo "Running preprocessing benchmark..."
python "${SCRIPT_DIR}/benchmark_preprocess.py" || echo "Preprocessing benchmark failed"

echo "Running detection benchmark..."
python "${SCRIPT_DIR}/benchmark_detection.py" || echo "Detection benchmark failed"

echo "Running tracking benchmark..."
python "${SCRIPT_DIR}/benchmark_tracking.py" || echo "Tracking benchmark failed"

echo "Running LAFM benchmark..."
python "${SCRIPT_DIR}/benchmark_lafm.py" || echo "LAFM benchmark failed"

echo "Running reconstruction benchmark..."
python "${SCRIPT_DIR}/benchmark_reconstruction.py" || echo "Reconstruction benchmark failed"

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "=========================================="
