#!/usr/bin/env bash
# Build helper. Works on Linux (x86_64 and aarch64) and macOS.
#
# Prereqs:
#   apt:   sudo apt install build-essential cmake libeigen3-dev   (Ubuntu / Debian)
#   brew:  brew install cmake eigen                                (macOS)
#   pip:   pip install pybind11 numpy                              (always)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${HERE}/build"

PYBIND_CMAKE_DIR=$(python3 -m pybind11 --cmakedir 2>/dev/null || true)
if [ -z "${PYBIND_CMAKE_DIR}" ]; then
    echo "ERROR: pybind11 not installed. Run: pip install pybind11" >&2
    exit 1
fi

cmake -S "${HERE}" -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="${PYBIND_CMAKE_DIR}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" -j

echo
echo "Built: ${HERE}/python/dr_mmse/dr_mmse_cpp*.so"
echo
echo "Quick check:"
PYTHONPATH="${HERE}/python:${PYTHONPATH:-}" \
    python3 -c "from dr_mmse import solve_dr_mmse_tac; print('import OK:', solve_dr_mmse_tac)"
