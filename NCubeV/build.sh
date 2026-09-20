#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: ./build.sh <julia-command> [julia-args...]"
    echo "Example: ./build.sh julia +1.10"
    exit 1
fi

JULIA_CMD=("$@")

echo "Checking Julia version..."
JULIA_MM="$("${JULIA_CMD[@]}" -e 'print("$(VERSION.major).$(VERSION.minor)")')"
if [ "$JULIA_MM" != "1.10" ]; then
    echo "Error: NCubeV requires Julia 1.10.x (detected $JULIA_MM)."
    echo "Please run this script with a Julia 1.10 executable."
    exit 1
fi

echo "Pulling submodules..."
git submodule update --init --recursive

if [ ! -f "deps/OVERTFixed/Project.toml" ]; then
    echo "Error: deps/OVERTFixed is missing after submodule update."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

echo "Building NCubeV with: ${JULIA_CMD[*]}"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "${JULIA_CMD[@]}" --project=. -e 'using Pkg; Pkg.instantiate(verbose=true); Pkg.build(verbose=true);'
