#!/bin/sh
# build-benchmarks.sh <source-dir> <build-dir>
# Configure nda with the benchmark_tracked preset from CMakePresets.json and build one
# target per benchmarks/tracked/*.cpp, matching what benchmarks/CMakeLists.txt globs.
# Shared by compare-benchmarks.sh and test-compare-benchmarks.sh.
set -eu
source=$1
build=$2

cmake --preset benchmark_tracked -S "$source" -B "$build" -DCMAKE_INSTALL_PREFIX="$build/install"
set --
for file in "$source"/benchmarks/tracked/*.cpp; do
  set -- "$@" "$(basename "$file" .cpp)"
done
cmake --build "$build" --parallel "$(nproc)" --target "$@"
