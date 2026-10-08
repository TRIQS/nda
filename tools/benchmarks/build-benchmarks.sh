#!/bin/sh
# build-benchmarks.sh <source-dir> <build-dir>
# Configure nda with the benchmark_tracked preset from CMakePresets.json and build one
# target per benchmarks/tracked/*.cpp, matching what benchmarks/CMakeLists.txt globs.
# Shared by compare-benchmarks.sh and test-compare-benchmarks.sh.
set -eu
source=$1
build=$2

# compile_commands.json lets compare.py record the flags the benchmarks were really compiled with.
cmake --preset benchmark_tracked -S "$source" -B "$build" -DCMAKE_INSTALL_PREFIX="$build/install" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
set --
for file in "$source"/benchmarks/tracked/*.cpp; do
  [ -e "$file" ] && set -- "$@" "$(basename "$file" .cpp)"
done
[ $# -eq 0 ] || cmake --build "$build" --parallel "$(nproc)" --target "$@"
