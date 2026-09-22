#!/bin/sh
# build-benchmarks.sh <source-dir> <build-dir>
# Configure nda for benchmarks only and build one target per benchmarks/tracked/*.cpp,
# matching what benchmarks/CMakeLists.txt globs. Shared by
# run-benchmarks.sh and compare-benchmarks.sh so the build flags live in one place.
set -eu
source=$1
build=$2

cmake -S "$source" -B "$build" -DCMAKE_INSTALL_PREFIX="$build/install" \
  -DBuild_Benchs=ON -DBuild_Tests=OFF -DPythonSupport=OFF -DCMAKE_BUILD_TYPE=Release
set --
for file in "$source"/benchmarks/tracked/*.cpp; do
  set -- "$@" "$(basename "$file" .cpp)"
done
cmake --build "$build" --parallel "${PARALLEL:?Set PARALLEL}" --target "$@"
