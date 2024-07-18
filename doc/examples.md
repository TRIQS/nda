@page examples Examples

[TOC]

| Example | Description |
|---------|-------------|
| @ref ex1 | A short introduction to **nda** |
| @ref ex2 | Different ways to construct and get started with arrays |
| @ref ex3 | Different ways to initialize already constructed arrays |
| @ref ex4 | How to take views and slices of arrays |
| @ref ex5 | How to write/read arrays and views to/from HDF5 files |
| @ref ex6 | How to broadcast, gather, scatter and reduce arrays and views |
| @ref ex7 | How to use symmetries with **nda** arrays |
| @ref ex8 | Doing linear algebra with **nda** arrys |

@section compiling Compiling the examples

All examples have been compiled on a MacBook Pro with an Apple M2 Max chip with
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) 1.14.3,
- [open-mpi](https://www.open-mpi.org/) 5.0.1 and
- [OpenBLAS](https://www.openblas.net/) 0.3.27

installed via [homebrew](https://brew.sh/).

We further used clang 19.1.2 together with cmake 3.30.5.

Assuming that **nda** has been installed locally (see @ref installation) and that the actual example code is in a file
`main.cpp`, the following generic `CMakeLists.txt` should work for all examples (see also @ref integration):

```cmake
cmake_minimum_required(VERSION 3.20)
project(example CXX)

# set required standard
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find nda
find_package(nda REQUIRED CONFIG)

# build the example
add_executable(ex main.cpp)
target_link_libraries(ex nda::nda_c)
```
