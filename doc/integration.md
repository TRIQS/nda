@page integration Integration in C++ projects

[TOC]

**nda** is a compiled library.

You can either
* build and install it beforehand (see @ref installation) and then link against it (see @ref find_package) or
* let CMake fetch and build it directly as part of your project (see @ref fetch).

To use **nda** in your own `C++` code, you simply have to include the relevant header files.
For example:

```cpp
#include <nda/nda.hpp> // core library incl. linear algebra functionalities
#include <nda/mpi.hpp> // MPI support (optional)
#include <nda/h5.hpp>  // HDF5 support (optional)
#include <nda/sym_grp.hpp> // Symmetry support (optional)

// use nda
```


@section cmake CMake

@subsection fetch FetchContent

If you use [CMake](https://cmake.org/) to build your source code, you can fetch **nda** directly from the
[Github repository](https://github.com/TRIQS/nda) using CMake's [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
module:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project CXX)

# fetch from github
include (FetchContent)
FetchContent_Declare(
  nda
  GIT_REPOSITORY https://github.com/TRIQS/nda.git
  GIT_TAG        2.0.x
)
FetchContent_MakeAvailable(nda)

# declare a target and link to nda
add_executable(my_executable main.cpp)
target_link_libraries(my_executable nda::nda_c)
```

This will link automatically to all of **nda's** dependencies, except for the HDF5 C library.
If you need to use some of the HDF5 C library features, you can simply link to it via the `h5::hdf5` CMake target,
which is exported by the bundled [TRIQS/h5](https://github.com/TRIQS/h5) dependency.

Note that the above will also build [googletest](https://github.com/google/googletest) and the unit tests for **nda**.
To disable this, you can put `set(Build_Tests OFF CACHE BOOL "" FORCE)` before fetching the content or by specifying
`-DBuild_Tests=OFF` on the command line.


@subsection find_package find_package

If you have already installed **nda** on your system by following the instructions from the @ref installation page, you can also make
use of CMake's [find_package](https://cmake.org/cmake/help/latest/command/find_package.html) command.
This has the advantage that you don't need to download anything, i.e. no internet connection is required.
Furthermore, you only need to build the library once and can use it in multiple independent projects.

Let's assume that **nda** has been installed to `path_to_install_dir`.
Then linking your project to **nda** with CMake is as easy as

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project CXX)

# find nda
find_package(nda REQUIRED CONFIG)

# declare a target and link to nda
add_executable(my_executable main.cpp)
target_link_libraries(my_executable nda::nda_c)
```

In case, CMake cannot find the package, you might have to tell it where to look for the `nda-config.cmake` file by setting the variable
`nda_DIR` to `path_to_install_dir/lib/cmake/nda` or by sourcing the provided `ndavars.sh` before running CMake:

```console
$ source path_to_install_dir/share/nda/ndavars.sh
```


@subsection add_sub add_subdirectory

You can also integrate **nda** into your CMake project by placing the entire source tree in a subdirectory and call `add_subdirectory()`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project CXX)

# add nda subdirectory
add_subdirectory(deps/nda)

# declare a target and link to nda
add_executable(my_executable main.cpp)
target_link_libraries(my_executable nda::nda_c)
```

Here, it is assumed that the **nda** source tree is in a subdirectory `deps/nda` relative to your `CMakeLists.txt` file.
