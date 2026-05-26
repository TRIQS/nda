@page installation Installation

[TOC]

**nda** supports the usual installation procedure using CMake.

If you want to skip the installation step, you can go directly to @ref integration to see how you can integrate
**nda** into your own C++ project by using CMake's @ref fetch.

> **Note**: To guarantee reproducibility in scientific calculations, we strongly recommend the use of a stable
> [release version](https://github.com/TRIQS/nda/releases).


@section dependencies Dependencies

The required dependencies of the C++ **nda** library are:

* C++20 compatible compiler
* CMake version 3.20
* a BLAS/LAPACK implementation (e.g. [oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.9vs16x) or
[OpenBLAS](https://www.openblas.net/))

We do not provide explicit version requirements.
Instead, we refer the user to the **nda** CI builds on
[GitHub Actions](https://github.com/TRIQS/nda/actions), where recent OS and compiler versions are tested.

The following dependencies are **optional and enabled by default**. They can be turned off via the corresponding CMake
option (see @ref cmake_options):

* HDF5 library version 1.8.2 or later — for reading/writing arrays to/from HDF5 files (disable with `-DHDF5Support=OFF`)
* a working MPI implementation, e.g. openmpi or Intel MPI — for distributed-memory parallelism
  (disable with `-DMPISupport=OFF`)
* [OpenMP](https://www.openmp.org/) — for shared-memory parallelism (disable with `-DOpenMPSupport=OFF`)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) — for GPU support (enable with `-DCudaSupport=ON`)
* [MAGMA](https://icl.utk.edu/magma/) — for some specific matrix operations on the GPU (enable with `-DUse_Magma=ON`)

**nda** also depends on the following TRIQS libraries, which are fetched and built automatically (unless the user
explicitly tells CMake to use local installations instead):

* [TRIQS/itertools](https://github.com/TRIQS/itertools)
* [TRIQS/h5](https://github.com/TRIQS/h5) (only when `HDF5Support=ON`)
* [TRIQS/mpi](https://github.com/TRIQS/mpi) (only when `MPISupport=ON`)


@section install_steps Installation steps

1. Download the source code of the latest stable version by cloning the [TRIQS/nda](https://github.com/triqs/nda)
repository from GitHub:

    ```console
    $ git clone https://github.com/TRIQS/nda nda.src
    ```

2. Create and move to a new directory where you will compile the code:

    ```console
     $ mkdir nda.build && cd nda.build
    ```

3. In the build directory, call cmake including any additional custom CMake options (see below):

    ```console
    $ cmake ../nda.src -DCMAKE_INSTALL_PREFIX=path_to_install_dir
    ```

    Note that it is required to specify ``CMAKE_INSTALL_PREFIX``, otherwise CMake will stop with an error.

4. Compile the code, run the tests and install the application:

    ```console
    $ make -j N
    $ make test
    $ make install
    ```

    Replace `N` with the number of cores you want to use to build the library.


@section versions Versions

To choose a particular version, go into the directory with the sources, and look at all available versions:

```console
$ cd nda.src && git tag
```

Checkout the version of the code that you want:

```console
$ git checkout 2.0.0
```

and follow steps 2 to 4 above to compile the code.


@section cmake_options Custom CMake options

The compilation of **nda** can be configured by calling cmake with additional command line options

```console
$ cmake ../nda.src -DCMAKE_INSTALL_PREFIX=path_to_install_dir -DOPTION1=value1 -DOPTION2=value2 ...
```

The following options are available:

**Install and build basics**

| Option                       | Description                                          | Default   |
|------------------------------|------------------------------------------------------|-----------|
| `CMAKE_INSTALL_PREFIX=path`  | Installation path (required)                         | —         |
| `CMAKE_BUILD_TYPE=type`      | Build type (`Release`, `Debug`, `RelWithDebInfo`, …) | `Release` |
| `BUILD_SHARED_LIBS=ON/OFF`   | Build shared libraries                               | `OFF`     |

**Optional features**

| Option                       | Description                                          | Default |
|------------------------------|------------------------------------------------------|---------|
| `HDF5Support=ON/OFF`         | Build with HDF5 support                              | `ON`    |
| `MPISupport=ON/OFF`          | Build with MPI support                               | `ON`    |
| `OpenMPSupport=ON/OFF`       | Build with OpenMP support                            | `ON`    |
| `PythonSupport=ON/OFF`       | Build Python bindings                                | `OFF`   |
| `CudaSupport=ON/OFF`         | Build with CUDA (GPU) support                        | `OFF`   |
| `Use_Magma=ON/OFF`           | Enable batched GEMM via MAGMA                        | `OFF`   |

**Tests, benchmarks and documentation**

| Option                          | Description                                          | Default |
|---------------------------------|------------------------------------------------------|---------|
| `Build_Tests=ON/OFF`            | Build the unit tests (not recommended to disable)    | `ON`    |
| `Build_SSO_Tests=ON/OFF`        | Build the SSO / custom-allocator test variants       | `OFF`   |
| `Build_c2py_clair_Tests=ON/OFF` | Build cpp2py/clair converter tests                   | `OFF`   |
| `Build_Benchs=ON/OFF`           | Build the benchmarks                                 | `OFF`   |
| `Build_Documentation=ON/OFF`    | Build the Doxygen documentation                      | `OFF`   |

**Static analysis and sanitizers** (developer use; require Clang/LLVM)

| Option                    | Description                                              | Default |
|---------------------------|----------------------------------------------------------|---------|
| `ANALYZE_SOURCES=ON/OFF`  | Run static analyzers (clang-tidy, cppcheck) during build | `OFF`   |
| `ASAN=ON/OFF`             | Build with LLVM Address Sanitizer                        | `OFF`   |
| `UBSAN=ON/OFF`            | Build with LLVM Undefined Behavior Sanitizer             | `OFF`   |
| `MSAN=ON/OFF`             | Build with LLVM Memory Sanitizer                         | `OFF`   |
| `TSAN=ON/OFF`             | Build with LLVM Thread Sanitizer                         | `OFF`   |
