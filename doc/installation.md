@page installation Installation

[TOC]

**nda** supports the usual installation procedure using CMake.

If you want to skip the installation step, you can go directly to @ref integration to see how you can integrate
**nda** into your own C++ project by using CMake's @ref fetch.

> **Note:** To guarantee reproducibility in scientific calculations, we strongly recommend the use of a stable
> [release version](https://github.com/TRIQS/nda/releases).


@section dependencies Dependencies

The dependencies of the C++ **nda** library are as follows:

* gcc version 12 or later OR clang version 15 or later OR IntelLLVM (icx) 2023.1.0 or later
* CMake version 3.20 or later (for installation or integration into an existing project via CMake)
* HDF5 library version 1.8.2 or later
* a working MPI implementation (openmpi and Intel MPI are tested)
* [OpenMP](https://www.openmp.org/)
* a BLAS/LAPACK implementation (e.g. [oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.9vs16x) or
[OpenBLAS](https://www.openblas.net/))

For specific functionality, the following optional dependencies might be required:

* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) for GPU support
* [MAGMA](https://icl.utk.edu/magma/) for some specific matrix operations on the GPU

**nda** fetches and builds the following libraries automatically (unless the user explicitly tells
CMake not to and to use local installations instead):

* [TRIQS/cpp2py](https://github.com/TRIQS/cpp2py)
* [TRIQS/itertools](https://github.com/TRIQS/itertools)
* [TRIQS/mpi](https://github.com/TRIQS/mpi)
* [TRIQS/h5](https://github.com/TRIQS/h5)

For the Python interface, additional dependencies are required:

* Python version 3.6 or later
* numpy version 1.11.0 or later
* mako version 0.9.1 or later
* scipy (version 1.11.3 is tested but older/newer versions should work as well)

The Python interface is built with [TRIQS/cpp2py](https://github.com/TRIQS/cpp2py).
Please refer to the [GitHub repository](https://github.com/TRIQS/cpp2py) for further information.


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
$ git checkout 1.2.0
```

and follow steps 2 to 4 above to compile the code.


@section cmake_options Custom CMake options

The compilation of **nda** can be configured by calling cmake with additional command line options

```console
$ cmake ../nda.src -DCMAKE_INSTALL_PREFIX=path_to_install_dir -DOPTION1=value1 -DOPTION2=value2 ...
```

The following options are available:

| Options                                 | Syntax                                            |
|-----------------------------------------|---------------------------------------------------|
| Specify an installation path            | ``-DCMAKE_INSTALL_PREFIX=path_to_install_dir``    |
| Build in Debugging Mode                 | ``-DCMAKE_BUILD_TYPE=Debug``                      |
| Disable testing (not recommended)       | ``-DBuild_Tests=OFF``                             |
| Build the documentation                 | ``-DBuild_Documentation=ON``                      |
| Build shared libraries                  | ``-DBUILD_SHARED_LIBS=ON``                        |
| Build benchmarks                        | ``-DBuild_Benchs=ON``                             |
| Test SSO memory optimizations           | ``-DBuild_SSO_Tests=ON``                          |
| Enable Python support                   | ``-DPythonSupport=ON``                            |
| Enable CUDA support                     | ``-DCudaSupport=ON``                              |
| Disable HDF5 support                    | ``-DHDF5Support=OFF``                             |
| Disable MPI support                     | ``-DMPISupport=OFF``                              |
| Disable OpenMP support                  | ``-DOpenMPSupport=OFF``                           |
| Enable MAGMA support                    | ``-DUse_Magma=ON``                                |
