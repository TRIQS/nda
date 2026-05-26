@page changelog Changelog

## Version 2.0.0

NDA Version 2.0.0 is a release that
* Significantly expands the `nda::linalg` API with eigenvalue, QR, LU, SVD and linear-solve routines
* Adds new LAPACK and BLAS wrappers (`geev`, `syev`/`heev`, `sygv`/`hegv`, `gerc`)
* Makes HDF5, MPI and OpenMP optional build-time dependencies
* Introduces deep partial evaluation for CLEF expressions
* Adds a Hadamard product for arrays and `std::vector`
* Migrates the documentation pipeline fully to Doxygen and ships extensive Doxygen-rendered C++ docs with worked, compiled examples
* Moves the c2py converters into `nda/c2py` and fixes several Python ↔ nda conversion edge cases
* Fixes several library issues

We thank all contributors: Marco Barbone, Thomas Hahn, Alexander Hampel, Sergei Iskakov, Jason Kaye, Dominik Kiese, Harrison LaBollita, Henri Menke, Miguel Morales, Olivier Parcollet, Dylan Simon, Nils Wentzell

Find below an itemized list of changes in this release.

### General
* Move c2py converters + additional files to nda/c2py
* Add `is_expression` trait for `expr`, `expr_unary` and `expr_call` types
* Remove nda++ compiler wrapper
* Add a function to sum over axes of an nda::Array and treat the empty-axis edge case
* Change `static_extents` encoding from 4-bit to 8-bit per extent
* Add deep partial evaluation for CLEF expressions (#93)
* Move CLEF lazy function evaluators for `det`/`inv` into the proper namespace
* Default to `double` value type in factory functions nda::eye, nda::zeros, nda::ones
* Fix CTAD deduction guide to preserve address space for array expressions
* Add compile-time check for device array assignment compatibility
* Change `get_first_element` to not add constness to its argument; in array cross-construction invoke explicit `ValueType` conversion
* Add a Hadamard product implementation for arrays and `std::vector` (#79)
* Generalize nda::sum and nda::product to allow arrays of arrays
* Add nda::reciprocal for elementwise `1.0/x`
* Add `std::initializer_list` overload for nda::diag and allow temporaries in nda::diagonal
* Add nda::AnyOf concept to nda/concepts.hpp
* Fix nda::CallableWithLongs concept for gcc13/14
* Redefine `is_contiguous`/`is_strided_1d` and introduce `has_positive_strides` in nda::idx_map
* Fix bug in assignment of a contiguous range to a view and disallow assignment to const arrays/views
* Fix ambiguous-assignment-operator issue, fix issues with 1D strided views with negative strides, and add tests
* Fix issue with nda::reshape/nda::flatten for arbitrary layouts and change the default shape/strides of nda::idx_map
* Add device implementations for `fill`, `fill_n`, `fill2D`, `fill_with_scalar`, `assign_from_scalar` and make them C++20-compatible
* Make mapped functions work with nda::Array as well as nda::Scalar objects
* Add empty braces to the `EXPECTS`, `ASSERT` and `ENSURES` macros
* Update the h5 interface to match changes in the h5 library and simplify nda/h5.hpp
* Update the MPI routines, move C-style MPI routines into the public namespace, add mpi/utils.hpp and extensive tests
* Make HDF5, MPI and OpenMP optional build-time dependencies (#94)
* Bring back the `bad_alloc` test and fix UBSAN/MSAN positives in factories, `basic_array_and_view`, `gemm_generic`/`gemv_generic`
* Remove unused `layout/rect_str.hpp` and clean up includes in basic_array.hpp and basic_array_view.hpp
* Small improvements to random array generators
* Update Apache copyright headers to a minimal form for all files; move copyright notice into a separate `COPYRIGHT` file; default copyright to the Simons Foundation
* Fix issue in nda::sym_grp MPI parallelization after adding optional MPI support

### blas/lapack
* Add nda::linalg::eig, nda::linalg::eig_in_place, nda::linalg::eigvals, nda::linalg::eigvals_in_place backed by a new nda::lapack::geev wrapper
* Add nda::linalg::eigh, nda::linalg::eigvalsh, including an overload for the generalized eigenvalue problem, backed by new nda::lapack::syev, nda::lapack::heev, nda::lapack::sygv and nda::lapack::hegv wrappers
* Add nda::linalg::qr, nda::linalg::qr_in_place, nda::linalg::lu, nda::linalg::lu_in_place, nda::linalg::solve, nda::linalg::solve_in_place, nda::linalg::svd, and nda::linalg::svd_in_place
* Add an outer-product function to `nda::linalg`; move `dot_generic` from `linalg::detail` into `nda::linalg` with a docstring
* Add nda::blas::gerc to `blas/ger.hpp` and improve nda::blas::ger and nda::blas::gerc test clarity and coverage
* Split `linalg/det_and_inverse.hpp` into linalg/det.hpp and linalg/inv.hpp; move nda::is_matrix_square and nda::is_matrix_diagonal to matrix_functions.hpp; move small-size `inv_in_place` optimizations to the `detail` namespace
* Relax the `getrf` check in `linalg::det_in_place` and fix the nda::lapack::gelss path for underdetermined systems
* Fix segfaults in cuBLAS calls; add device-implementation check in lapack/gesvd.hpp; add a test for linalg routines on the device
* Generalize `get_ld`/`get_ncols` in blas/tools.hpp, simplify `blas::get_op` to deduce flags from the nda::Matrix type, and add `blas::get_array`
* Make function-argument names consistent (lowercase) across the BLAS, LAPACK, cuBLAS and cuSOLVER interfaces
* Fix `gemm_vbatch` signature for fallback functions; fix lowercase/uppercase mismatch in FORTRAN prototypes; fix Intel vs GNU ABI when building against MKL

### cmake
* Downgrade required C++ standard from 23 to 20
* Directly use the imported targets provided for HDF5
* Remove `PythonSupport` requirement for building docs

### jenkins
* Synchronize Jenkinsfile with app4triqs
* Fix file issue in Jenkins Dockerfile creation and fix the environment setting for osx builds

### ghactions
* Use Ninja for parallel builds; modernize and simplify `build.yml`
* Synchronize with app4triqs/notriqs branch and always build against the respective TRIQS branch
* Bump OSX and gcc to 15; update runner images and compiler version
* Make sure `SDKROOT` is set in the OSX environment; link against brew's `libomp` and `libc++` for macos+clang
* Use full `BUILD_CONFIG` for unique ccache keys and avoid cache-key collisions across retries
* Update the Ubuntu package list to use OpenBLAS over liblapack
* Remove the custom Doxygen build and build docs on macos
* Generate and deploy test-coverage information

### doc
* Switch the documentation pipeline fully to Doxygen and update the Doxyfile to v1.16.1
* Add worked code examples to the doc folder, ensure they compile in CI
* Add an FI support notice to `README.md`
* Remove clang-specific setting from `Doxyfile.in`

### python support
* Add c2py converters and additional files into `nda/c2py`
* Remove any pybind11 references

### docker
* Update the intel image in `Dockerfile.ubuntu-intel`
* Update `Dockerfile.msan` to the latest llvm and library versions
* Synchronize Dockerfiles with triqs
* Rename `Dockerfile.build` to `Dockerfile` for consistency with triqs apps


## Version 1.3.0

NDA Version 1.3.0 is a release that
* Adds new functionality to the symmetry library
* Adds extensive doxygen documentation to the public API
* Generates reference documentation for website using doxygen
* Optimizes CLEF by removing redundant copies
* Extends the LAPACK/BLAS functionality
* Improves the interface to read/write HDF5 files
* Fixes several library issues

We thank all contributors: Thomas Hahn, Alexander Hampel, Sergei Iskakov, Jason Kaye, Dominik Kiese, Harrison LaBollita, Henri Menke, Miguel Morales, Olivier Parcollet, Dylan Simon, Nils Wentzell

Find below an itemized list of changes in this release.

### General
* Allow nda::flatten to be called for non-trivial memory layouts
* Fix bug in for_each_static_impl, clean up layout/for_each.hpp and add tests
* Fix bug in encode function, clean up layout/permutation.hpp and add tests
* Fix Intel vs GNU ABI when building against MKL
* Fix bug in clef evaluator and add corresponding tests
* Improve h5 interface by removing unnecessary copies and restrictions and update tests
* Remove redundant files c++/nda/TODO and c++/nda/mapped_functions.vim
* Avoid redundant copies in clef evaluation
* Add sym_grp functions to obtain / initialize from representative data
* Add nda::concatenate implementation
* Update Dockerfile.msan to more recent Ubuntu and library versions
* Generalize nda::transpose to work with arrays of arbitrary rank
* Protect nda::reshape, should not allow non-standard memory order
* Remove redundant ARRAY_INT definition
* Add layout_to_policy trait specialization for Fortran layouts
* Use Default Seed for std::mt19937 construction
* Throw std::bad_alloc for failed memory allocation in handle and basic_array construction
* Various smaller code readability improvements
* Add missing includes in various files
* Regenerate Apache copyright headers
* Format .clang-tidy

### doc
* Add extensive documentation of public API using Doxygen doc strings
* Use doxygen to automatically generate documentation

### cmake
* Disable -ffast-math by default for intel compilers
* Use GNUInstallDirs in install commands
* Use CPP2PY_PYTHON_xxx variables instead of PYTHON_xxx
* Fix install command to include hxx files
* Set policy CMP0144 to new
* Improve logic when MKL is detected as LAPACK distribution
* Run NDA checks also in RelWithDebInfo build mode
* Disable finite-math-only for IntelLLVM compiler
* When using mkl enforce single dynamic lib and explicitly set mkl_interface_layer (#48)
* Disable BadAlloc test when ASAN is ON

### jenkins
* Enable PythonSupport on all CIs
* Add ubuntu-intel build
* Fix ubuntu-intel gpg file in Dockerfile

### ghactions
* Build and deploy documentation

### blas/lapack
* Fix type error in magma interface
* Add implementations and tests for geqp3, orgqr, ungqr of LAPACK
* Allow right hand side object lapack::gelss to be a vector or a matrix (#56)
* Allow temporary views in call to lapack::getrs
* Fix MKL version check when using BLAS gemm_batch_strided


## Version 1.2.0

NDA Version 1.2.0 is a release that
* Introduces NVidia GPU support for array and view types
* Adds GPU blas/lapack backends using the CuBLAS and CuSOLVER backend
* Allows the use of symmetries for initialization and symmetrization of arrays
* Uses C++20 concepts to constrain generic function implementations
* Enables sliced hdf5 read/write operations
* Fixes several library issues

We thank all the people who have contributed to this release: Thomas Hahn, Alexander Hampel, Dominik Kiese, Sergei Iskakoff, Harrison LaBollita, Henri Menke, Miguel Morales, Olivier Parcollet, Dylan Simon, Nils Wentzell

Find below an itemized list of changes in this release.

### General
* Add new test for matmul with permuted view
* Add flatten function to layout_transforms.hpp
* Add generic p-norm function
* Add bindings for batched GEMM through gemm_batch function
* Add support for slicing in h5_read
* Enable fast small matrix inverse for matrices of size 1x1, 2x2, 3x3
* Enable contiguous memory traversal for array iteration
* Merge reshape_view and reshape, add overload that takes list of integers
* Allow non-contiguous views in hdf5 read/write
* Generalize nda::rand for complex value_t
* Generalize basic_array_view deduction guide from contiguous range
* Unify public member-types between basic_array and basic_array_view
* In nda::blas::outer_product check contiguity only at runtime
* Allow construction of array views from std::array
* Define algebra of array_adapter as 'A'
* Generalize get_first_element for scalar types
* Add test for various nda traits
* Generalize get_rank for contiguous_range types
* Enable slicing with ranges that have negative steps + test
* Make basic_array(idx_map, &&mem_handle) constructor public
* Allow temporaries in calls to lapack wrapping functions
* Generalize nda_lapack test to run both double and complex versions
* Add blas::has_C_layout and blas::has_F_layout traits and use for cleanup
* Generalize nda_blas and nda_cublas test for various value_t and layout combinations
* In make_regular do not invoke copy of regular arrays
* make_regular now returns a decltype(basic_array{...})
* make_regular converts types with a regular_t member type
* get_regular_t<T> now uses basic_array{T} instead of make_regular
* In transpose(A) allow for unary expr_call arguments
* Allow basic_array rvalues in basic_array_view constructor
* Extend deduction guides for basic_array and basic_array_view
* Fix preservation of layout properties in idx_map.transpose(permutation)
* Rename Layout to LayoutPolicy in array/view template parameters
* Short-circuit in assign_from_ndarray for empty arrays
* Generalize most traits to apply equally to A and A&
* ArrayInitializer concept is now templated on the array type it initializes
* Add benchmark for array copy operations
* Allow for copy of block-strided arrays from host/device to host/device + test
* Restore handle_sso copy constructor
* Add get_view_t<T> trait
* Add Automatic include for c2py
* Add function is_matrix_diagonal
* Add stack_vector and stack_matrix alias
* Generalize operator== for idx_maps of different types
* Allow discarding info return value for lapack functions
* Generalize nda::diag for types matching the contiguous_range concept
* Generalize clef expr for multiarg subscript
* Use range::all over default constructed range
* Remove REQUIRES macro and use 'requires'
* Enable slicing also for h5_write operations, assume existing dataset
* Make pivot array const in getri signature
* Minor cleanup in nda/h5.hpp template constraints and doc
* Allow to pass dimensions as integers to factory functions basic_array::ones/zeros/rand
* Add the 1d array factory nda::arange mimicking numpy arange + test
* Allow bound checks also for array.extent(int) function
* Configure and install nda/version.hpp header
* Synchronize clang-tidy config file with app4triqs
* Add bugprone checks to clang-tidy
* Allow multiplication of std::array<T,N> by a T
* Regenerate GPL copyright headers for C++ files
* Fix compiler and linter warnings
* Various documentation improvements
* Clang-format all source files
* General cleanup

### cmake
* Find and Link against openmp
* Add generation of and install nda++ compiler wrapper
* Do not use Accelerate Framework on OSX
* Some cleanup in STATUS messages
* Pick up existing LAPACK_ROOT on OSX builds
* Only build Benchmarks if not subproject and not sanitizing
* Do not build documentation as subproject
* Do not find CUDAToolkit twice in nda-config.cmake
* Use google-bench main branch
* Link both cudart and cublas using imported targets provided by CUDAToolkit
* Update Findsanitizer.cmake to include TSAN and MSAN
* Disable Python Support by default
* Only find cpp2py when built with PythonSupport=ON
* Install cpp2py, needed as a linktime dependency for nda_py
* Fix llvm package version for ubuntu clang ghactions build
* In nda-config.cmake.in find Cpp2Py before including exported targets
* Fix issue in extract_flags.cmake where generator expressions where not properly removed from flags
* Fix Findsanitizer.cmake for new asan/ubsan runtime location with clang13+
* Add missing find_dep(Cpp2Py 2.0) to nda-config.cmake.in
* General cleanup

### Concepts
* Use C++20 concepts to constrain various generic functions and classes
* Introduce concepts: Array, MemoryArray, Matrix, Vector, Handle
* Various concept related simplifications and refactoring

### GPU Support
* Introduce GPU support for arrays and views
* Added traits to check address space compatibility
* Magma vbatch bindings + test + benchmark
* Cublas backend for dot, gemm, gemv and ger + test
* Cusolver backend for gesvd, getrf and getrs + test
* Add helper functions to_host/to_device/to_unified for the copy of a MemoryArray to different address space
* Add traits mem::on_host<T>, mem::on_device<T>, mem::on_unified<T> to test memory location
* Add traits get_regular_host_t, get_regular_device_t and get_regular_unified_t
* Generic get_addr_space variable template
* Add variable template have_same_address_space<A0, A1, ..>
* Allow multiple arguments to on_host, on_device traits. Add on_unified trait
* Add address space generic memory operations: malloc, free, memset, memcpy

### Symmetries
* Add sym_grp class to perform symmetry operations on arrays
* Add extensive tests for sym_grp
* Allow for OpenMP parallelized array initialization
* 'symmetrize' method to symmetrize an existing array
* 'init' method to init an array with the minimum number of rhs evaluations

### jenkins
* Specificy LAPACK_ROOT for osx builds
* Update docker base images
* Don't keep / publish any nda install
* Synchronize Jenkinsfile with app4triqs

### lapack/blas
* General cleanup and doc improvements in bindings
* Use concepts in generic lapack bindings
* When possible use gemm/gemv with op='C' when passing conj(M)
* Simplify logic in gemm
* Rename 'trans' to 'op' in the blas bindings

### Doc
* Add document on design principles for arrays, views and lazy expressions
* Add link instructions for cmake based projects
* Provide a link to install instructions in README.md
* Add additional build options to doc/install.rst

### Fixes
* Fix signature of expr::operator[]
* Fix bug in lapack::getrs for C-layout matrix input
* Fix const issue in map_layout_transform for rvalues
* Fix operation char for non-fortran layout in getrs
* Fix gcc compilation issue in assignment between tuple and std::array
* Promote memory layout in matrix multiplication
* Add Workaround for gcc11 bug
* In nda::memcpy make sure to take src as a 'const *'
* Do not create views from temporary arrays in gelss_worker
* Fix issue in expr_call implementation for the slicing case
* Fix bug in is_contiguous and is_strided_1d for Fortran layout arrays
* Fix issue when calling h5::write for array_view<const T>
* Matrix * Vector now returns a Vector and not a 1d array
* Bugfix in print for arrays of rank>2
* Avoid narrowing conversions in std::accumulate in multiple places
* Add missing operator- to stdutil/array.hpp
* Enable bound-checking for range-based array slicing FIX #22
* Fix out of bounds error in lapack::gtsv


## Version 1.1.0

nda is a C++ library providing an efficient and flexible multi-dimensional array class.
It is an essential building-block of the TRIQS project. Some features include
* coded in C++20 using concepts
* expressions are implemented lazily for maximum performance
* flexible and lightweight view-types
* matrix and vector class with BLAS / LAPACK backend
* easily store and retrieve arrays to and from hdf5 files using [h5](https://github.com/TRIQS/h5)
* common mpi functionality using [mpi](https://github.com/TRIQS/mpi)

This is the initial release for this project.

We thank all the people who have contributed to this release: Philipp Dumitrescu, Alexander Hampel, Olivier Parcollet, Dylan Simon, Hugo U. R. Strand, Nils Wentzell
