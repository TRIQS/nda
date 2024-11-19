@page changelog Changelog

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
