@page documentation API Documentation

[TOC]

The **nda** library offers efficient and flexible multi-dimensional array classes and associated functionality.

Efficiency is achieved by
* lazy expressions to delay evaluation and to remove redundant temporaries,
* views to access only part of already existing arrays and to avoid unnecessary allocations,
* evaluating expressions as much as possible at compile-time by using type traits and concepts,
* compile-time guarantees that specify the memory layout of arrays and views,
* using optimized libraries like BLAS, LAPACK or MAGMA,
* making use of known symmetries,
* and more.

Flexibility is achieved by
* general memory layouts that allow different strides and stride orders (e.g. C-order vs. Fortran-order),
* memory handles that can manage memory on the heap, the stack and on other devices like GPUs,
* the possibility to specify sizes and shapes at compile-time, at runtime or both,
* simulating Python's `:` and `...` syntax for easy access to slices of an array/view,
* and more.

The following provides a detailed reference documentation grouped into logical units.

If you are looking for a specific function, class, etc., try using the search bar in the top left corner.

## Arrays and views

@ref arrays_views form the backbone of the **nda** library. The generic array (nda::basic_array) and view
(nda::basic_array_view) classes are used to store and access multi-dimensional data and they come with a wide range
of operations and functions to manipulate them:

- @ref nda::basic_array is the generic array class in **nda**. Regular arrays own the memory they use for storing
the data. That means they are responsible for managing their memory storage.
- @ref nda::basic_array_view is the generic view class in **nda**. In contrast to regular arrays, a view does
not own the data. This makes it very lightweight and efficient.
- A few @ref av_algs, similar to the ones found in the standard library header `<algorithm>`, are implemented for arrays
and views.
- @ref av_ops provide the usual arithmetic operations for arrays and views. Note that the behavior of these operations
depends on the algebra of the involved objects (see @ref nda::get_algebra). They are all evaluated lazily (with few
exceptions).
- @ref av_utils contain type traits, concepts and more functionality related to array and view types.
- @ref av_factories create new array/view objects or transform existing ones.
- @ref av_hdf5 allows us to write/read arrays and views to/from HDF5 files.
- @ref av_mpi allows us to broadcast, scatter, gather and reduce arrays and views across/over multiple processes using
MPI.
- Various @ref av_math are implemented for arrays and views. They are all evaluated lazily (with few exceptions).
- @ref av_sym contain tools to use and detect symmetries in nda::Array objects.
- @ref av_types provide convenient aliases for the most common array and view types.

## CLEF - Compile-time lazy expressions and functions

@ref clef are called lazy because they are not evaluated right away. Instead, they usually return some proxy, i.e. lazy,
object which can be used to create new lazy expressions and which can be evaluated whenever the result is actually
needed.

- @ref clef_autoassign uses lazy expressions and placeholders to simplify assigning values to multi-dimensional
arrays/views and other container like objects.
- @ref clef_utils contain various internally used type traits as well as overloads of the `operator<<` to output lazy
objects to a `std::ostream`.
- @ref clef_eval can be done by calling the generic nda::clef::eval function which forwards the evaluation to
specialized evaluators.
- @ref clef_expr provide the classes that represent these lazy objects and the tools to create them.
- @ref clef_placeholders are lazy objects that can be used in lazy expressions. When a lazy expression is evaluated,
one usually assigns a value to the placeholders which are then plugged into the expression.

## Linear algebra

**nda** offers some @ref linalg related functionality:

- A @ref linalg_blas to parts of the BLAS library to work with arrays and views.
- An @ref linalg_lapack to parts of the LAPACK library to work with arrays and views.
- Some generic @ref linalg_tools that wrap the BLAS and LAPACK interfaces to provide a more user-friendly approach for
the most common tasks, like matrix-matrix multiplication, matrix-vector multiplication, eigen decompositions, etc.

## Tensor support

@ref tensor extends **nda** beyond matrix-level linear algebra by exposing general tensor operations on top of two
external backends — [TBLIS](https://github.com/MatthewsResearchGroup/tblis) for host (CPU) execution and
[cuTENSOR](https://docs.nvidia.com/cuda/cutensor/) for device (CUDA) execution. The backend is selected automatically
based on the address space of the input arrays, and a generic nda fallback is available for a subset of the
operations when neither backend is configured.

Most operations are parameterised by Einstein-notation index strings (e.g. `"ij"`, `"ijk"`), so contractions, axis
permutations and reductions are expressed in a uniform way independent of the rank of the tensors involved.

- @ref tensor_utils provide supporting types and functions used by various tensor operations.
- @ref tensor_ops collect the user-facing operations.

## Memory layout

@ref layout contains tools that allow us to specify how the data of a multi-dimensional array (nda::basic_array) or view
(nda::basic_array_view) is laid out in memory and how we can access it in a generic, i.e. independent of the actual
memory layout, way.

- @ref layout_pols specify the various layout policies that can be used with nda::basic_array and nda::basic_array_view.
- @ref layout_utils provide bounds checking, `for_each` functions to loop over multi-dimensional indices, structs that
mimic Python's `:` and `...` syntax and more.
- @ref layout_idx is used to map multi-dimensional indices to a linear/flat index for a given memory layout and to
calculate the resulting memory layout when taking slices of already existing arrays/views.

## Memory management

@ref memory tools are classes, functions and other definitions that help with the management of different kinds of
memory.

- @ref mem_addrspcs define the currently supported address spaces and tools to work with them.
- @ref mem_allocators implement custom allocators that satisfy the nda::mem::Allocator concept.
- @ref mem_handles are responsible for managing specific blocks of memory.
- @ref mem_pols specify the various memory managing policies that can be used with nda::basic_array and
nda::basic_array_view.
- @ref mem_utils provide generic versions (w.r.t. the different address spaces) of memory managing functions like
@ref nda::mem::malloc "malloc", @ref nda::mem::memcpy "memcpy" or @ref nda::mem::memset "memset" as well as other tools
related to memory management.

In most cases, users shouldn't use any of the memory management tools directly. They are used internally by
nda::basic_array and nda::basic_array_view to take care of their memory needs.

However, some advanced users might be interested in the different @ref mem_pols.

## Testing tools

@ref testing for checking nda::basic_array and nda::basic_array_view objects with googletest.

## Utilities

@ref utilities contain general @ref utils_type_traits and @ref utils_concepts as well as @ref utils_std and tools to
work with @ref utils_perms.
