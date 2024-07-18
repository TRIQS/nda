@page ex6 Example 6: MPI support

[TOC]

In this example, we show how to scatter, gather, broadcast and reduce **nda** arrays and views across MPI processes.

**nda** uses the [mpi](https://triqs.github.io/mpi/unstable/index.html) library to provide MPI support.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <nda/mpi.hpp>
#include <mpi/mpi.hpp>
#include <iostream>

template <typename A>
void print(const A &arr, const mpi::communicator &comm) {
  std::cout << "Rank " << comm.rank() << ": " << arr << std::endl;
}

int main(int argc, char *argv[]) {
  // initialize MPI environment and communicator
  mpi::environment env(argc, argv);
  mpi::communicator comm;
  const int root = 0;

  // code snippets go here ...
}
```

The examples below are run on 4 processes.

> **Note**: Only regular arrays and views are allowed in the **nda** MPI routines, no lazy expressions. You can use
> nda::make_regular to turn your lazy expressions into regular arrays.

@subsection ex6_p1 Broadcasting an array/view

Let us first default construct an array on all MPI ranks and then resize and initialize it on rank 0, the root rank:

```cpp
// create an array and initialize it on root
auto A = nda::array<int, 2, nda::F_layout>();
if (comm.rank() == root) {
  A.resize(2, comm.size());
  for (int i = 0; auto &x : A) x = i++;
}
print(A, comm);
comm.barrier();
```

Output:

```
Rank 2:
[]
Rank 3:
[]
Rank 0:
[[0,2,4,6]
 [1,3,5,7]]
Rank 1:
[]
```

As you can see, the array is only initialized on root.
The reason why we use Fortran-order will be explained below.

> **Note**: The `comm.barrier()` is not necessary. It makes sure that the `print` statements are executed and finished
> before we proceed.

Broadcasting the array from the root to all other ranks is easy:

```cpp
// broadcast the array from root to all other ranks
mpi::broadcast(A, comm);
print(A, comm);
comm.barrier();
```

Output:

```
Rank 2:
[[0,2,4,6]
 [1,3,5,7]]
Rank 3:
[[0,2,4,6]
 [1,3,5,7]]
Rank 0:
[[0,2,4,6]
 [1,3,5,7]]
Rank 1:
[[0,2,4,6]
 [1,3,5,7]]
```

Although the output might not be in order, it is clear that the array is the same on all ranks after broadcasting.

Also note that the `broadcast` call automatically resizes arrays if necessary.
This is not true for views since they cannot be resized.
The caller is responsible for making sure that the shapes of the views are correct (otherwise an exception is thrown).

Before we broadcast a view, let us again prepare our arrays:

```cpp
// prepare the array
A = 0;
if (comm.rank() == root) {
  A(nda::range::all, root) = 1;
}
print(A, comm);
comm.barrier();
```

Output:

```
Rank 0:
[[1,0,0,0]
 [1,0,0,0]]
Rank 2:
[[0,0,0,0]
 [0,0,0,0]]
Rank 1:
[[0,0,0,0]
 [0,0,0,0]]
Rank 3:
[[0,0,0,0]
 [0,0,0,0]]
```

Now we want to broadcast the first column of the array on root to the \f$ i^{th} \f$ column of the array on rank
\f$ i \f$:

```cpp
// broadcast the first column from root to all other ranks
auto A_v = A(nda::range::all, comm.rank());
mpi::broadcast(A_v, comm);
print(A, comm);
comm.barrier();
```

Output:

```
Rank 1:
[[0,1,0,0]
 [0,1,0,0]]
Rank 0:
[[1,0,0,0]
 [1,0,0,0]]
Rank 2:
[[0,0,1,0]
 [0,0,1,0]]
Rank 3:
[[0,0,0,1]
 [0,0,0,1]]
```

If `A` would not be in Fortran-order, this code snippet wouldn't compile since the elements of a single column are not
contiguous in memory in C-order.

> **Note**: All MPI routines have certain requirements for the arrays/views involved in the operation. Please check out
> the documentation of the individual function, e.g. in this case nda::mpi_broadcast, if you have doubts.

@subsection ex6_p2 Gathering an array/view

Suppose we have a 1-dimensional array with rank specific elements and sizes:

```cpp
// prepare the array to be gathered
auto B = nda::array<int, 1>(comm.rank() + 1);
B() = comm.rank();
print(B, comm);
comm.barrier();
```

Output:

```
Rank 1: [1,1]
Rank 3: [3,3,3,3]
Rank 0: [0]
Rank 2: [2,2,2]
```

Let us gather the 1-dimensional arrays on the root process:

```cpp
// gather the arrays on root
auto B_g = mpi::gather(B, comm, root);
print(B_g, comm);
comm.barrier();
```

Output:

```
Rank 2: []
Rank 3: []
Rank 0: [0,1,1,2,2,2,3,3,3,3]
Rank 1: []
```

Only the root process will have the gathered result available.
If the other ranks need access to the result, we can do an all-gather instead:

```cpp
// all-gather the arrays
auto B_g = mpi::gather(B, comm, root, /* all */ true);
```

which is equivalent to

```cpp
auto B_g = mpi::all_gather(B, comm);
```

Views work exactly the same way.
For example, let us gather 2-dimensional reshaped views of 1-dimensional arrays:

```cpp
// resize and reshape the arrays and gather the resulting views
B.resize(4);
B = comm.rank();
auto B_r = nda::reshape(B, 2, 2);
auto B_rg = mpi::gather(B_r, comm, root);
print(B_rg, comm);
comm.barrier();
```

Output:

```
Rank 1:
[]
Rank 0:
[[0,0]
 [0,0]
 [1,1]
 [1,1]
 [2,2]
 [2,2]
 [3,3]
 [3,3]]
Rank 2:
[]
Rank 3:
[]
```

Higher-dimensional arrays/views are always gathered along the first dimension.
The extent of the first dimension does not have to be the same on all ranks.
For example, if we gather \f$ M \f$ arrays of size \f$ N_1^{(i)} \times \dots \times N_d \f$, then the resulting array
will be of size \f$ \prod_{i=1}^M N_1^{(i)} \times \dots \times N_d \f$ .

nda::mpi_gather requires all arrays/views to be in C-order.
Other memory layouts have to be transposed or permuted (see nda::transpose and the more general
nda::permuted_indices_view) before they can be gathered.

```cpp
// gather Fortran-layout arrays
auto B_f = nda::array<int, 2, nda::F_layout>(2, 2);
B_f = comm.rank();
auto B_fg = mpi::gather(nda::transpose(B_f), comm, root);
print(B_fg, comm);
comm.barrier();
```

Output:

```
Rank 1:
[]
Rank 3:
[]
Rank 0:
[[0,0]
 [0,0]
 [1,1]
 [1,1]
 [2,2]
 [2,2]
 [3,3]
 [3,3]]
Rank 2:
[]
```

The resulting array `B_fg` has to be in C-order.
If we want it to be in Fortran-order, we can simply transpose it:

```cpp
// transpose the result
print(nda::transpose(B_fg), comm);
comm.barrier();
```

Output:

```
Rank 2:
[]
Rank 0:
[[0,0,1,1,2,2,3,3]
 [0,0,1,1,2,2,3,3]]
Rank 1:
[]
Rank 3:
[]
```

@subsection ex6_p3 Scattering an array/view

Scattering of an array/view is basically the inverse operation of gathering.
It takes an array/view and splits it along the first dimensions as evenly as possible among the processes.

For example, to scatter the same array that we just gathered, we can do

```cpp
// scatter an array from root to all other ranks
nda::array<int, 2> C = mpi::scatter(B_rg, comm, root);
print(C, comm);
comm.barrier();
```

Output:

```
Rank 2:
[[2,2]
 [2,2]]
Rank 1:
[[1,1]
 [1,1]]
Rank 0:
[[0,0]
 [0,0]]
Rank 3:
[[3,3]
 [3,3]]
```

Similar to the nda::mpi_gather, nda::mpi_scatter requires the input and output arrays to be in C-order.

If the extent along the first dimension is not divisible by the number of MPI ranks, processes with lower ranks will
receive more data than others:

```cpp
// scatter an array with extents not divisible by the number of ranks
auto C_s = mpi::scatter(C, comm, 2);
print(C_s, comm);
comm.barrier();
```

Output:

```
Rank 2:
[]
Rank 0:
[[2,2]]
Rank 3:
[]
Rank 1:
[[2,2]]
```

Here, a 2-by-2 array is scattered from rank 2 to all other processes.
It is split along the first dimension and the resulting 1-by-2 subarrays are sent to the ranks 0 and 1 while the ranks
2 and 3 do not receive any data.

@subsection ex6_p4 Reducing an array/view

Let us reduce the same 2-by-2 arrays from above.
Be default, `mpi::reduce` performs an element-wise summation among the ranks in the communicator and makes the result
available only on the root process:

```cpp
// reduce an array on root using MPI_SUM
auto D = mpi::reduce(C, comm, root);
print(D, comm);
comm.barrier();
```

Output:

```
Rank 2:
[]
Rank 3:
[]
Rank 1:
[]
Rank 0:
[[6,6]
 [6,6]]
```

To use a different reduction operation or to send the result to all ranks, we can do

```cpp
auto D = mpi::reduce(C, comm, root, /* all */ true, MPI_OP);
```

or the more compact

```cpp
auto D = mpi::all_reduce(C, comm, root, MPI_OP);
```

It is also possible to perform the reduction in-place, i.e. the result is written to the input array:

```cpp
// all-reduce an array in-place
mpi::all_reduce_in_place(C, comm);
print(C, comm);
comm.barrier();
```

Output:

```
Rank 2:
[[6,6]
 [6,6]]
Rank 0:
[[6,6]
 [6,6]]
Rank 3:
[[6,6]
 [6,6]]
Rank 1:
[[6,6]
 [6,6]]
```

In contrast to the standard `mpi::all_reduce` function, the in-place operation does not create and return a new array.
Instead the result is directly written into the input array.
