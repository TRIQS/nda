[![build](https://github.com/TRIQS/nda/workflows/build/badge.svg?branch=unstable)](https://github.com/TRIQS/nda/actions?query=workflow%3Abuild)

# nda

nda is a C++ library providing an efficient and flexible multi-dimensional array class.
It is an essential building-block of the TRIQS project. Some features include
* coded in C++17/20 using concepts
* expressions are implemented lazily for maximum performance
* flexible and lightweight view-types
* matrix and vector class with BLAS / LAPACK backend
* easily store and retrieve arrays to and from hdf5 files using [h5](https://github.com/TRIQS/h5)
* common mpi functionality using [mpi](https://github.com/TRIQS/mpi)


## Simple Example

```c++
#include <nda/nda.hpp>
#include <nda/h5.hpp>

using namespace nda;

int main() {

  // Create array of shape (4,4,4)
  array<long, 3> A(4, 4, 4);

  // Create an array given its data
  array<long, 2> B{{1, 2}, {3, 4}, {5, 6}};

  // Assign
  A() = 0;
  A(0, 1, 2) = 40;

  // Access
  int a = A(0, 1, 2) + B(0, 1);

  // Slicing to a view of shape (3, 2)
  auto V = A(range(0, 3), range(0, 2), 0);

  // Lazy Arithmetic operations
  auto C = V + 2 * B;

  // Algorithms
  min_element(V);
  max_element(V);
  sum(V);

  // write to file
  {
    h5::file file("dat.h5", 'w');
    h5_write(file, "A", A);
  }

  // read from file
  array<long, 3> D;
  {
    h5::file file("dat.h5", 'r');
    h5_read(file, "A", D);
  }

}
```

For further examples we refer the users to our [tests](https://github.com/TRIQS/nda/tree/unstable/test/c++).
