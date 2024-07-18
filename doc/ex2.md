@page ex2 Example 2: Constructing arrays

[TOC]

In this example, we show how to construct arrays in different ways.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <complex>
#include <iostream>

int main(int argc, char *argv[]) {
  // code snippets go here ...
}
```

@subsection ex2_p1 Default constructor

The default constructor creates an empty array of size 0:

```cpp
// default constructor
auto A1 = nda::array<double, 2>();
std::cout << "A1 = " << A1 << std::endl;
std::cout << "A1.size() = " << A1.size() << std::endl;
std::cout << "A1.shape() = " << A1.shape() << std::endl;
```

Output:

```
A =
[]
A.size() = 0
A.shape() = (0 0)
```

Since the size of the array is zero, no memory has been allocated.
To be able to use the array in a productive way, one has to resize it.
This can be done either
- directly via the free nda::resize_or_check_if_view function,
- the nda::basic_array::resize member or
- by assigning another array/view to it which indirectly will call nda::basic_array::resize:

```cpp
// resize the array using the resize method
A.resize(3, 2);
std::cout << "A.size() = " << A.size() << std::endl;
std::cout << "A.shape() = " << A.shape() << std::endl;

// resize using assignment
A = nda::array<double, 2>(10, 10);
std::cout << "A.size() = " << A.size() << std::endl;
std::cout << "A.shape() = " << A.shape() << std::endl;
```

Output:

```
A.size() = 6
A.shape() = (3 2)
A.size() = 100
A.shape() = (10 10)
```

@subsection ex2_p2 Constructing an array with a given shape

The usual way to create an array is by specifying its shape.
While the shape is a runtime parameter, the rank of the array still has to be known at compile-time (it is a template
parameter):

```cpp
// create a 100x100 complex matrix
auto M1 = nda::matrix<std::complex<double>>(100, 100);
std::cout << "M1.size() = " << M1.size() << std::endl;
std::cout << "M1.shape() = " << M1.shape() << std::endl;
```

Output:

```
M1.size() = 10000
M1.shape() = (100 100)
```

While for higher dimensional arrays the elements are in general left uninitialized, it is possible to construct an
1-dimensional array with a given size and initialize its elements to a constant value:

```cpp
// create a 1-dimensional vector of size 5 with the value 10 + 1i
using namespace std::complex_literals;
auto v1 = nda::vector<std::complex<double>>(5, 10 + 1i);
std::cout << "v1 = " << v1 << std::endl;
std::cout << "v1.size() = " << v1.size() << std::endl;
std::cout << "v1.shape() = " << v1.shape() << std::endl;
```

Output:

```
v1 = [(10,1),(10,1),(10,1),(10,1),(10,1)]
v1.size() = 5
v1.shape() = (5)
```

@subsection ex2_p3 Copy/Move constructors

The copy and move constructors behave as expected:

- the copy constructor simply copies the memory handle and layout of an array and
- the move constructor moves them:

```cpp
// copy constructor
auto v2 = v1;
std::cout << "v2 = " << v2 << std::endl;
std::cout << "v2.size() = " << v2.size() << std::endl;
std::cout << "v2.shape() = " << v2.shape() << std::endl;

// move constructor
auto v3 = std::move(v2);
std::cout << "v3 = " << v3 << std::endl;
std::cout << "v3.size() = " << v3.size() << std::endl;
std::cout << "v3.shape() = " << v3.shape() << std::endl;
std::cout << "v2.empty() = " << v2.empty() << std::endl;
```

Output:

```
v2 = [(10,1),(10,1),(10,1),(10,1),(10,1)]
v2.size() = 5
v2.shape() = (5)
v3 = [(10,1),(10,1),(10,1),(10,1),(10,1)]
v3.size() = 5
v3.shape() = (5)
v2.empty() = 1
```

> **Note**: After moving, the vector `v2` is empty, i.e. its memory handle does not manage any memory at the moment. To
> use it, one should again resize it, so that new memory is allocated.

@subsection ex2_p4 Constructing an array from its data

1-, 2- and 3-dimensional arrays can be constructed directly from their data using `std::initializer_list` objects:
- a 1-dimensional array is constructed from a single list,
- a 2-dimensional array is constructed from a nested list (a list of lists) and
- a 3-dimensional array is constructed from a double nested list (a list of lists of lists).

```cpp
// 1-dimensional array from a std::initializer_list
auto A1 = nda::array<int, 1>{1, 2, 3, 4, 5};
std::cout << "A1 = " << A1 << std::endl;
std::cout << "A1.size() = " << A1.size() << std::endl;
std::cout << "A1.shape() = " << A1.shape() << std::endl;

// 2-dimensional array from a std::initializer_list
auto A2 = nda::array<int, 2>{{1, 2}, {3, 4}, {5, 6}};
std::cout << "A2 = " << A2 << std::endl;
std::cout << "A2.size() = " << A2.size() << std::endl;
std::cout << "A2.shape() = " << A2.shape() << std::endl;

// 3-dimensional array from a std::initializer_list
auto A3 = nda::array<int, 3>{{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}};
std::cout << "A3 = " << A3 << std::endl;
std::cout << "A3.size() = " << A3.size() << std::endl;
std::cout << "A3.shape() = " << A3.shape() << std::endl;
```

Output:

```
A1 = [1,2,3,4,5]
A1.size() = 5
A1.shape() = (5)
A2 =
[[1,2]
 [3,4]
 [5,6]]
A2.size() = 6
A2.shape() = (3 2)
A3 = [1,2,3,4,5,6,7,8,9,10,11,12]
A3.size() = 12
A3.shape() = (2 3 2)
```

@subsection ex2_p5 Constructing an array from an nda::Array

We can also construct an array from any object that satisfies the nda::Array concept, has a compatible value type and
the same rank.

For example, this could be a lazy expression or another array/view with a possibly different
- value type,
- memory layout (i.e. stride order),
- algebra and/or
- container policy (e.g. construct an array on the GPU from an array on the host).

```cpp
// construct an array from a lazy expression
nda::array<int, 1> A1_sum = A1 + A1;
std::cout << "A1_sum = " << A1_sum << std::endl;
std::cout << "A1_sum.size() = " << A1_sum.size() << std::endl;
std::cout << "A1_sum.shape() = " << A1_sum.shape() << std::endl;

// construct an array from another array with a different memory layout
nda::array<double, 2, nda::F_layout> A2_f(A2);
std::cout << "A2_f = " << A2_f << std::endl;
std::cout << "A2_f.size() = " << A2_f.size() << std::endl;
std::cout << "A2_f.shape() = " << A2_f.shape() << std::endl;
```

Output:

```
A1_sum = [2,4,6,8,10]
A1_sum.size() = 5
A1_sum.shape() = (5)
A2_f =
[[1,2]
 [3,4]
 [5,6]]
A2_f.size() = 6
A2_f.shape() = (3 2)
```

@subsection ex2_p6 Factories and transformations

**nda** provides various factory functions and transformations that allow us to construct special arrays very easily
either from scratch or from some other input arrays.

Here, we only give a few examples and refer the interested reader to the @ref av_factories documentation.

```cpp
// 1-dimensional array with only even integers from 2 to 20
auto v4 = nda::arange(2, 20, 2);
std::cout << "v4 = " << v4 << std::endl;

// 3x3 identity matrix
auto I = nda::eye<double>(3);
std::cout << "I = " << I << std::endl;

// 2x2x2 array with random numbers from 0 to 1
auto R = nda::rand(2, 2, 2);
std::cout << "R = " << R << std::endl;
std::cout << "R.shape() = " << R.shape() << std::endl;

// 3x6 array with zeros
auto Z = nda::zeros<double>(3, 6);
std::cout << "Z = " << Z << std::endl;
```

Output:

```
v4 = [2,4,6,8,10,12,14,16,18]
I =
[[1,0,0]
 [0,1,0]
 [0,0,1]]
R = [0.349744,0.226507,0.444232,0.229969,0.370864,0.440588,0.607665,0.754914]
R.shape() = (2 2 2)
Z =
[[0,0,0,0,0,0]
 [0,0,0,0,0,0]
 [0,0,0,0,0,0]]
```
