@page ex3 Example 3: Initializing arrays

[TOC]

In this example, we show how to initialize arrays and assign to them in different ways.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <complex>
#include <iostream>

int main(int argc, char *argv[]) {
  // code snippets go here ...
}
```

@subsection ex3_p1 Assigning a scalar to an array

The simplest way to initialize an array is to assign a scalar to it:

```cpp
// assign a scalar to an array
auto A = nda::array<std::complex<double>, 2>(3, 2);
A = 0.1 + 0.2i;
std::cout << "A = " << A << std::endl;
```

Output:

```
A =
[[(0.1,0.2),(0.1,0.2)]
 [(0.1,0.2),(0.1,0.2)]
 [(0.1,0.2),(0.1,0.2)]]
```

Note that the behavior of the assignment depends on the algebra of the array.

While the scalar is assigned to all elements of an array, for matrices it is only assigned to the elements on the
shorter diagonal:

```cpp
// assign a scalar to a matrix
auto M = nda::matrix<std::complex<double>>(3, 2);
M = 0.1 + 0.2i;
std::cout << "M = " << M << std::endl;
```

Output:

```
M =
[[(0.1,0.2),(0,0)]
 [(0,0),(0.1,0.2)]
 [(0,0),(0,0)]]
```

The scalar type can also be more complex, e.g. another nda::array:

```cpp
// assign an array to an array
auto A_arr = nda::array<nda::array<int, 1>, 1>(4);
A_arr = nda::array<int, 1>{1, 2, 3};
std::cout << "A_arr = " << A_arr << std::endl;
```

Output:

```
A_arr = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
```

@subsection ex3_p2 Copy/Move assignment

The copy and move assignment operations behave as expected:

- copy assignment simply copies the memory handle and layout of an array and
- move assignment moves them:

```cpp
// copy assignment
auto M_copy = nda::matrix<std::complex<double>>(3, 2);
M_copy = M;
std::cout << "M_copy = " << M_copy << std::endl;

// move assignment
auto M_copy = nda::matrix<std::complex<double>>();
M_move = std::move(M_copy);
std::cout << "M_move = " << M_move << std::endl;
std::cout << "M_copy.empty() = " << M_copy.empty() << std::endl;
```

Output:

```
M_copy =
[[(0.1,0.2),(0,0)]
 [(0,0),(0.1,0.2)]
 [(0,0),(0,0)]]
M_move =
[[(0.1,0.2),(0,0)]
 [(0,0),(0.1,0.2)]
 [(0,0),(0,0)]]
M_copy.empty() = 1
```

> **Note**: Be careful when reusing an object after it has been moved (see the note at @ref ex2_p3)!

@subsection ex3_p3 Assigning an nda::Array to an array

To assign an object that satisfies the nda::Array concept to an array is similar to @ref ex2_p5.

```cpp
// assign a lazy expression
nda::matrix<std::complex<double>, nda::F_layout> M_f(3, 2);
M_f = M + M;
std::cout << "M_f = " << M_f << std::endl;

// assign another array with a different layout and algebra
nda::array<std::complex<double>, 2> A2;
A2 = M_f;
std::cout << "A2 = " << A2 << std::endl;
```

Output:

```
M_f =
[[(0.2,0.4),(0,0)]
 [(0,0),(0.2,0.4)]
 [(0,0),(0,0)]]
A2 =
[[(0.2,0.4),(0,0)]
 [(0,0),(0.2,0.4)]
 [(0,0),(0,0)]]
```

Many assignment operations resize the left hand side array if necessary.

In the above example, the first assignment does not need to do a resize because the left hand side array already has the
correct shape.

This is not true for the second assignment.
`A2` has been default constructed and therefore has a size of 0.

@subsection ex3_p4 Assigning a contiguous range

It is possible to assign an object that satisfies the `std::ranges::contiguous_range` concept to an 1-dimensional array:

```cpp
// assign a contiguous range to an 1-dimensional array
std::vector<long> vec{1, 2, 3, 4, 5};
auto A_vec = nda::array<long, 1>();
A_vec = vec;
std::cout << "A_vec = " << A_vec << std::endl;
```

Output:

```
A_vec = [1,2,3,4,5]
```

As expected, the elements of the range are simply copied into the array.

@subsection ex3_p5 Initializing an array manually

We can also initialize an array by assigning to each element manually.
This can be done in different ways.

For example,

- using traditional for-loops:

  ```cpp
  // initialize an array using traditional for-loops
  auto B = nda::array<int, 2>(2, 3);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      B(i, j) = i * 3 + j;
    }
  }
  std::cout << "B = " << B << std::endl;
  ```

  Output:

  ```
  B =
  [[0,1,2]
   [3,4,5]]
  ```

- using a range-based for-loop:

  ```cpp
  // initialize an array using a range-based for-loop
  auto B2 = nda::array<int, 2>(2, 3);
  for (int i = 0; auto &x : B2) x = i++;
  std::cout << "B2 = " << B2 << std::endl;
  ```

  Output:

  ```
  B2 =
  [[0,1,2]
   [3,4,5]]
  ```

- using nda::for_each:

  ```cpp
  // initialize an array using nda::for_each
  auto B3 = nda::array<int, 2>(2, 3);
  nda::for_each(B3.shape(), [&B3](auto i, auto j) { B3(i, j) = i * 3 + j; });
  std::cout << "B3 = " << B3 << std::endl;
  ```

  Output:

  ```
  B3 =
  [[0,1,2]
   [3,4,5]]
  ```

While the traditional for-loops are perhaps the most flexible option, it becomes tedious quite fast with increasing
dimensionality.

@subsection ex3_p6 Initializing an array using automatic assignment

This has already been explained in @ref ex1_p12.

Let us give another example:

```cpp
// initialize an array using CLEF's automatic assignment
using namespace nda::clef::literals;
auto C = nda::array<int, 4>(2, 2, 2, 2);
C(i_, j_, k_, l_) << i_ * 8 + j_ * 4 + k_ * 2 + l_;
std::cout << "C = " << C << std::endl;
```

Output:

```
C = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
```

The advantage of CLEF's automatic assignment becomes obvious when we rewrite line 3 from above using traditional
for-loops:

```cpp
for (int i = 0; i < 2; ++i) {
  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < 2; ++k) {
      for (int l = 0; l < 2; ++l) {
        C(i, j, k, l) = i * 8 + j * 4 + k * 2 + l;
      }
    }
  }
}
```
