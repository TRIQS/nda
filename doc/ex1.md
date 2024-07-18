@page ex1 Example 1: A quick overview

[TOC]

In this example, we give a quick overview of the basic capabilities of **nda**.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <h5/h5.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  // code snippets go here ...
}
```

@subsection ex1_p1 Creating and initializing an array

In its simplest form, an nda::array has 2 template parameters

- its value type and
- its rank or number of dimensions.

The following creates an integer (`int`) array of rank 2 and initializes its elements:

```cpp
// create a 3x2 integer array and initialize it
auto A = nda::array<int, 2>(3, 2);
for (int i = 0; auto &x : A) x = i++;
```

The first line calls the constructor with the intended shape of the array, i.e. the extents along the two dimensions.
In this case, we want it to be 3-by-2.

In the second line, we use a range-based for loop (note the reference `auto &`) to assign the first few positive integer
numbers to the elements of the array.

We could have achieved the same using the constructor which takes `std::initializer_list` objects:

```cpp
// create a 3x2 integer array using initializer lists
auto A2 = nda::array<int, 2>{{0, 1}, {2, 3}, {4, 5}};
```

@subsection ex1_p2 Choosing a memory layout

By default, nda::array stores its elements in C-order. To create an array in Fortran-order, we can specify a third
template parameter:

```cpp
// create a 3x2 integer array in Fortran order and initialize it
auto B = nda::array<int, 2, nda::F_layout>(3, 2);
for (int i = 0; auto &x : B) x = i++;
```

Here, nda::F_layout is one of the @ref layout_pols.

While in 2-dimensions, the only possibilities are C-order or Fortran-order, in higher dimensions one can also specify
other stride orders (see nda::basic_layout and nda::basic_layout_str).

@subsection ex1_p3 Printing an array

Let's check the contents, sizes and shapes of the arrays using the overloaded streaming operators:

```cpp
// print the formatted arrays, its sizes and its shapes
std::cout << "A = " << A << std::endl;
std::cout << "A.size() = " << A.size() << std::endl;
std::cout << "A.shape() = " << A.shape() << std::endl;
std::cout << std::endl;
std::cout << "B = " << B << std::endl;
std::cout << "B.size() = " << B.size() << std::endl;
std::cout << "B.shape() = " << B.shape() << std::endl;
```

Output:

```
A =
[[0,1]
 [2,3]
 [4,5]]
A.size() = 6
A.shape() = (3 2)
B =
[[0,3]
 [1,4]
 [2,5]]
B.size() = 6
B.shape() = (3 2)
```
You can see the difference between the memory layouts of the array:

- nda::C_layout iterates first over the second dimension while
- nda::F_layout iterates first over the first dimension.

> **Note**: The reason why we see a difference in the two arrays is because the range-based for loop is optimized to
> iterate over contiguous memory whenever possible. If we would have used traditional for loops instead,
> ```cpp
> int n = 0;
> for (int i = 0; i < 3; ++i) {
>   for (int j = 0; j < 2; ++j) {
>     B(i, j) = n++;
>   }
> }
> ```
> there would be no difference between the output of `A` and `B`:
> ```
> B =
> [[0,1]
>  [2,3]
>  [4,5]]
> ```

@subsection ex1_p4 Accessing single elements

We can access single elements of the array using the function call operator of the array object.
For a 2-dimensional array, we have to pass exactly two indices, otherwise it won't compile:

```cpp
// access and assign to a single element
A(2, 1) = 100;
B(2, 1) = A(2, 1);
std::cout << "A = " << A << std::endl;
std::cout << "B = " << B << std::endl;
std::cout << "B(2, 1) = " << B(2, 1) << std::endl;
```

Output:

```
A =
[[0,1]
 [2,3]
 [4,100]]
B =
[[0,3]
 [1,4]
 [2,100]]
B(2, 1) = 100
```

> **Note**: Out-of-bounds checks are only enabled in debug mode. The following code will compile and result in undefined
> behavior:
> ```cpp
> std::cout << "A(3, 2) = " << A(3, 2) << std::endl;
> ```
> Output (might be different on other machines):
> ```
> A(3, 2) = 9
> ```

@subsection ex1_p5 Assigning to an array

It is straightforward to assign a scalar or another array to an existing array:

```cpp
// assigning a scalar to an array
A = 42;
std::cout << "A = " << A << std::endl;

// assigning an array to another array
A = B;
std::cout << "A = " << A << std::endl;
```

Output:

```
A =
[[42,42]
 [42,42]
 [42,42]]
A =
[[0,3]
 [1,4]
 [2,100]]
```

@subsection ex1_p6 Working with views

Views offer a lightweight and efficient way to manipulate and operate on existing arrays since they do not own their
data, i.e. there is no memory allocation or copying involved when creating a view (see nda::basic_array_view).

Taking a view on an existing array can be done with an empty function call:

```cpp
// create a view on A
auto A_v = A();
std::cout << "A_v =" << A_v << std::endl;
std::cout << "A_v.size() = " << A_v.size() << std::endl;
std::cout << "A_v.shape() = " << A_v.shape() << std::endl;
```

Output:

```
A_v =
[[0,3]
 [1,4]
 [2,100]]
A_v.size() = 6
A_v.shape() = (3 2)
```

Here, `A_v` has the same data, size and shape as `A`.
Since `A_v` is a view, it points to the same memory location as `A`.
That means if we change something in `A_v`, we will also see the changes in `A`:

```cpp
// manipulate the data in the view
A_v(0, 1) = -12;
std::cout << "A = " << A << std::endl;
```

Output:

```
A =
[[0,-12]
 [1,4]
 [2,100]]
```

In most cases, views will just behave like arrays and the majority of functions and operations that can be performed
with arrays, also work with views.

@subsection ex1_p7 Working with slices

A slice is a view on only some parts of an existing array.

To take a slice of an array, one uses again the function call operator but this time with one or more `nda::range`,
`nda::range::all` or nda::ellipsis objects:

- `nda::range` is similar to a Python [range](https://docs.python.org/3/library/stdtypes.html#range) and specifies which
elements should be included in the slice along a certain dimension.
- `nda::range::all` is similar to Python's `:` syntax and specifies that all elements along a certain dimension should
be included in the slice.
- nda::ellipsis is similar to Python's `...` syntax and specifies that all "missing" dimensions should be included in
the slice. One can achieve the same with multiple `nda::range::all` objects.

For example, we can view only the second row of `A`:

```cpp
// create a slice on A (a view on its second row)
auto A_s = A(1, nda::range::all);
std::cout << "A_s = " << A_s << std::endl;
std::cout << "A_s.size() = " << A_s.size() << std::endl;
std::cout << "A_s.shape() = " << A_s.shape() << std::endl;
```

Output:

```
A_s = [1,4]
A_s.size() = 2
A_s.shape() = (2)
```

As seen in this example, slices usually have a different size and shape compared to the original array.
In this case, it is a 1-dimensional view of size 2.

Since a slice is just a view, everything mentioned above about views still applies:

```cpp
// assign to a slice
A_s = nda::array<int, 1>{-1, -4};
std::cout << "A_s = " << A_s << std::endl;
std::cout << "A = " << A << std::endl;
```

Output:

```
A_s = [-1,-4]
A =
[[0,-12]
 [-1,-4]
 [2,100]]
```

@subsection ex1_p8 Performing arithmetic operations

We can perform various @ref av_ops with arrays and views.
Arithmetic operations are (mostly) implemented as lazy expressions. That means the operations are not performed right
away but only when needed at a later time.

For example, the following tries to add two 2-dimensional arrays element-wise:

```cpp
// add two arrays element-wise
auto C = nda::array<int, 2>({{1, 2}, {3, 4}});
auto D = nda::array<int, 2>({{5, 6}, {7, 8}});
std::cout << C + D << std::endl;
```

Output:

```
(
[[1,2]
 [3,4]] +
[[5,6]
 [7,8]])
```

Note that `C + D` does not return the result of the addition in the form of a new array object, but instead it returns
an nda::expr object that describes the operation it is about to do.

To turn a lazy expression into an array/view, we can either assign it to an existing array/view, construct a new array
or use nda::make_regular:

```cpp
// evaluate the lazy expression by assigning it to an existing array
auto E = nda::array<int, 2>(2, 2);
E = C + D;
std::cout << "E = " << E << std::endl;

// evaluate the lazy expression by constructing a new array
std::cout << "nda::array<int, 2>{C + D} = " << nda::array<int, 2>{C + D} << std::endl;

// evaluate the lazy expression using nda::make_regular
std::cout << "nda::make_regular(C + D) = " << nda::make_regular(C + D) << std::endl;
```

Output:

```
E =
[[6,8]
 [10,12]]
nda::array<int, 2>{C + D} =
[[6,8]
 [10,12]]
nda::make_regular(C + D) =
[[6,8]
 [10,12]]
```

The behavior of arithmetic operations (and some other functions) depends on the algebra (see nda::get_algebra) of its
operands.

For example, consider matrix-matrix multiplication (``'M'`` algebra) vs. element-wise multiplication of two arrays
(``'A'`` algebra):

```cpp
// multiply two arrays elementwise
std::cout << "C * D =" << nda::array<int, 2>(C * D) << std::endl;

// multiply two matrices
auto M1 = nda::matrix<int>({{1, 2}, {3, 4}});
auto M2 = nda::matrix<int>({{5, 6}, {7, 8}});
std::cout << "M1 * M2 =" << nda::matrix<int>(M1 * M2) << std::endl;
```

Output:

```
C * D =
[[5,12]
 [21,32]]
M1 * M2 =
[[19,22]
 [43,50]]
```

Here, an nda::matrix is the same as an nda::array of rank 2, except that it belongs to the ``'M'`` algebra instead of
the ``'A'`` algebra.

@subsection ex1_p9 Applying mathematical functions and algorithms

Similar to arithmetic operations, most of the @ref av_math that can operate on arrays and views return a lazy function
call expression (nda::expr_call) and are sensitive to their algebras:

```cpp
// element-wise square
nda::array<double, 2> C_sq = nda::pow(C, 2);
std::cout << "C^2 =" << C_sq << std::endl;

// element-wise square root
nda::array<double, 2> C_sq_sqrt = nda::sqrt(C_sq);
std::cout << "sqrt(C^2) =" << C_sq_sqrt << std::endl;

// trace of a matrix
std::cout << "trace(M1) = " << nda::trace(M1) << std::endl;
```

Output:

```
C^2 =
[[1,4]
 [9,16]]
sqrt(C^2) =
[[1,2]
 [3,4]]
trace(M1) = 5
```

Besides arithmetic operations and mathematical functions, one can apply some basic @ref av_algs to arrays and views:

```cpp
// find the minimum/maximum element in an array
std::cout << "min_element(C) =" << nda::min_element(C) << std::endl;
std::cout << "max_element(C) =" << nda::max_element(C) << std::endl;

// is any (are all) element(s) greater than 1?
auto greater1 = nda::map([](int x) { return x > 1; })(C);
std::cout << "any(C > 1) = " << nda::any(greater1) << std::endl;
std::cout << "all(C > 1) = " << nda::all(greater1) << std::endl;

// sum/multiply all elements in an array
std::cout << "sum(C) = " << nda::sum(C) << std::endl;
std::cout << "product(C) = " << nda::product(C) << std::endl;
```

Output:

```
min_element(C) = 1
max_element(C) = 4
any(C > 1) = 1
all(C > 1) = 0
sum(C) = 10
product(C) = 24
```

Most of the algorithms in the above example are self-explanatory, except for the nda::any and nda::all calls.
They expect the given array elements to have a `bool` value type which is not the case for the `C` array.
We therefore use nda::map to create a lazy nda::expr_call that returns a `bool` upon calling it.
Since nda::expr_call fulfills the nda::Array concept, it can be passed to the algorithms.

@subsection ex1_p10 Writing/Reading HDF5

Writing arrays and views to HDF5 files using **nda's** @ref av_hdf5 is as easy as

```cpp
// write an array to an HDF5 file
h5::file out_file("ex1.h5", 'w');
h5::write(out_file, "C", C);
```

Dumping the resulting HDF5 file gives

```
HDF5 "ex1.h5" {
GROUP "/" {
   DATASET "C" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 2, 2 ) / ( 2, 2 ) }
      DATA {
      (0,0): 1, 2,
      (1,0): 3, 4
      }
   }
}
}
```

Reading the same file into a new array is just as simple:

```cpp
// read an array from an HDF5 file
nda::array<int, 2> C_copy;
h5::file in_file("ex1.h5", 'r');
h5::read(in_file, "C", C_copy);
std::cout << "C_copy = " << C_copy << std::endl;
```

Output:

```
C_copy =
[[1,2]
 [3,4]]
```

@subsection ex1_p11 Doing linear algebra with arrays

**nda** has a @ref linalg_blas and an @ref linalg_lapack and provides the nda::matrix and nda::vector types.
While nda::matrix is a 2-dimensional array belonging to the ``'M'`` algebra, nda::vector is a 1-dimensional array
belonging to the ``'V'`` algebra.
Under the hood, both of them are just type aliases of the general nda::basic_array class.
Everything mentioned above for nda::array objects is therefore also true for matrices and vectors, e.g. writing/reading
to HDF5, accessing single elements, taking slices, etc.

Let us demonstrate some of the supported @ref linalg features:

```cpp
// create a 2x2 matrix
auto M3 = nda::matrix<double>{{1, 2}, {3, 4}};

// get the inverse of the matrix (calls LAPACK routines)
auto M3_inv = nda::inverse(M3);
std::cout << "M3_inv = " << M3_inv << std::endl;

// get the inverse of a matrix manually using its adjugate and determinant
auto M3_adj = nda::matrix<double>{{4, -2}, {-3, 1}};
auto M3_det = nda::determinant(M3);
auto M3_inv2 = nda::matrix<double>(M3_adj / M3_det);
std::cout << "M3_inv2 = " << M3_inv2 << std::endl;

// check the inverse using matrix-matrix multiplication
std::cout << "M3 * M3_inv = " << M3 * M3_inv << std::endl;
std::cout << "M3_inv * M3 = " << M3_inv * M3 << std::endl;

// solve a linear system using the inverse
auto b = nda::vector<double>{5, 8};
auto x = M3_inv * b;
std::cout << "M3 * x = " << M3 * x << std::endl;
```

Output:

```
M3_inv =
[[-2,1]
 [1.5,-0.5]]
M3_inv2 =
[[-2,1]
 [1.5,-0.5]]
M3 * M3_inv =
[[1,0]
 [0,1]]
M3_inv * M3 =
[[1,0]
 [0,1]]
M3 * x = [5,8]
```

> **Note**: Matrix-matrix and matrix-vector multiplication do not return lazy expressions, since they call the
> corresponding BLAS routines directly, while element-wise array-array multiplication does return a lazy expression.

@subsection ex1_p12 Initializing with CLEF's automatic assignment

**nda** contains the @ref clef (CLEF) library which is a more or less standalone implementation of general lazy
expressions.

One of the most useful tools of CLEF is its @ref clef_autoassign feature:

```cpp
// initialize a 4x7 array using CLEF automatic assignment
using namespace nda::clef::literals;
auto F = nda::array<int, 2>(4, 7);
F(i_, j_) << i_ * 7 + j_;
std::cout << "F = " << F << std::endl;
```

Output:

```
F =
[[0,1,2,3,4,5,6]
 [7,8,9,10,11,12,13]
 [14,15,16,17,18,19,20]
 [21,22,23,24,25,26,27]]
```

Here, we first construct an uninitialized 4-by-7 array before we use automatic assignment to initialize the array.
This is done with the overloaded `operator<<` and nda::clef::placeholder objects (`i_` and `j_` in the above example).
It will simply loop over all possible multidimensional indices of the array, substitute them for the placeholders and
assign the result to the corresponding array element, e.g. `F(3, 4) = 3 * 7 + 4 = 25`.

This is especially helpful for high-dimensional arrays where the element at `(i, j, ..., k)` can be written as some
function \f$ g \f$ of its indices, i.e. \f$ F_{ij \dots k} = g(i, j, \dots, k) \f$.

@subsection ex1_p13 Further examples

The above features constitute only a fraction of what you can do with **nda**.

For more details, please look at the other @ref examples and the @ref documentation.
