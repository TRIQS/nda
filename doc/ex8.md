@page ex8 Example 8: Linear algebra support

[TOC]

In this example, we show how do linear algebra with **nda** arrays.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <complex>
#include <iostream>

int main(int argc, char *argv[]) {
  // code snippets go here ...
}
```

Before showing some linear algebra related operations, let us first introduce the nda::matrix and nda::vector types and
highlight their similarities and differences with respect to nda::array.

@subsection ex8_p1 Matrix and vector types

As already mentioned in the quick introduction @ref ex1_p11, **nda** provides an nda::matrix and an nda::vector type.
Both are specializations of the general nda::basic_array with the following features:
- nda::matrix is a 2-dimensional array that belongs to the ``'M'`` algebra.
- nda::vector is a 1-dimensional array that belongs to the ``'V'`` algebra.

Their algebras determine how matrices and vectors behave in certain operations and expressions.

Otherwise, everything that is true for nda::basic_array objects is also true for nda::matrix and nda::vector objects.

@subsection ex8_p2 Constructing matrices and vectors

To construct a matrix or a vector, we can use the same methods as for nda::array objects.
We refer the user to @ref ex2 for more information.

For example, to construct a matrix and a vector of a certain shape, we can do

```cpp
// construct a 4x3 matrix and a vector of size 3
nda::matrix<double> M(4, 3);
nda::vector<double> v(3);
```

Here, a \f$ 4 \times 3 \f$ matrix and a vector of size \f$ 3 \f$ are created.

Let us note that there are some @ref av_factories that are specific to matrices:

```cpp
// construct a 3x3 identity matrix
auto I = nda::eye<double>(3);
std::cout << "I = " << I << std::endl;

// construct a 2x2 diagonal matrix
auto D = nda::diag(nda::array<int, 1>{1, 2});
std::cout << "D = " << D << std::endl;
```

Output:

```
I =
[[1,0,0]
 [0,1,0]
 [0,0,1]]
D =
[[1,0]
 [0,2]]
```

nda::eye constructs an identity matrix of a certain size and nda::diag takes a 1-dimensional array and constructs a
square diagonal matrix containing the values of the given array.

@subsection ex8_p3 Initializing and assigning to matrices and vectors

Again, initializing and assigning to matrices and vectors works (almost) exactly in the same way as it does for arrays
(see @ref ex3)

The main difference occurs, when we are assigning a scalar. While for arrays and vectors the assignment is done
element-wise, for matrices the scalar is assigned only to the elements on the shorter diagonal and the rest is zeroed
out:

```cpp
// assigning a scalar to a matrix and vector
M = 3.1415;
v = 2.7182;
std::cout << "M = " << M << std::endl;
std::cout << "v = " << v << std::endl;
```

Output:

```
M =
[[3.1415,0,0]
 [0,3.1415,0]
 [0,0,3.1415]
 [0,0,0]]
v = [2.7182,2.7182,2.7182]
```

@subsection ex8_p4 Views on matrices and vectors

There are some @ref av_factories for views that are specific to matrices and vectors, otherwise everything mentioned in
@ref ex4 still applies:

```cpp
// get a vector view of the diagonal of a matrix
auto d = nda::diagonal(D);
std::cout << "d = " << d << std::endl;
std::cout << "Algebra of d: " << nda::get_algebra<decltype(d)> << std::endl;

// transform an array into a matrix view
auto A = nda::array<int, 2>{{1, 2}, {3, 4}};
auto A_mv = nda::make_matrix_view(A);
std::cout << "A_mv = " << A_mv << std::endl;
std::cout << "Algebra of A: " << nda::get_algebra<decltype(A)> << std::endl;
std::cout << "Algebra of A_mv: " << nda::get_algebra<decltype(A_mv)> << std::endl;
```

Output:

```
d = [1,2]
Algebra of d: V
A_mv =
[[1,2]
 [3,4]]
Algebra of A: A
Algebra of A_mv: M
```

@subsection ex8_p5 HDF5, MPI and symmetry support for matrices and vectors

We refer the user to the examples
- @ref ex5,
- @ref ex6 and
- @ref ex7.

There are no mentionable differences between arrays, matrices and vectors regarding those features.

@subsection ex8_p6 Arithmetic operations with matrices and vectors

Here the algebra of the involved types becomes important.
In section @ref ex1_p8, we have shortly introduced how arithmetic operations are implemented in **nda** in terms of lazy
expressions and how the algebra of the operands influences the results.

Let us be more complete in this section.
Suppose we have the following objects:
- nda::Array (any type that satisfies this concept): `O1`, `O2`, ...
- nda::array: `A1`, `A2`, ...
- nda::matrix: `M1`, `M2`, ...
- nda::vector: `v1`, `v2`, ...
- scalar: `s1`, `s2`, ...

Then the following operations are allowed  (all operations are lazy unless mentioned otherwise)

- **Addition** / **Subtraction**
  - `O1 +/- O2`: element-wise addition/subtraction, shapes of `O1` and `O2` have to be the same, result has the same
    shape
  - `s1 +/- A1`/`A1 +/- s1`: element-wise addition/subtraction, result has the same shape as `A1`
  - `s1 +/- v1`/`v1 +/- s1`: element-wise addition/subtraction, result has the same shape as `v1`
  - `s1 +/- M1`/`M1 +/- s1`: scalar is added/subtracted to/from the elements on the shorter diagonal of `M1`, result has
    the same shape as `M1`

- **Multiplication**
  - `A1 * A2`: element-wise multiplication, shapes of `A1` and `A2` have to be the same, result has the same shape
  - `M1 * M2`: non-lazy matrix-matrix multiplication, shapes have the expected requirements for matrix multiplication,
    calls nda::matmul
  - `M1 * v1`: non-lazy matrix-vector multiplication, shapes have the expected requirements for matrix multiplication,
    calls nda::matvecmul
  - `s1 * O1`/`O1 * s1`: element-wise multiplication, result has the same shape as `O1`

- **Division**
  - `A1 / A2`: element-wise division, shapes of `A1` and `A2` have to be the same, result has the same shape
  - `M1 / M2`: non-lazy matrix-matrix multiplication of `M1` by the inverse of `M2`, shapes have the expected
    requirements for matrix multiplication with the additional requirement that `M2` is square (since `M2` is inverted),
    result is also square with the same size
  - `O1 / s1`: element-wise division, result has the same shape as `O1`
  - `s1 / A1`: element-wise division, result has the same shape as `A1` (note this is `!= A1 / s1`)
  - `s1 / v1`: element-wise division, result has the same shape as `v1` (note this is `!= v1 / s1`)
  - `s1 / M1`: multiplies (lazy) the scalar with the inverse (non-lazy) of `M1`, only square matrices are allowed (since
    `M1` is inverted), result is also square with the same size

@subsection ex8_p7 Using linear algebra tools

In addition to the basic matrix-matrix and matrix-vector multiplications described above, **nda** provides some useful
@ref linalg_tools.
While some of them have a custom implementation, e.g. nda::linalg::cross_product, most make use of the @ref linalg_blas
and @ref linalg_lapack to call more optimized routines.

Let us demonstrate some of its features:

```cpp
// take the cross product of two vectors
auto v1 = nda::vector<double>{1.0, 2.0, 3.0};
auto v2 = nda::vector<double>{4.0, 5.0, 6.0};
auto v3 = nda::linalg::cross_product(v1, v2);
std::cout << "v1 = " << v1 << std::endl;
std::cout << "v2 = " << v2 << std::endl;
std::cout << "v3 = v1 x v2 = " << v3 << std::endl;
```

Output:

```
v1 = [1,2,3]
v2 = [4,5,6]
v3 = v1 x v2 = [-3,6,-3]
```

Here, we first create two 3-dimensional vectors and then take their cross product.

To check that the cross product is perpendicular to `v1` and `v2`, we can use the dot product:

```cpp
// check the cross product using the dot product
std::cout << "v1 . v3 = " << nda::dot(v1, v3) << std::endl;
std::cout << "v2 . v3 = " << nda::dot(v2, v3) << std::endl;
```

Output:

```
v1 . v3 = 0
v2 . v3 = 0
```

The cross product can also be expressed as the matrix-vector product of a skew-symmetric matrix and one of the vectors:
\f[
  \mathbf{a} \times \mathbf{b} =
  \begin{bmatrix}
  0 & -a_3 & a_2 \\
  a_3 & 0 & -a_1 \\
  -a_2 & a_1 & 0
  \end{bmatrix}
  \begin{bmatrix}
  b_1 \\
  b_2 \\
  b_3
  \end{bmatrix}
\f]

```cpp
// cross product via matrix-vector product
auto M_v1 = nda::matrix<double>{{0, -v1[2], v1[1]}, {v1[2], 0, -v1[0]}, {-v1[1], v1[0], 0}};
std::cout << "M_v1 = " << M_v1 << std::endl;
auto v3_mv = M_v1 * v2;
std::cout << "v3_mv = " << v3_mv << std::endl;
```

Output:

```
M_v1 =
[[0,-3,2]
 [3,0,-1]
 [-2,1,0]]
v3_mv = [-3,6,-3]
```

Comparing this result to `v3` above, we see that this is indeed correct.

Let us now turn to an eigenvalue problem.
**nda** offers the convenience functions nda::linalg::eigenelements and nda::linalg::eigenvalues to obtain the
eigenvalues and eigenvectors of a symmetric or hermitian matrix.

We start from the following symmetric matrix:

```cpp
// define a symmetric matrix
auto M1 = nda::matrix<double>{{4, -14, -12}, {-14, 10, 13}, {-12, 13, 1}};
std::cout << "M1 = " << M1 << std::endl;
```

Output:

```
M1 =
[[4,-14,-12]
 [-14,10,13]
 [-12,13,1]]
```

Getting the eigenvalues and eigenvectors is quite easy:

```cpp
// calculate the eigenvalues and eigenvectors of a symmetric matrix
auto [s, Q] = nda::linalg::eigenelements(M1);
std::cout << "Eigenvalues of M1: s = " << s << std::endl;
std::cout << "Eigenvectors of M1: Q = " << Q << std::endl;
```

Output:

```
Eigenvalues of M1: s = [-9.64366,-6.89203,31.5357]
Eigenvectors of M1: Q =
[[0.594746,-0.580953,0.555671]
 [-0.103704,-0.740875,-0.663588]
 [0.797197,0.337041,-0.500879]]
```

To check the correctness of our calculation, we us the fact that the matrix \f$ \mathbf{M}_1 \f$ can be factorized as
\f[
  \mathbf{M}_1 = \mathbf{Q} \mathbf{\Sigma} \mathbf{Q}^{-1} \; ,
\f]
where \f$ \mathbf{Q} \f$ is the matrix with the eigenvectors in its columns and \f$ \mathbf{\Sigma} \f$ is the diagonal
matrix of eigenvalues:

```cpp
// check the eigendecomposition
auto M1_reconstructed = Q * nda::diag(s) * nda::transpose(Q);
std::cout << "M1_reconstructed = " << M1_reconstructed << std::endl;
```

Output:

```
M1_reconstructed =
[[4,-14,-12]
 [-14,10,13]
 [-12,13,1]]
```

@subsection ex8_p8 Using the BLAS/LAPACK interface

While the functions in @ref linalg_tools offer a very user-friendly experience, @ref linalg_blas and @ref linalg_lapack
are more low-level and usually require more input from the users.

Let us show how to use the @ref linalg_lapack to solve a system of linear equations.
The system we want to solve is \f$ \mathbf{A}_1 \mathbf{x}_1 = \mathbf{b}_1 \f$ with

```cpp
// define the linear system of equations and the right hand side matrix
auto A1 = nda::matrix<double, nda::F_layout>{{3, 2, -1}, {2, -2, 4}, {-1, 0.5, -1}};
auto b1 = nda::vector<double>{1, -2, 0};
std::cout << "A1 = " << A1 << std::endl;
std::cout << "b1 = " << b1 << std::endl;
```

Output:

```
A1 =
[[3,2,-1]
 [2,-2,4]
 [-1,0.5,-1]]
b1 = [1,-2,0]
```

First, we perform an LU factorization using nda::lapack::getrf:

```cpp
// LU factorization using the LAPACK interface
auto ipiv = nda::vector<int>(3);
auto LU = A1;
auto info = nda::lapack::getrf(LU, ipiv);
if (info != 0) {
  std::cerr << "Error: nda::lapack::getrf failed with error code " << info << std::endl;
  return 1;
}
```

Here, `ipiv` is a pivot vector that is used internally by the LAPACK routine.
Because nda::lapack::getrf overwrites the input matrix with the result, we first copy the original matrix `A1` into a
new one `LU`.

Then we solve the system by calling nda::lapack::getrs with the LU factorization, the right hand side vector and the
pivot vector:

```cpp
// solve the linear system of equations using the LU factorization
nda::matrix<double, nda::F_layout> x1(3, 1);
x1(nda::range::all, 0) = b1;
info = nda::lapack::getrs(LU, x1, ipiv);
if (info != 0) {
  std::cerr << "Error: nda::lapack::getrs failed with error code " << info << std::endl;
  return 1;
}
std::cout << "x1 = " << x1(nda::range::all, 0) << std::endl;
```

Output:

```
x1 = [1,-2,-2]
```

Since `b1` is a vector but nda::lapack::getrs expects a matrix, we first copy it into the matrix `x1`.
After the LAPACK call return, `x1` contains the result in its first column.

Let's check that this is actually the solution to our original system of equations:

```cpp
// check the solution
std::cout << "A1 * x1 = " << A1 * x1(nda::range::all, 0) << std::endl;
```

Output:

```
A1 * x1 = [1,-2,2.22045e-16]
```

Considering the finite precision of our calculations, this is indeed equal to the right hand side vector `b1`.

> **Note**: BLAS and LAPACK assume Fortran-ordered arrays. We therefore recommend to work with matrices in the
> nda::F_layout to avoid any confusion.

