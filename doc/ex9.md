@page ex9 Example 9: Tensor support

[TOC]

In this example, we show how to use the @ref tensor_ops in **nda** to perform tensor addition, contraction, full dot 
product, reduction to a scalar, in-place scaling and element-wise binary ops.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <complex>
#include <iostream>

int main() {
  using nda::tensor::binary_op;
  using nda::tensor::unary_op;
  // code snippets go here ...
}
```

@section ex9_p1 What is a tensor in nda?

In **nda**, a tensor is simply an nda::array of arbitrary rank. The `nda::tensor` namespace adds a small set of 
operations that work uniformly across ranks and use the Einstein summation convention, where dimensions are named by 
single-letter index strings and repeated indices imply summation.

In this example, all arrays live on the host (the default address space). The tensor operations dispatch to the
optimized [TBLIS](https://github.com/MatthewsResearchGroup/tblis) backend whenever it is available, and otherwise fall
back to plain nda expressions for the cases where this is possible (identical ranks and identical index orderings).

@section ex9_p2 Prerequisite: building with TBLIS support

Tensor contraction with arbitrary einsum patterns cannot be expressed by the basic nda expression machinery alone.
To enable the full feature set demonstrated here, configure **nda** with `-DTblisSupport=ON`:

```
cmake ../nda.src -DCMAKE_BUILD_TYPE=Release -DTblisSupport=ON
```

@section ex9_p3 Constructing tensors and index strings

We start by creating a rank-3 tensor of shape \f$ 2 \times 3 \times 4 \f$ and filling it with consecutive integers:

```cpp
// construct a rank-3 tensor of shape 2x3x4 and fill it with consecutive integers
auto A = nda::array<double, 3>(2, 3, 4);
for (int n = 0; auto &x : A) x = n++;
std::cout << "A = " << A << std::endl;
std::cout << "A.shape() = " << A.shape() << std::endl;
```

Output:

```
A = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
A.shape() = (2 3 4)
```

Note that nda streams the underlying storage of rank-3 (and higher) arrays as a single flat list.

The tensor operations identify dimensions by single-letter labels. For convenience, nda::tensor::default_index returns a 
default string of length equal to the rank, starting with `'a'`:

```cpp
std::cout << "default_index<3>() = " << nda::tensor::default_index<3>() << std::endl;
```

Output:

```
default_index<3>() = abc
```

@section ex9_p4 Tensor addition

nda::tensor::add performs an in-place tensor addition with optional scaling:
\f[
  B_{\text{idx}_B} \leftarrow \alpha \, A_{\text{idx}_A} + \beta \, B_{\text{idx}_B} \;.
\f]

```cpp
// in-place tensor addition: B_ijk <- alpha * A_ijk + beta * B_ijk
auto B    = nda::array<double, 3>(2, 3, 4);
B         = 1.0;
double a1 = 2.0;
double b1 = 3.0;
nda::tensor::add(a1, A, "ijk", b1, B, "ijk");
std::cout << "B = 2*A + 3 = " << B << std::endl;
```

Output:

```
B = 2*A + 3 = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]
```

Every entry of `B` equals \f$ 2 A_{ijk} + 3 \f$.

@section ex9_p5 Tensor contraction

nda::tensor::contract is the central tensor operation: a general einsum-style contraction
\f[
  C_{\text{idx}_C} \leftarrow \alpha \, A_{\text{idx}_A} \cdot B_{\text{idx}_B} + \beta \, C_{\text{idx}_C} \;,
\f]
where indices that appear in both \f$ A \f$ and \f$ B \f$ but not in \f$ C \f$ are summed over.

A rank-3 contraction \f$ C_{il} = \sum_{jk} A_{ijk} B_{jkl} \f$:

```cpp
// tensor contraction: C_il = sum_jk A_ijk * B_jkl
auto A2 = nda::array<double, 3>(2, 3, 4);
for (int n = 0; auto &x : A2) x = 0.1 * n++;
auto B2 = nda::array<double, 3>(3, 4, 5);
for (int n = 0; auto &x : B2) x = 0.1 * n++;
auto C2 = nda::array<double, 2>::zeros({2, 5});
nda::tensor::contract(1.0, A2, "ijk", B2, "jkl", 0.0, C2, "il");
std::cout << "C_il = A_ijk * B_jkl = " << C2 << std::endl;
```

Output:

```
C_il = A_ijk * B_jkl =
[[25.3,25.96,26.62,27.28,27.94]
 [64.9,67,69.1,71.2,73.3]]
```

The matrix product \f$ \mathbf{A}_m \mathbf{B}_m \f$ is just the special rank-2 case
\f$ C_{ik} = \sum_j A_{ij} B_{jk} \f$:

```cpp
// matrix-matrix multiplication expressed as a rank-2 contraction: C_ik = A_ij * B_jk
auto Am = nda::array<double, 2>{{1, 2, 3}, {4, 5, 6}};
auto Bm = nda::array<double, 2>{{7, 8}, {9, 10}, {11, 12}};
auto Cm = nda::array<double, 2>::zeros({2, 2});
nda::tensor::contract(1.0, Am, "ij", Bm, "jk", 0.0, Cm, "ik");
std::cout << "A_ij * B_jk = " << Cm << std::endl;
```

Output:

```
A_ij * B_jk =
[[58,64]
 [139,154]]
```

This is exactly the result returned by `Am * Bm` in the @ref ex8_p6 "matrix algebra" of nda::matrix; **nda** exposes
the same operation in einsum form, generalized to arbitrary rank.

@section ex9_p6 Full tensor dot product

nda::tensor::dot computes the scalar
\f[
  z \leftarrow \sum_{\text{idx}_A = \text{idx}_B} A_{\text{idx}_A} \cdot B_{\text{idx}_B} \;,
\f]
i.e. it contracts every dimension of \f$ A \f$ against the matching dimension of \f$ B \f$.

```cpp
// tensor dot product (sum over all matching indices, returns a scalar)
auto d_indexed = nda::tensor::dot(A2, "ijk", A2, "ijk");
auto d_default = nda::tensor::dot(A2, A2);
std::cout << "dot(A2, A2) [indexed] = " << d_indexed << std::endl;
std::cout << "dot(A2, A2) [default] = " << d_default << std::endl;
```

Output:

```
dot(A2, A2) [indexed] = 43.24
dot(A2, A2) [default] = 43.24
```

The two calls return the same value: the indexed form spells out the pairing explicitly, while the convenience overload
uses nda::tensor::default_index for both operands.

@section ex9_p7 Full tensor reduction

nda::tensor::reduce reduces every element of a tensor to a single scalar via one of the operations defined by
nda::tensor::binary_op:

```cpp
// full tensor reductions to a scalar with different binary operations
std::cout << "reduce(A, SUM)    = " << nda::tensor::reduce(A) << std::endl;
std::cout << "reduce(A, MAX)    = " << nda::tensor::reduce(A, binary_op::MAX) << std::endl;
std::cout << "reduce(A, MIN)    = " << nda::tensor::reduce(A, binary_op::MIN) << std::endl;
std::cout << "reduce(A, NORM_2) = " << nda::tensor::reduce(A, binary_op::NORM_2) << std::endl;
```

Output:

```
reduce(A, SUM)    = 276
reduce(A, MAX)    = 23
reduce(A, MIN)    = 0
reduce(A, NORM_2) = 65.7571
```

The default reduction is `binary_op::SUM`. See nda::tensor::binary_op for the full list of supported reductions.

@section ex9_p8 In-place tensor scaling

nda::tensor::scale rescales a tensor in place and optionally applies an element-wise unary operation
(see nda::tensor::unary_op):
\f[
  A \leftarrow \alpha \, \text{op}(A) \;.
\f]

```cpp
// in-place tensor scaling with an optional element-wise unary operation
auto S = nda::array<double, 3>(2, 3, 4);
for (int n = 0; auto &x : S) x = static_cast<double>(n++ + 1);
nda::tensor::scale(2.0, S);
std::cout << "2 * S = " << S << std::endl;
nda::tensor::scale(1.0, S, unary_op::SQRT);
std::cout << "sqrt(2 * S) = " << S << std::endl;
```

Output:

```
2 * S = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48]
sqrt(2 * S) = [1.41421,2,2.44949,2.82843,3.16228,3.4641,3.74166,4,4.24264,4.47214,4.69042,4.89898,5.09902,5.2915,5.47723,5.65685,5.83095,6,6.16441,6.32456,6.48074,6.63325,6.78233,6.9282]
```

After the two calls, every entry of `S` equals \f$ \sqrt{2 \, n} \f$ for \f$ n = 1, 2, \dots, 24 \f$.

For a complex tensor, `unary_op::CONJ` flips the sign of the imaginary parts:

```cpp
using cplx = std::complex<double>;
auto Z     = nda::array<cplx, 3>(2, 2, 2);
for (int n = 0; auto &x : Z) {
  x = cplx(n, n + 1);
  ++n;
}
nda::tensor::scale(cplx{1.0, 0.0}, Z, unary_op::CONJ);
std::cout << "conj(Z) = " << Z << std::endl;
```

Output:

```
conj(Z) = [(0,-1),(1,-2),(2,-3),(3,-4),(4,-5),(5,-6),(6,-7),(7,-8)]
```

@section ex9_p9 Element-wise binary operations

nda::tensor::elementwise applies an arbitrary nda::tensor::binary_op element-wise:
\f[
  B_{\text{idx}_B} \leftarrow \text{op}(\alpha \, A_{\text{idx}_A}, \, \beta \, B_{\text{idx}_B}) \;.
\f]

With `binary_op::SUM` it reduces to nda::tensor::add:

```cpp
// element-wise binary operation: B_ijk <- op(alpha * A_ijk, beta * B_ijk)
auto E1 = nda::array<double, 3>(2, 3, 4);
for (int n = 0; auto &x : E1) x = static_cast<double>(n++);
auto E2 = nda::array<double, 3>(2, 3, 4);
E2      = 1.0;
nda::tensor::elementwise(1.0, E1, "ijk", 1.0, E2, "ijk", binary_op::SUM);
std::cout << "E2 <- E1 + E2 = " << E2 << std::endl;
```

Output:

```
E2 <- E1 + E2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
```

With `binary_op::PROD` we get an element-wise (Hadamard) product:

```cpp
auto F1 = nda::array<double, 3>(2, 3, 4);
F1      = 2.0;
auto F2 = nda::array<double, 3>(2, 3, 4);
F2      = 3.0;
nda::tensor::elementwise(1.0, F1, "ijk", 1.0, F2, "ijk", binary_op::PROD);
std::cout << "F2 <- F1 * F2 = " << F2 << std::endl;
```

Output:

```
F2 <- F1 * F2 = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]
```

Every entry of `F2` is \f$ 1 \cdot 2 \cdot 1 \cdot 3 = 6 \f$.

@section ex9_full Full source

The complete, compilable source for this example:

@include ex9.cpp
