@page ex7 Example 7: Making use of symmetries

[TOC]

In this example, we show how use, spot and define symmetries in **nda** arrays.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <nda/sym_grp.hpp>
#include <complex>
#include <functional>
#include <iostream>

int main(int argc, char *argv[]) {
  // size of the matrix
  constexpr int N = 3;

  // some useful typedefs
  using idx_t = std::array<long, 2>;
  using sym_t = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;

  // code snippets go here ...
}
```

Here, we have defined some basic quantities/types:
- \f$ N \times N \f$ is the size of the matrix that we are working with.
- `idx_t` defines a multi-dimensional index type (in our case it is 2-dimensional).
- `sym_t` is a tuple type consisting of a multi-dimensional index and an nda::operation.
- `sym_func_t` is a callable type that takes a multi-dimensional index and returns a `sym_t` object. The index in the
`sym_t` object refers to an element which is related to the element at the original index by the nda::operation.

Let's take a look at a simple example.

@subsection ex7_p1 Defining the symmetry

We start by defining a hermitian symmetry for our matrix.
A symmetry is simply an object of type `sym_func_t` that specifies how elements in an array are related to one another.

```cpp
// define a hermitian symmetry (satisfies nda::NdaSymmetry)
auto h_symmetry = [](idx_t const &x) {
  if (x[0] == x[1]) return sym_t{x, nda::operation{false, false}}; // sign flip = false, complex conjugate = false
  return sym_t{idx_t{x[1], x[0]}, nda::operation{false, true}};    // sign flip = false, complex conjugate = true
};
```

As you can see, `h_symmetry` is a callable object that takes an `idx_t` and returns a `sym_t` object.
The two lines have the following meaning:
1. Diagonal elements are not related to any other elements (except to themselves by the identity operation).
2. Off-diagonal elements with an index `(i,j)` are related to the element `(j,i)` by a complex conjugation.

@subsection ex7_p2 Constructing the symmetry group

Once all of the desired symmetries have been defined, we can construct an nda::sym_grp object:

```cpp
// construct the symmetry group
nda::array<std::complex<double>, 2> A(N, N);
auto grp = nda::sym_grp{A, std::vector<sym_func_t>{h_symmetry}};
```

The constructor of the symmetry group takes an array and a list of symmetries as input.
It uses the shape of the array and the symmetries to find all symmetry classes.

For our \f$ 3 \times 3 \f$ hermitian matrix, there should be 6 symmetry classes corresponding to the 6 independent
elements of the matrix:

```cpp
// check the number of symmetry classes
std::cout << "Number of symmetry classes: " << grp.num_classes() << std::endl;
```

Output:

```
Number of symmetry classes: 6
```

A symmetry class contains all the elements that are related to each other by some symmetry.
It stores the flat index of the element as well as the nda::operation specifying how it is related to a special
representative element of the class.
In practice, the representative element is simply the first element of the class.

Let us print all the symmetry classes:

```cpp
// print the symmetry classes
for (int i = 1; auto const &c : grp.get_sym_classes()) {
  std::cout << "Symmetry class " << i << ":" << std::endl;
  for (auto const &x : c) {
    std::cout << "  Idx: " << x.first << ", Sign flip: " << x.second.sgn << ", Complex conjugation: " << x.second.cc << std::endl;
  }
}
```

Output:

```
Symmetry class 1:
  Idx: 0, Sign flip: 0, Complex conjugation: 0
Symmetry class 1:
  Idx: 1, Sign flip: 0, Complex conjugation: 0
  Idx: 3, Sign flip: 0, Complex conjugation: 1
Symmetry class 1:
  Idx: 2, Sign flip: 0, Complex conjugation: 0
  Idx: 6, Sign flip: 0, Complex conjugation: 1
Symmetry class 1:
  Idx: 4, Sign flip: 0, Complex conjugation: 0
Symmetry class 1:
  Idx: 5, Sign flip: 0, Complex conjugation: 0
  Idx: 7, Sign flip: 0, Complex conjugation: 1
Symmetry class 1:
  Idx: 8, Sign flip: 0, Complex conjugation: 0
```

Keep in mind that elements of our matrix have the following flat indices:

```cpp
// print flat indices
nda::array<int, 2> B(3, 3);
for (int i = 0; auto &x : B) x = i++;
std::cout << B << std::endl;
```

Output:

```
[[0,1,2]
 [3,4,5]
 [6,7,8]]
```

Now we can see why the elements 0, 4 and 8 are in their own symmetry class and elements 1 and 3, 2 and 6 and elements 5
and 7 are related to each other via a complex conjugation.

@subsection ex7_p3 Initializing an array

One of the features of symmetry groups is that they can be used to initialize or assign to an existing array with an
initializer function satisfying the nda::NdaInitFunc concept:

```cpp
// define an initializer function
int count_eval = 0;
auto init_func = [&count_eval, &B](idx_t const &idx) {
  ++count_eval;
  const double val = B(idx[0], idx[1]);
  return std::complex<double>{val, val};
};
```

An nda::NdaInitFunc object should take a multi-dimensional index as input and return a value for the given index.
Since we are working with complex matrices, the return type in our case is `std::complex<double>`.

`count_eval` simply counts the number of times `init_func` has been called.

Let's initialize our array:

```cpp
// initialize the array using the symmetry group
grp.init(A, init_func);
std::cout << "A = " << A << std::endl;
```

Output:

```
A =
[[(0,0),(1,1),(2,2)]
 [(1,-1),(4,4),(5,5)]
 [(2,-2),(5,-5),(8,8)]]
```

As expected the array is hermitian.

The advantage of using symmetry groups to initialize an array becomes clear when we check how often our `init_func`
object has been called:

```cpp
// check the number of evaluations
std::cout << "Number of evaluations: " << count_eval << std::endl;
```

Output:

```
Number of evaluations: 6
```

It has been called exactly once for each symmetry class.

In our simple example, this does not really make a difference but for large arrays where the number of symmetry classes
is much smaller than the size of the array and where evaluating `init_func` is expensive, this can give a considerable
speedup.

@subsection ex7_p4 Representative data

We have already learned above about representative elements of symmetry classes.
The symmetry group object gives us access to those elements:

```cpp
// get representative elements
auto reps = grp.get_representative_data(A);
auto reps_view = nda::array_view<std::complex<double>, 1>(reps);
std::cout << "Representative elements = " << reps_view << std::endl;
```

Output:

```
Representative elements = [(0,0),(1,1),(2,2),(4,4),(5,5),(8,8)]
```

The representative elements in our case simply belong to the upper triangular part of the matrix.

Representative data can also be used to initialize or assign to an array.
Let's try this out:

```cpp
// use representative data to initialize a new array
reps_view *= 2.0;
nda::array<std::complex<double>, 2> B(N, N);
grp.init_from_representative_data(B, reps);
std::cout << "B = " << B << std::endl;
```

Output:

```
B =
[[(0,0),(2,2),(4,4)]
 [(2,-2),(8,8),(10,10)]
 [(4,-4),(10,-10),(16,16)]]
```

Here, we first multiplied the original representative data by 2 and then initialized a new array with it.

@subsection ex7_p5 Symmetrizing an array

The nda::sym_grp class provides a method that let's us symmetrize an existing array and simultaneously obtain the
maximum symmetry violation.

For each symmetry class, it first calculates the representative element by looping over all elements of the class,
applying the stored nda::operation to each element (for a sign flip and a complex conjugation this is equal to applying
the inverse operation) and summing the resulting values.
The sum divided by the number of elements in the symmetry class gives us the representative element.

It then compares each element in the class to the representative element, notes their absolute difference to obtain
the maximum symmetry violation and sets the element to the value of the representative element (modulo the
nda::operation).

Symmetrizing one of our already hermitian matrices shouldn't change the matrix and the maximum symmetry violation should
be zero:

```cpp
// symmetrize an already symmetric array
auto v1 = grp.symmetrize(A);
std::cout << "Symmetrized A = " << A << std::endl;
std::cout << "Max. violation at index " << v1.first << " = " << v1.second << std::endl;
```

Output:

```
Symmetrized A =
[[(0,0),(1,1),(2,2)]
 [(1,-1),(4,4),(5,5)]
 [(2,-2),(5,-5),(8,8)]]
Max. violation at index (0 0) = 0
```

Let's change one of the off-diagonal elements:

```cpp
// change an off-diagonal element
A(0, 2) *= 2.0;
std::cout << "A = " << A << std::endl;
```

Output:

```
A =
[[(0,0),(1,1),(4,4)]
 [(1,-1),(4,4),(5,5)]
 [(2,-2),(5,-5),(8,8)]]
```

The matrix is not hermitian anymore and therefore violates our initial `h_symmetry`.

If we symmetrize again:

```cpp
// symmetrize again
auto v2 = grp.symmetrize(A);
std::cout << "Symmetrized A = " << A << std::endl;
std::cout << "Max. violation at index " << v2.second << " = " << v2.first << std::endl;
```

Output:

```
Symmetrized A =
[[(0,0),(1,1),(3,3)]
 [(1,-1),(4,4),(5,5)]
 [(3,-3),(5,-5),(8,8)]]
Max. violation at index (0 2) = 1.41421
```

We can see that the symmetrized matrix is now different from the original and that the maximum symmetry violation is
not equal to zero.
