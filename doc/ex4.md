@page ex4 Example 4: Views and slices

[TOC]

In this example, we show how to construct and work with views and slices.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  // code snippets go here ...
}
```

@subsection ex4_p1 Creating a full view on an array/view

We have already seen in @ref ex1_p6 how we can get a full view on an existing array by doing an empty function call:

```cpp
// create a full view on an array
auto A = nda::array<int, 2>(5, 5);
for (int i = 0; auto &x : A) x = i++;
auto A_v = A();
std::cout << "A_v = " << A_v << std::endl;
```

Output:

```
A_v =
[[0,1,2,3,4]
 [5,6,7,8,9]
 [10,11,12,13,14]
 [15,16,17,18,19]
 [20,21,22,23,24]]
```

The same could have been achieved by

- calling the member function nda::basic_array::as_array_view,
- calling the free function nda::make_array_view or
- taking a full slice of the array, e.g. `A(nda::ellipsis{})` or `A(nda::range::all, nda::range::all)`.

Since views behave mostly as arrays, we can also take a view of a view:

```cpp
// create a view of a view
auto A_vv = A_v();
std::cout << "A_vv = " << A_vv << std::endl;
```

Output:

```
A_vv =
[[0,1,2,3,4]
 [5,6,7,8,9]
 [10,11,12,13,14]
 [15,16,17,18,19]
 [20,21,22,23,24]]
```

@subsection ex4_p2 Value type of views

While the value type of an array is always non-const, views can have const or non-const value types.

Usually, views will have the same value type as the underlying array/view:

```cpp
// check the value type of the view and assign to it
static_assert(std::is_same_v<decltype(A_v)::value_type, int>);
A_v(0, 0) = -1;
std::cout << "A = " << A << std::endl;
```

Output:

```
A =
[[-1,1,2,3,4]
 [5,6,7,8,9]
 [10,11,12,13,14]
 [15,16,17,18,19]
 [20,21,22,23,24]]
```

This is not the case if we take a view of a const array/view, then the value type will be const:

```cpp
// taking a view of a const array
auto A_vc = static_cast<const nda::array<int, 2>>(A)();
static_assert(std::is_same_v<decltype(A_vc)::value_type, const int>);

// taking a view of a view with a const value type
auto A_vvc = A_vc();
static_assert(std::is_same_v<decltype(A_vvc)::value_type, const int>);
```

As expected, we cannot assign to const arrays/views or views with a const value type, i.e.

```cpp
// this will not compile
A_vc(0, 0) = -2;
```

@subsection ex4_p3 Creating a slice of an array/view

@ref ex1_p7 has already explained what a slice is and how we can create one.
In the following, we will give some more examples to show how slices can be used in practice.

Here is the original array that we will be working on:

```cpp
// original array
A(0, 0) = 0;
std::cout << "A = " << A << std::endl;
```

Output:

```
A =
[[0,1,2,3,4]
 [5,6,7,8,9]
 [10,11,12,13,14]
 [15,16,17,18,19]
 [20,21,22,23,24]]
```

Let us start by taking a slice of every other column:

```cpp
// slice containing every other column
auto S_1 = A(nda::range::all, nda::range(0, 5, 2));
std::cout << "S_1 = " << S_1 << std::endl;
```

Output:

```
S_1 =
[[0,2,4]
 [5,7,9]
 [10,12,14]
 [15,17,19]
 [20,22,24]]
```

Now we take a slice of every other row of this slice:

```cpp
// slice containing every other row of S_1
auto S_2 = S_1(nda::range(0, 5, 2), nda::range::all);
std::cout << "S_2 = " << S_2 << std::endl;
```

Output:

```
S_2 =
[[0,2,4]
 [10,12,14]
 [20,22,24]]
```

We could have gotten the same slice with

```cpp
// slice containing every other row and column
auto S_3 = A(nda::range(0, 5, 2), nda::range(0, 5, 2));
std::cout << "S_3 = " << S_3 << std::endl;
```

Output:

```
S_3 =
[[0,2,4]
 [10,12,14]
 [20,22,24]]
```

@subsection ex4_p4 Assigning to views

Before assigning to the view `S_3`, let's make a copy of its contents so that we can restore everything later on:

```cpp
// make a copy of the view using nda::make_regular
auto A_tmp = nda::make_regular(S_3);
```

> nda::make_regular turns view, slices and lazy expressions into regular nda::basic_array objects.

`A_tmp` is an nda::array of the same size as `S_3` and contains the same values but in a different memory location.

Now, we can assign to the view just like to arrays (see @ref ex3):

```cpp
// assign a scalar to a view
S_3 = 0;
std::cout << "S_3 = " << S_3 << std::endl;
```

Output:

```
S_3 =
[[0,0,0]
 [0,0,0]
 [0,0,0]]
```

Any changes that we make to a view will be reflected in the original array and all other views that are currently
looking at the same memory locations:

```cpp
// check the changes to the original array and other views
std::cout << "S_1 = " << S_1 << std::endl;
std::cout << "A = " << A << std::endl;
```

Output:

```
S_1 =
[[0,0,0]
 [5,7,9]
 [0,0,0]
 [15,17,19]
 [0,0,0]]
A =
[[0,1,0,3,0]
 [5,6,7,8,9]
 [0,11,0,13,0]
 [15,16,17,18,19]
 [0,21,0,23,0]]
```

To restore the original array, we can assign our copy from before:

```cpp
// assign an array to a view
S_3 = A_tmp;
std::cout << "A = " << A << std::endl;
```

Output:

```
A =
[[0,1,2,3,4]
 [5,6,7,8,9]
 [10,11,12,13,14]
 [15,16,17,18,19]
 [20,21,22,23,24]]
```

> **Note**: In contrast to arrays, views cannot be resized. When assigning some general nda::Array object to a view,
> their shapes have to match, otherwise this may result in undefined behavior.

@subsection ex4_p5 Copy/Move operations

The copy and move operations of views are a little bit different than their array counterparts:

- The copy constructor makes a new view that points to the same data as the other view.
- The move constructor does the same as the copy constructor.
- The copy assignment operator makes a deep copy of the contents of the other view.
- The move assignment operator does the same as the copy assignment.

```cpp
// copy construct a view
auto S_3_copy = S_3;
std::cout << "S_3.data() == S_3_copy.data() = " << (S_3.data() == S_3_copy.data()) << std::endl;

// move construct a view
auto S_3_move = std::move(S_3);
std::cout << "S_3.data() == S_3_move.data() = " << (S_3.data() == S_3_move.data()) << std::endl;

// copy assign to a view
auto B = nda::array<int, 2>(S_3.shape());
auto B_v = B();
B_v = S_3;
std::cout << "B = " << B << std::endl;

// move assign to a view
auto C = nda::array<int, 2>(S_3.shape());
auto C_v = C();
C_v = std::move(S_3);
std::cout << "C = " << C << std::endl;
```

Output:

```
S_3.data() == S_3_copy.data() = 1
S_3.data() == S_3_move.data() = 1
B =
[[0,2,4]
 [10,12,14]
 [20,22,24]]
C =
[[0,2,4]
 [10,12,14]
 [20,22,24]]
```

@subsection ex4_p6 Operating on views/slices

We can perform various arithmetic operations, mathematical functions and algorithms with views and slices just like we
did with arrays in @ref ex1_p8 and @ref ex1_p9.

```cpp
// arithmetic operations and math functions
C_v = S_3 * 2 + nda::pow(S_3, 2);
std::cout << "C = " << C << std::endl;

// algorithms
std::cout << "min_element(S_3) = " << nda::min_element(S_3) << std::endl;
std::cout << "max_element(S_3) = " << nda::max_element(S_3) << std::endl;
std::cout << "sum(S_3) = " << nda::sum(S_3) << std::endl;
```

Output:

```
C =
[[0,8,24]
 [120,168,224]
 [440,528,624]]
min_element(S_3) = 0
max_element(S_3) = 24
sum(S_3) = 108
product(S_3) = 0
```

@subsection ex4_p7 Rebinding a view to another array/view

If we want to bind an existing view to a new array/view/memory location, we cannot simply use the copy assignment (since
it makes a deep copy of the view's contents). Instead we have to call nda::basic_array_view::rebind:

```cpp
// rebind a view
std::cout << "S_3.data() == C_v.data() = " << (S_3.data() == C_v.data()) << std::endl;
C_v.rebind(S_3);
std::cout << "S_3.data() == C_v.data() = " << (S_3.data() == C_v.data()) << std::endl;
```

Output:

```
S_3.data() == C_v.data() = 0
S_3.data() == C_v.data() = 1
```

@subsection ex4_p8 Viewing generic 1-dimensional ranges

The views in **nda** can also view generic 1-dimensional ranges like `std::vector` or `std::array`.
The only requirement is that they are contiguous:

```cpp
// take a view on a std::array
std::array<double, 5> arr{1.0, 2.0, 3.0, 4.0, 5.0};
auto arr_v = nda::basic_array_view(arr);
std::cout << "arr_v = " << arr_v << std::endl;

// change the value of the vector through the view
arr_v *= 2.0;
std::cout << "arr = " << arr << std::endl;
```

Output:

```
arr_v = [1,2,3,4,5]
arr = (2 4 6 8 10)
```

@subsection ex4_p9 Factories and transformations

@ref av_factories contain various functions to create new and transform existing views.

In the following, we will show some examples.

Let us start from this array:

```cpp
// original array
auto D = nda::array<int, 2>(3, 4);
for (int i = 0; auto &x : D) x = i++;
std::cout << "D = " << D << std::endl;
```

Output:

```
D =
[[0,1,2,3]
 [4,5,6,7]
 [8,9,10,11]]
```

We can transpose `D` using nda::transpose:

```cpp
// create a transposed view
auto D_t = nda::transpose(D);
std::cout << "D_t = " << D_t << std::endl;
```

Output:

```
D_t =
[[0,4,8]
 [1,5,9]
 [2,6,10]
 [3,7,11]]
```

> **Note**: `D_t` is a view and not an array. That means that transposing an existing array/view is a very cheap
> operation since it doesn't allocate or copy any data.
>
> The same is true for most of the other transformations.

As long as the data is contiguous in memory and the memory layout is either in C-order or Fortran-order, we are allowed
to change the shape of an array/view:

```cpp
// reshape the array
auto D_r = nda::reshape(D, 2, 6);
std::cout << "D_r = " << D_r << std::endl;

// reshape the view
auto D_tr = nda::reshape(D_t, 2, 6);
std::cout << "D_tr = " << D_tr << std::endl;
```

Output:

```
D_r =
[[0,1,2,3,4,5]
 [6,7,8,9,10,11]]
D_tr =
[[0,2,4,6,8,10]
 [1,3,5,7,9,11]]
```

To interpret some array/view as a contiguous 1-dimensional range, we can call nda::flatten which is just a convenient
wrapper around nda::reshape:

```cpp
// flatten the view
auto D_t_flat = nda::flatten(D_t);
std::cout << "D_t_flat = " << D_t_flat << std::endl;

// flatten the array using reshape
auto D_flat = nda::reshape(D, D.size());
std::cout << "D_flat = " << D_flat << std::endl;
```

Output:

```
D_t_flat = [0,1,2,3,4,5,6,7,8,9,10,11]
D_flat = [0,1,2,3,4,5,6,7,8,9,10,11]
```

**nda** provides some more advanced transformations which are especially useful for higher-dimensional arrays/views.
We refere the interested user to the @ref documentation.
