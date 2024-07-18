@page ex5 Example 5: HDF5 support

[TOC]

In this example, we show how to write/read **nda** arrays and views to/from HDF5 files.

**nda** uses the [h5](https://triqs.github.io/h5/unstable/index.html) library and especially the
[h5::array_interface](https://triqs.github.io/h5/unstable/group__rw__arrayinterface.html) to provide HDF5 support.

All the following code snippets are part of the same `main` function:

```cpp
#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <h5/h5.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  // HDF5 file
  h5::file file("ex5.h5", 'w');

  // code snippets go here ...
}
```

Before we dive into the HDF5 capabilities of **nda**, let us first specify the array that we will be working with:

```cpp
// original array
auto A = nda::array<int, 2>(5, 5);
for (int i = 0; auto &x : A) x = i++;
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

> **Note**: In the examples below, we will restrict ourselves to arrays and views in C-order. The reason is that when
> writing/reading to/from HDF5, the interface always checks if the arrays/views are in C-order. If this is not the case,
> it will use a temporary C-order array to perform the writing/reading.

@subsection ex5_p1 Writing an array/view

Writing an array to an HDF5 file is as simple as

```cpp
// write the array to the file
h5::write(file, "A", A);
```

Dumping the HDF5 file gives

```
HDF5 "ex5.h5" {
GROUP "/" {
   DATASET "A" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 5, 5 ) / ( 5, 5 ) }
      DATA {
      (0,0): 0, 1, 2, 3, 4,
      (1,0): 5, 6, 7, 8, 9,
      (2,0): 10, 11, 12, 13, 14,
      (3,0): 15, 16, 17, 18, 19,
      (4,0): 20, 21, 22, 23, 24
      }
   }
}
}
```

The array is written into a newly created dataset with a dataspace that has the same dimensions and the same shape as
the original array.
In this case, it is a 5-by-5 dataspace.

When writing to HDF5, the interface doesn't distinguish between arrays and views.
We could have done the same with a view:

```cpp
// write a view to HDF5
h5::write(file, "A_v", A(nda::range(0, 5, 2), nda::range(0, 5, 2)));
```

Dumping the corresponding HDF5 dataspace gives

```
HDF5 "ex5.h5" {
DATASET "/A_v" {
   DATATYPE  H5T_STD_I32LE
   DATASPACE  SIMPLE { ( 3, 3 ) / ( 3, 3 ) }
   DATA {
   (0,0): 0, 2, 4,
   (1,0): 10, 12, 14,
   (2,0): 20, 22, 24
   }
}
}
```

Here, we write a view of every other row and column of the original array.
Again, the created dataspace `A_v` has the same dimensions and shape as the object that is written.
In this case, a 3-by-3 view.

> **Note**: nda::h5_write takes a fourth parameter which determines if the data should be compressed before it is
> written. By default, this is set to `true`. To turn the compression off, one can specify it in the `h5::write` call,
> e.g
> ```cpp
> h5::write(file, "A", A, /* compression off */ false);
> ```

@subsection ex5_p2 Reading into an array/view

Reading a full dataset into an array is straightforward:

```cpp
// read a dataset into an array
auto A_r = nda::array<int, 2>();
h5::read(file, "A", A_r);
std::cout << "A_r = " << A_r << std::endl;
```

Output:

```
A_r =
[[0,1,2,3,4]
 [5,6,7,8,9]
 [10,11,12,13,14]
 [15,16,17,18,19]
 [20,21,22,23,24]]
```

As you can see, the array does not need to have the same shape as the dataset, since the nda::h5_read function will
resize it if needed.

The same is not true for views.
Views cannot be resized, so when we read into a view, we have to make sure that it has the correct shape (otherwise an
exception will be thrown):

```cpp
// read a dataset into a view
auto B = nda::zeros<int>(5, 5);
auto B_v = B(nda::range(1, 4), nda::range(0, 5, 2));
h5::read(file, "A_v", B_v);
std::cout << "B = " << B << std::endl;
```

Output:

```
B =
[[0,0,0,0,0]
 [0,0,2,0,4]
 [10,0,12,0,14]
 [20,0,22,0,24]
 [0,0,0,0,0]]
```

Here, we read the 3-by-3 dataset `A_v` into a view `B_v` consisting of every other column and the rows 1, 2 and 3 of the
underlying 5-by-5 array `B`.

@subsection ex5_p3 Writing to a slice of an existing dataset

So far we have only written to an automatically created dataset with exactly the same size and shape as the array/view
that is being written.
It is also possible to write to a slice of an existing dataset as long as the selected slice has the same shape and size
as the array/view.

To demonstrate this, let us first create a dataset and zero it out (in production code, one would probably call the HDF5
C library directly to create a dataspace and a dataset but this is not needed for this simple example):

```cpp
// prepare a dataset
h5::write(file, "B", nda::zeros<int>(5, 5));
```

Dumping this dataset gives

```
HDF5 "ex5.h5" {
DATASET "/B" {
   DATATYPE  H5T_STD_I32LE
   DATASPACE  SIMPLE { ( 5, 5 ) / ( 5, 5 ) }
   DATA {
   (0,0): 0, 0, 0, 0, 0,
   (1,0): 0, 0, 0, 0, 0,
   (2,0): 0, 0, 0, 0, 0,
   (3,0): 0, 0, 0, 0, 0,
   (4,0): 0, 0, 0, 0, 0
   }
}
}
```

Then, we can take a slice of this dataset, e.g. by specifying every other row:

```cpp
// a slice that specifies every other row of "B"
auto slice_r024 = std::make_tuple(nda::range(0, 5, 2), nda::range::all);
```

Let's write the first 3 rows of `A` to this slice:

```cpp
// write the first 3 rows of A to the slice
h5::write(file, "B", A(nda::range(0, 3), nda::range::all), slice_r024);
```

Dumping the dataset gives

```
HDF5 "ex5.h5" {
DATASET "/B" {
   DATATYPE  H5T_STD_I32LE
   DATASPACE  SIMPLE { ( 5, 5 ) / ( 5, 5 ) }
   DATA {
   (0,0): 0, 1, 2, 3, 4,
   (1,0): 0, 0, 0, 0, 0,
   (2,0): 5, 6, 7, 8, 9,
   (3,0): 0, 0, 0, 0, 0,
   (4,0): 10, 11, 12, 13, 14
   }
}
}
```

Under the hood, **nda** takes the slice and transforms it into an HDF5 hyperslab to which the data is then written.

If we write the remaining last 2 rows of `A` to the empty rows in `B`,

```cpp
// write the last 2 rows of A to the empty rows in "B"
auto slice_r13 = std::make_tuple(nda::range(1, 5, 2), nda::range::all);
h5::write(file, "B", A(nda::range(3, 5), nda::range::all), slice_r13);
```

we get

```
HDF5 "ex5.h5" {
DATASET "/B" {
   DATATYPE  H5T_STD_I32LE
   DATASPACE  SIMPLE { ( 5, 5 ) / ( 5, 5 ) }
   DATA {
   (0,0): 0, 1, 2, 3, 4,
   (1,0): 15, 16, 17, 18, 19,
   (2,0): 5, 6, 7, 8, 9,
   (3,0): 20, 21, 22, 23, 24,
   (4,0): 10, 11, 12, 13, 14
   }
}
}
```

@subsection ex5_p4 Reading a slice from an existing dataset

Instead of reading the full dataset as we have done before, it is possible to specify a slice of the dataset that should
be read.

We can reuse the slices from above to first read the rows 0, 2, and 4 and then the rows 1 and 3 of `A` into the first
3 and the last 2 rows of a 5-by-5 array, respectively:

```cpp
// read every other row of "A" into the first 3 rows of C
auto C = nda::zeros<int>(5, 5);
auto C_r012 = C(nda::range(0, 3), nda::range::all);
h5::read(file, "A", C_r012, slice_r024);
std::cout << "C = " << C << std::endl;

// read rows 1 and 3 of "A" into the empty 2 rows of C
auto C_r13 = C(nda::range(3, 5), nda::range::all);
h5::read(file, "A", C_r13, slice_r13);
std::cout << "C = " << C << std::endl;
```

Output:

```
C =
[[0,1,2,3,4]
 [10,11,12,13,14]
 [20,21,22,23,24]
 [0,0,0,0,0]
 [0,0,0,0,0]]
C =
[[0,1,2,3,4]
 [10,11,12,13,14]
 [20,21,22,23,24]
 [5,6,7,8,9]
 [15,16,17,18,19]]
```

@subsection ex5_p5 Writing/Reading 1-dimensional arrays/views of strings

For the user, writing and reading an 1-dimensional array/view of strings works exactly the same way as with an
array/view of arithmetic scalars:

```cpp
// write an array of strings
auto S = nda::array<std::string, 1>{"Hi", "my", "name", "is", "John"};
h5::write(file, "S", S);

// read an array of strings
auto S_r = nda::array<std::string, 1>();
h5::read(file, "S", S_r);
std::cout << "S_r = " << S_r << std::endl;
```

Output:

```
S_r = [Hi,my,name,is,John]
```

The dumped HDF5 dataset gives

```
HDF5 "ex5.h5" {
DATASET "/S" {
   DATATYPE  H5T_STRING {
      STRSIZE 5;
      STRPAD H5T_STR_NULLTERM;
      CSET H5T_CSET_UTF8;
      CTYPE H5T_C_S1;
   }
   DATASPACE  SIMPLE { ( 5 ) / ( 5 ) }
   DATA {
   (0): "Hi", "my", "name", "is", "John"
   }
}
}
```

@subsection ex5_p6 Writing/Reading arrays/views of generic types

**nda** allows us to write/read arbitrary arrays/views as long as the objects contained in the array have specialized
`h5_write` and `h5_read` functions (see [h5 docs](https://triqs.github.io/h5/unstable/group__rw__generic.html)).

For example, an array of integer arrays can be written/read as

```cpp
// write an array of integer arrays
auto I = nda::array<nda::array<int, 1>, 1>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
h5::write(file, "I", I);

// read an array of integer arrays
auto I_r = nda::array<nda::array<int, 1>, 1>();
h5::read(file, "I", I_r);
std::cout << "I_r = " << I_r << std::endl;
```

Output:

```
I_r = [[0,1,2],[3,4,5],[6,7,8]]
```

Dumping the corresponding HDF5 group gives

```
HDF5 "ex5.h5" {
GROUP "/I" {
   DATASET "0" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 3 ) / ( 3 ) }
      DATA {
      (0): 0, 1, 2
      }
   }
   DATASET "1" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 3 ) / ( 3 ) }
      DATA {
      (0): 3, 4, 5
      }
   }
   DATASET "2" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 3 ) / ( 3 ) }
      DATA {
      (0): 6, 7, 8
      }
   }
   DATASET "shape" {
      DATATYPE  H5T_STD_I64LE
      DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
      DATA {
      (0): 3
      }
   }
}
}
```

Now, `I` is an HDF5 group and not a dataset and each object of the array, i.e. each integer array, is written to its
own dataset with a name corresponding to its index in the array.
In this case, "0", "1" and "2".
