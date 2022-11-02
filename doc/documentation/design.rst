Let us briefly outline the general design principles of the fundamental objects in nda.

Arrays
******

The generic :code:`nda::array` class can store both built-in datatype such as :code:`int` and :code:`double`,
but also objects of custom types such as classes and structs. At construction of the array
a memory-block of sufficient size will be allocated, which is then automatically freed when the
array object is destroyed. For this purpose the :code:`nda::array` class contains a handle on the
block of memory, typically of type :code:`nda::mem::handle_heap`, which contains the memory pointer :code:`T* p`.
Additionally the array stores an :code:`nda::idx_map` which provides the information on how to index into this block of memory.
This is provided through a mapping between the array indices to an pointer offset :code:`(i,j,k,l) -> offset`,
such that the associated array element is located in memory at :code:`p + offset`.
For array objects this mapping is always contiguous.


Views
*****

The :code:`nda::array_view` type provides an interface very similar to that of the :code:`nda::array`,
it does however not take care of managing memory. In other words, an object of
type :code:`nda::array_view` can only access into the memory of another object, e.g. of
type :code:`nda::array`. If the underlying object is destroyed, we say that the view gets invalidated.
This is best illustrated with an example

.. code-block:: c

   nda::array_view<int, 1> bad_array_factory() {
     auto A = nda::array<int, 1>{1,2,3};
     return A;
   } // <- A is destroyed here!! A view to it does not make sense

Differently from array objects, views sometimes access into memory non-contiguously.
Let us give a simple example for this

.. code-block:: c

   auto A = nda::array<int, 1>{1,2,3,4};
   auto B = A(nda::range(0,4,2));
   // The view B can only access elements 1 and 3
   // Values 2 and 4 can't be reached!
   B[0]; // -> 1
   B[1]; // -> 3


Lazy Expressions
****************

NDA can map operations on all values of an array or view.
An example for this is :code:`nda::conj`, that performs complex conjugation on all elements
of a complex array. Note however that the operation is not immediately executed,
but is delayed until we use the new values. We say that the operation is performed lazily.
This is best demonstrated with an example

.. code-block:: c

   using dcplx = std::complex<double>;
   auto C = nda::array<dcplx, 1>{{1, 2}, {4, 1}}; // An array of complex values

   auto D = nda::conj(C); // <- No work is done here! We only store C by reference and the operation
   D[0]; // -> 1 - 2i

We call :code:`D` a lazy-expression. Similar to views, lazy-expressions get invalidated
if the object they operate on is destroyed. The following example demonstrates this

.. code-block:: c

   auto bad_conjugator() {
     using dcplx = std::complex<double>;
     auto C = nda::array<dcplx, 1>{{1, 2}, {4, 1}};
     return nda::conj(C);
   } // <- C is destroyed here!! The returned expression is invalid

We should be returning an array here :code:`return nda::basic_array{nda::conj(C)};`.
For these use-cases we can also use the :code:`nda::make_regular` function to convert an
array-like or view object into an array.
