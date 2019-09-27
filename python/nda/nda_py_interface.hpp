#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>
#include <nda/nda.hpp>

namespace nda::python {

  using v_t = std::vector<long>;

  // the basic information for a numpy array
  struct view_info {
    int rank          = 0;
    long element_type = 0;
    void *data        = nullptr;
    bool is_const     = false;
    v_t extents, strides;
    PyObject *base = nullptr; // The ref. coounting guard
  };

  // intermediate information on a numpyarray : just rank and element type
  struct numpy_rank_and_type {
    PyArrayObject *arr = nullptr;
    long element_type  = 0;
    int rank           = 0;
  };

  // C to Python
  // If failure, return null with the Python exception set
  PyObject *to_python(view_info &a); // it is const & except that C interface will not like the const ...

  view_info from_python(PyObject *obj, bool enforce_copy);

  // Python to C : step 1 : get rank and type
  numpy_rank_and_type get_numpy_rank_and_type(PyObject *obj);

  // Python to C : step 2 : get the dims, strides
  view_info from_python(numpy_rank_and_type info);

  // Make a copy in Python with the given rank and element_type
  // If failure, return null with the Python exception set
  PyObject *make_numpy_copy(PyObject *obj, int rank, long elements_type);

  //
  template <typename T>
  long npy_type;

#define CONVERT(C, P)                                                                                                                                \
  template <>                                                                                                                                        \
  long npy_type<C> = P;
  CONVERT(bool, NPY_BOOL);
  CONVERT(char, NPY_STRING);
  CONVERT(signed char, NPY_BYTE);
  CONVERT(unsigned char, NPY_UBYTE);
  CONVERT(short, NPY_SHORT);
  CONVERT(unsigned short, NPY_USHORT);
  CONVERT(int, NPY_INT);
  CONVERT(unsigned int, NPY_UINT);
  CONVERT(long, NPY_LONG);
  CONVERT(unsigned long, NPY_ULONG);
  CONVERT(long long, NPY_LONGLONG);
  CONVERT(unsigned long long, NPY_ULONGLONG);
  CONVERT(float, NPY_FLOAT);
  CONVERT(double, NPY_DOUBLE);
  CONVERT(long double, NPY_LONGDOUBLE);
  CONVERT(std::complex<float>, NPY_CFLOAT);
  CONVERT(std::complex<double>, NPY_CDOUBLE);
  CONVERT(std::complex<long double>, NPY_CLONGDOUBLE);
#undef CONVERT

  // Convert array to view_info
  template <typename A>
  view_info make_view_info_from_array(A &a) {

    int rank = A::rank;
    v_t extents(rank), strides(rank);

    for (size_t i = 0; i < rank; ++i) {
      extents[i] = a.indexmap().lengths()[i];
      strides[i] = a.indexmap().strides()[i] * sizeof(typename A::value_type);
    }

    return {rank,
            npy_type<typename A::value_type>,
            (void *)a.data_start(),
            std::is_const_v<A>,
            std::move(extents),
            std::move(strides),
            make_pycapsule(a.storage())};
  }

  // Make a new array_view from numpy view
  template <typename T, int R>
  array_view<T, R> make_array_view_from_view_info(view_info const &v) {
    EXPECTS(v.rank == R);
    std::array<long, R> extents, strides;
    for (int u = 0; u < R; ++u) {
      extents[u] = v.extents[u];
      strides[u] = v.strides[u] / sizeof(T);
    }
    using idx_t = typename array_view<T, R>::idx_map_t;
    return array_view<T, R>{idx_t{extents, strides}, static_cast<T *>(v.data)};
  }

} // namespace nda::python
