#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>
#include <nda/nda.hpp>

namespace nda::python {

  using v_t = std::vector<long>;

  // the basic information for a numpy array
  struct numpy_proxy {
    int rank          = 0;
    long element_type = 0;
    void *data        = nullptr;
    bool is_const     = false;
    v_t extents, strides;
    PyObject *base = nullptr; // The ref. counting guard typically

    // Returns a new ref (or NULL if failure) with a new numpy.
    // If failure, return null with the Python exception set
    PyObject *to_python();
  };

  // From a numpy, extract the info. Better than a constructor, I want to use the aggregate constructor of the struct also.
  numpy_proxy make_numpy_copy(PyObject *);

  // Make a copy in Python with the given rank and element_type
  // If failure, return null with the Python exception set
  PyObject *make_numpy_copy(PyObject *obj, int rank, long elements_type);

  //
  template <typename T>
  inline long npy_type;

#define CONVERT(C, P)                                                                                                                                \
  template <>                                                                                                                                        \
  inline long npy_type<C> = P;
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

  // ------------------------------------------

  // Convert array to numpy_proxy
  template <typename A>
  numpy_proxy make_numpy_proxy_from_array(A &a) {

    v_t extents(A::rank), strides(A::rank);

    for (size_t i = 0; i < A::rank; ++i) {
      extents[i] = a.indexmap().lengths()[i];
      strides[i] = a.indexmap().strides()[i] * sizeof(typename A::value_type);
    }

    return {A::rank,
            npy_type<typename A::value_type>,
            (void *)a.data_start(),
            std::is_const_v<A>,
            std::move(extents),
            std::move(strides),
            make_pycapsule(a.storage())};
  }

  // ------------------------------------------

  template <typename T, int R>
  bool is_convertible_to_array_view(PyObject *obj) {
    NDA_PRINT("OK");
    if (not PyArray_Check(obj)) return false;
    NDA_PRINT("OK");
    PyArrayObject *arr = (PyArrayObject *)(obj);
    NDA_PRINT("OK");
    if (PyArray_TYPE(arr) != npy_type<T>) return false;
    NDA_PRINT("OK");
#ifdef PYTHON_NUMPY_VERSION_LT_17
    int rank = arr->nd;
#else
    int rank = PyArray_NDIM(arr);
#endif
    NDA_PRINT((rank == R));
    return (rank == R);
  }

  // ------------------------------------------

  // Make a new array_view from numpy view
  template <typename T, int R>
  array_view<T, R> make_array_view_from_numpy_proxy(numpy_proxy const &v) {
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
