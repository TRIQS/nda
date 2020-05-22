#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>

#include <nda/nda.hpp>

#include <cpp2py/numpy_proxy.hpp>

#include "make_py_capsule.hpp"

namespace nda::python {

  using cpp2py::make_numpy_copy;
  using cpp2py::make_numpy_proxy;
  using cpp2py::numpy_proxy;

  // Convert array to numpy_proxy
  template <typename A>
  numpy_proxy make_numpy_proxy_from_array(A &a) {

    v_t extents(A::rank), strides(A::rank);

    for (size_t i = 0; i < A::rank; ++i) {
      extents[i] = a.indexmap().lengths()[i];
      strides[i] = a.indexmap().strides()[i] * sizeof(typename A::value_type);
    }

    return {A::rank,
            npy_type<std::remove_const_t<typename A::value_type>>,
            (void *)a.data_start(),
            std::is_const_v<typename A::value_type>,
            std::move(extents),
            std::move(strides),
            make_pycapsule(a.storage())};
  }

  // ------------------------------------------

  template <typename T, int R>
  bool is_convertible_to_array_view(PyObject *obj) {
    if (not PyArray_Check(obj)) return false;
    PyArrayObject *arr = (PyArrayObject *)(obj);
    if (PyArray_TYPE(arr) != npy_type<T>) return false;
#ifdef PYTHON_NUMPY_VERSION_LT_17
    int rank = arr->nd;
#else
    int rank = PyArray_NDIM(arr);
#endif
    return (rank == R);
  }

  // ------------------------------------------

  // Make a new array_view from numpy view
  template <typename T, int R>
  array_view<T, R> make_array_view_from_numpy_proxy(numpy_proxy const &v) {
    std::array<long, R> extents, strides;
    for (int u = 0; u < R; ++u) {
      extents[u] = v.extents[u];
      strides[u] = v.strides[u] / sizeof(T);
    }
    using idx_t = typename array_view<T, R>::layout_t;
    return array_view<T, R>{idx_t{extents, strides}, static_cast<T *>(v.data)};
  }

} // namespace nda::python
