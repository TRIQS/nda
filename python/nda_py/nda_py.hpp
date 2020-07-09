#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>

#include <nda/nda.hpp>

#include <cpp2py/numpy_proxy.hpp>
#include <cpp2py/py_converter.hpp>
#include <cpp2py/pyref.hpp>

#include "make_py_capsule.hpp"

namespace nda::python {

  using cpp2py::make_numpy_copy;
  using cpp2py::make_numpy_proxy;
  using cpp2py::npy_type;
  using cpp2py::numpy_proxy;

  // Convert array to numpy_proxy
  template <typename AUR>
  numpy_proxy make_numpy_proxy_from_array(AUR &&a) REQUIRES(is_ndarray_v<std::decay_t<AUR>>) {

    using A          = std::decay_t<AUR>;
    using value_type = typename A::value_type;

    if constexpr (cpp2py::has_npy_type<value_type>) {
      std::vector<long> extents(A::rank), strides(A::rank);

      for (size_t i = 0; i < A::rank; ++i) {
        extents[i] = a.indexmap().lengths()[i];
        strides[i] = a.indexmap().strides()[i] * sizeof(value_type);
      }

      return {A::rank,
              npy_type<std::remove_const_t<value_type>>,
              (void *)a.data_start(),
              std::is_const_v<value_type>,
              std::move(extents),
              std::move(strides),
              make_pycapsule(a.storage())};
    } else {
      array<cpp2py::pyref, A::rank> aobj = map([](value_type &x) {
        if constexpr (is_view_v<A> or std::is_reference_v<AUR>) {
          return cpp2py::py_converter<std::decay_t<value_type>>::c2py(x);
        } else { // nda::array rvalue, be sure to move
          return cpp2py::py_converter<std::decay_t<value_type>>::c2py(std::move(x));
        }
      })(a);
      return make_numpy_proxy_from_array(std::move(aobj));
    }
  }

  // ------------------------------------------

  template <typename T, int R>
  bool is_convertible_to_array(PyObject *obj) {
    if (not PyArray_Check(obj)) return false;
    PyArrayObject *arr = (PyArrayObject *)(obj);
    if constexpr (cpp2py::has_npy_type<T>) {
      if (PyArray_TYPE(arr) != npy_type<T>) return false;
    }
    if (PyArray_TYPE(arr) == NPY_OBJECT && PyArray_SIZE(arr) > 0) {
      auto *ptr = static_cast<PyObject **>(PyArray_DATA(arr));
      if (not cpp2py::py_converter<T>::is_convertible(*ptr, false)) return false;
    }
#ifdef PYTHON_NUMPY_VERSION_LT_17
    int rank = arr->nd;
#else
    int rank = PyArray_NDIM(arr);
#endif
    return (rank == R);
  }

  template <typename T, int R>
  bool is_convertible_to_array_view(PyObject *obj) {
    return cpp2py::has_npy_type<T> && is_convertible_to_array<T, R>(obj);
  }

  // ------------------------------------------

  // Make a new array_view from numpy view
  template <typename T, int R>
  array_view<T, R> make_array_view_from_numpy_proxy(numpy_proxy const &v) {
    EXPECTS(v.extents.size() == R);
    EXPECTS((std::is_same_v<T, cpp2py::pyref> or v.element_type != NPY_OBJECT));

    std::array<long, R> extents, strides;
    for (int u = 0; u < R; ++u) {
      extents[u] = v.extents[u];
      strides[u] = v.strides[u] / sizeof(T);
    }
    using layout_t = typename array_view<T, R>::layout_t;

    return array_view<T, R>{layout_t{extents, strides}, static_cast<T *>(v.data)};
  }

  // ------------------------------------------

  // Make a new array from numpy view
  template <typename T, int R>
  array<T, R> make_array_from_numpy_proxy(numpy_proxy const &v) {
    EXPECTS(v.extents.size() == R);

    std::array<long, R> extents, strides;
    for (int u = 0; u < R; ++u) {
      extents[u] = v.extents[u];
      strides[u] = v.strides[u] / sizeof(T);
    }

    auto layout = typename array_view<T, R>::layout_t{extents, strides};

    if (v.element_type == npy_type<cpp2py::pyref>) {
      auto arr_dat = array_view<PyObject *, R>{layout, static_cast<PyObject **>(v.data)};
      return map([](PyObject *o) { return cpp2py::py_converter<std::decay_t<T>>::py2c(o); })(arr_dat);
    } else {
      return array_view<T, R>{layout, static_cast<T *>(v.data)};
    }
  }

} // namespace nda::python
