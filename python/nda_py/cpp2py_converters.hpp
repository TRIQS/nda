#pragma once

#include <cpp2py/py_converter.hpp>
#include <nda_py/nda_py.hpp>

namespace cpp2py {

  // -----------------------------------
  // array_view
  // -----------------------------------

  template <typename T, int R>
  struct py_converter<nda::array_view<T, R>> {
    using _type = nda::array_view<T, R>;

    static _type py2c(PyObject *src) REQUIRES(cpp2py::has_npy_type<T>) {
      _import_array();
      nda::python::numpy_proxy p = nda::python::make_numpy_proxy(src);
      auto res                   = nda::python::make_array_view_from_numpy_proxy<T, R>(p);
      if (PyErr_Occurred()) PyErr_Print();
      return res;
    }

    static PyObject *c2py(_type src) {
      _import_array();
      nda::python::numpy_proxy p = nda::python::make_numpy_proxy_from_array(src);
      return p.to_python();
    }

    static bool is_convertible(PyObject *src, bool raise_exception) {
      //import_numpy();
      auto res = nda::python::is_convertible_to_array_view<T, R>(src);
      if (!res and raise_exception) throw std::runtime_error("Cannot convert to array/matrix");
      return res;
    }
  };

  // -----------------------------------
  // array
  // -----------------------------------

  template <typename T, int R>
  struct py_converter<nda::array<T, R>> {

    static nda::array<T, R> py2c(PyObject *src) {
      _import_array();
      nda::python::numpy_proxy p = nda::python::make_numpy_proxy(src);
      if constexpr (cpp2py::has_npy_type<T>) {
        auto res = nda::python::make_array_view_from_numpy_proxy<T, R>(p);
        if (PyErr_Occurred()) PyErr_Print();
        return res;
      } else {
        auto res = nda::python::make_array_from_numpy_proxy<T, R>(p);
        if (PyErr_Occurred()) PyErr_Print();
        return res;
      }
    }

    static PyObject *c2py(nda::array<T, R> src) {
      _import_array();
      nda::python::numpy_proxy p = nda::python::make_numpy_proxy_from_array(std::move(src));
      return p.to_python();
    }

    static bool is_convertible(PyObject *src, bool raise_exception) {
      auto res = nda::python::is_convertible_to_array<T, R>(src);
      if (!res and raise_exception) throw std::runtime_error("Cannot convert to array/matrix");
      return res;
    }
  };

  //// -----------------------------------
  //// range
  //// -----------------------------------

  //// range can not be directly converted from slice (slice is more complex)
  //// convert from python slice and int (interpreted are slice(i,i+1,1))
  //itertools::range range_from_slice(PyObject *src, long len) {
    //if (PyInt_Check(src)) {
      //long i = PyInt_AsLong(src);
      //if ((i < -len) || (i >= len)) CPP2PY_RUNTIME_ERROR << "Integer index out of range : expected [0," << len << "], got " << i;
      //if (i < 0) i += len;
      //// std::cerr  << " range int "<< i << std::endl;
      //return {i, i + 1, 1};
    //}
    //Py_ssize_t start, stop, step, slicelength;
    //if (!PySlice_Check(src) || (PySlice_GetIndicesEx((PySliceObject *)src, len, &start, &stop, &step, &slicelength) < 0))
      //CPP2PY_RUNTIME_ERROR << "Can not converted the slice to C++";
    //// std::cerr  << "range ( "<< start << " "<< stop << " " << step<<std::endl;
    //return {start, stop, step};
  //}

  //template <>
  //struct py_converter<itertools::range> {
    //static PyObject *c2py(itertools::range const &r) {
      //return PySlice_New(convert_to_python(r.first()), convert_to_python(r.last()), convert_to_python(r.step()));
    //}
    //static itertools::range py2c(PyObject *src)                     = delete;
    //static bool is_convertible(PyObject *src, bool raise_exception) = delete;
  //};
} // namespace cpp2py
