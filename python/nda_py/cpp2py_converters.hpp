#pragma once

#include <cpp2py/py_converter.hpp>
#include <nda_py/nda_py.hpp>

namespace cpp2py {

  // -----------------------------------
  // array_view
  // -----------------------------------

  template <typename T, int R>
  struct py_converter<nda::array_view<T, R>> {

    using U = std::decay_t<T>;
    static_assert(has_npy_type<U>, "Logical Error");
    static_assert(not std::is_same_v<U, pyref>, "Not implemented"); // would require to take care of the incref...
    // However, it works for PyObject *

    // --------- C -> PY --------

    static PyObject *c2py(nda::array_view<T, R> v) {
      _import_array();
      auto p = nda::python::make_numpy_proxy_from_array_or_view(v);
      return p.to_python();
    }

    // --------- PY -> C is possible --------

    static bool is_convertible(PyObject *obj, bool raise_python_exception) {
      // has_npy_type<T> is true (static_assert at top)
      _import_array();
      // check the rank and type. First protects the rest
      if (not PyArray_Check(obj)) {
        if (raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array_view : Python object is not a Numpy array");
        return false;
      }
      PyArrayObject *arr = (PyArrayObject *)(obj);
      
      if (PyArray_NDIM(arr) != R) {
        if (raise_python_exception)
          PyErr_SetString(
             PyExc_TypeError,
             ("Cannot convert to array_view : Rank is not correct. Expected " + std::to_string(R) + "\n Got " + std::to_string(PyArray_NDIM(arr))).c_str());
        return false;
      }

      if (PyArray_TYPE(arr) == npy_type<U>) {
        if (raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array_view : Type mismatch");
        return false;
      }

      return true;
    }

    // --------- PY -> C --------

    static nda::array_view<T, R> py2c(PyObject *obj) {
      _import_array();
      auto p = make_numpy_proxy(obj);
      EXPECTS(p.extents.size() == R);
      EXPECTS(p.element_type == npy_type<T>);

      std::array<long, R> extents, strides;
      for (int u = 0; u < R; ++u) {
        extents[u] = p.extents[u];
        strides[u] = p.strides[u] / sizeof(T);
      }
      return nda::array_view<T, R>{{extents, strides}, static_cast<T *>(p.data)};
    }
  }; 

  // -----------------------------------
  // array
  // -----------------------------------

  template <typename T, int R>
  struct py_converter<nda::array<T, R>> {

    // T can be a npy type cpp2py::has_npy_type<T> == true or NOT (then we need to convert using cpp2py)
    static_assert(not std::is_same_v<T, pyref>, "Not implemented");
    static_assert(not std::is_same_v<T, PyObject *>, "Not implemented");

    using converter_T             = py_converter<std::decay_t<T>>;
    using converter_view_T        = py_converter<nda::array_view<T, R>>;
    using converter_view_pyobject = py_converter<nda::array_view<PyObject *, R>>;

    // --------- C -> PY --------

    template <typename A>
    static PyObject *c2py(A &&src) {
      static_assert(std::is_same_v<std::decay_t<A>, nda::array<T, R>>, "Logic Error in array c2py conversion");
      _import_array();
      auto p = nda::python::make_numpy_proxy_from_array_or_view(std::forward<A>(src));
      return p.to_python();
    }

    // --------- PY -> C is possible --------

    static bool is_convertible(PyObject *obj, bool raise_python_exception) {
      if constexpr (has_npy_type<T>) {
        // in this case, we convert to a view and then copy to the array, so the condition is the same
        return converter_view_T::is_convertible(obj, raise_python_exception);
      } else {
        // T is a type requiring conversion.
        // First I see whether I can convert it to a array of PyObject* ( i.e. it is an array of object and it has the proper rank...)
        bool res = converter_view_pyobject::is_convertible(obj, raise_python_exception);
        if (not res) return false;
#if 1
        // ok, now I need to see if all elements are convertible ...
        // I make the array of PyObject*, and use all_of
        nda::array_view<PyObject *, R> v = converter_view_pyobject::py2c(obj);
        res                              = std::all_of(v.begin(), v.end(), [](PyObject *ob) { return converter_T::is_convertible(ob, false); });
#else
	// FIXME : Shall I keep this ??
        // Shorter version where I check only the first ??
        PyArrayObject *arr = (PyArrayObject *)(obj);
        if (PyArray_TYPE(arr) == NPY_OBJECT && PyArray_SIZE(arr) > 0) {
          auto *ptr = static_cast<PyObject **>(PyArray_DATA(arr));
          if (not py_converter<T>::is_convertible(*ptr, false)) return false;
        }
#endif
        if (!res and raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array. One element can not be converted to C++.");
        return res;
      }
    }

    // --------- PY -> C --------

    static nda::array<T, R> py2c(PyObject *obj) {
      _import_array();
      if constexpr (has_npy_type<T>) {
        return converter_view_T::py2c(obj);
      } else {
        auto arr             = converter_view_pyobject::py2c(obj);
        nda::array<T, R> res = map([](PyObject *ob) { return converter_T::py2c(ob); })(arr);
        if (PyErr_Occurred()) PyErr_Print();
        return res;
      }
    }
  };

 } // namespace cpp2py
