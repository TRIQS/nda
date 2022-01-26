#pragma once

#include <cpp2py/py_converter.hpp>
#include <nda_py/nda_py.hpp>

namespace cpp2py {

  template <typename T, int R, typename L, char A, typename AP, typename OP>
  struct is_view<nda::basic_array_view<T, R, L, A, AP, OP>> : std::true_type {};

  // -----------------------------------
  // array_view
  // -----------------------------------

  template <int R, typename Layout>
  bool numpy_check_layout(PyObject * obj) {
    EXPECTS(PyArray_Check(obj));
    PyArrayObject *arr = (PyArrayObject *)(obj);
    return Layout::template mapping<R>::is_stride_order_valid(PyArray_DIMS(arr), PyArray_STRIDES(arr));
  }

  template <typename T, int R, typename Layout, char Algebra>
  struct py_converter<nda::basic_array_view<T, R, Layout, Algebra>> {

    using U = std::decay_t<T>;
    static_assert(has_npy_type<U>, "Logical Error");
    static_assert(not std::is_same_v<U, pyref>, "Not implemented"); // would require to take care of the incref...
    // However, it works for PyObject *

    // --------- C -> PY --------

    static PyObject *c2py(nda::array_view<T, R> v) {
      auto p = nda::python::make_numpy_proxy_from_array_or_view(v);
      return p.to_python();
    }

    // --------- PY -> C is possible --------

    static bool is_convertible(PyObject *obj, bool raise_python_exception, bool allow_lower_rank = false, bool require_c_order = true) {
      // has_npy_type<T> is true (static_assert at top)
      // check the rank and type. First protects the rest
      if (not PyArray_Check(obj)) {
        if (raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array_view : Python object is not a Numpy array");
        return false;
      }
      PyArrayObject *arr = (PyArrayObject *)(obj);

      auto r = PyArray_NDIM(arr);
      if (allow_lower_rank ? r < R : r != R) {
        if (raise_python_exception)
          PyErr_SetString(
             PyExc_TypeError,
             ("Cannot convert to array_view : Rank is not correct. Expected " + std::to_string(R) + "\n Got " + std::to_string(PyArray_NDIM(arr)))
                .c_str());
        return false;
      }

      if (not allow_lower_rank && r == R) {
        if (has_npy_type<U> && (PyArray_TYPE(arr) != npy_type<U>)) {
          if (raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array_view : Type mismatch");
          return false;
        }
      }

      if (require_c_order and not numpy_check_layout<R, Layout>(obj)) {
        if (raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array_view : Numpy array is not in C order");
        return false;
      }

      return true;
    }

    // --------- PY -> C --------

    static nda::array_view<T, R> py2c(PyObject *obj) {
      auto p = make_numpy_proxy(obj);
      EXPECTS(p.extents.size() >= R);
      EXPECTS(p.element_type == npy_type<T> or p.extents.size() > R);
      EXPECTS((numpy_check_layout<R, Layout>(obj)));

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
  template <typename T, int R, char Algebra>
  struct py_converter<nda::basic_array<T, R, nda::C_layout, Algebra, nda::heap<>>> {

    // T can be a npy type cpp2py::has_npy_type<T> == true or NOT (then we need to convert using cpp2py)
    static_assert(not std::is_same_v<T, pyref>, "Not implemented");
    static_assert(not std::is_same_v<T, PyObject *>, "Not implemented");

    using converter_T             = py_converter<std::decay_t<T>>;
    using converter_view_T        = py_converter<nda::array_view<T, R>>;
    using converter_view_pyobject = py_converter<nda::array_view<PyObject *, R>>;

    // --------- C -> PY --------

    template <typename A>
    static PyObject *c2py(A &&src) {
      static_assert(std::is_same_v<std::decay_t<A>, nda::basic_array<T, R, nda::C_layout, Algebra, nda::heap<>>>,
                    "Logic Error in array c2py conversion");
      auto p = nda::python::make_numpy_proxy_from_array_or_view(std::forward<A>(src));
      return p.to_python();
    }

    // --------- PY -> C is possible --------

    static PyObject *make_numpy(PyObject *obj) {
      return PyArray_FromAny(obj, PyArray_DescrFromType(npy_type<T>), R, R, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY, NULL); // new ref
    }

    static bool is_convertible(PyObject *obj, bool raise_python_exception) {
      // if obj is not an numpy, we try to make a numpy with the proper type
      if (not PyArray_Check(obj) or (PyArray_Check(obj) and has_npy_type<T> and (PyArray_TYPE((PyArrayObject *)(obj)) != npy_type<T>))) {
        cpp2py::pyref numpy_obj = make_numpy(obj);
        if (PyErr_Occurred()) {
          if (!raise_python_exception) PyErr_Clear();
          return false;
        }
        return is_convertible(numpy_obj, raise_python_exception);
      }

      if constexpr (has_npy_type<T>) {
        // in this case, we convert to a view and then copy to the array, so the condition is the same
        return converter_view_T::is_convertible(obj, raise_python_exception, false /*allow_lower_rank*/, false /*require_c_order*/);
      } else {
        // T is a type requiring conversion.
        // First I see whether I can convert it to a array of PyObject* ( i.e. it is an array of object and it has the proper rank...)
        bool res = converter_view_pyobject::is_convertible(obj, raise_python_exception, true /*allow_lower_rank*/);
        if (not res) return false;

        // Check if all elements are convertible
        // CAUTION: numpy array might be of higher rank!
        // We Extract the first R extents from the numpy proxy
        // and use __getitem__ to check convertibility of
        // each such element
        auto p = make_numpy_proxy(obj);
        std::array<long, R> shape;
        for (int i = 0; i < R; ++i) shape[i] = p.extents[i];
        auto l = [obj](auto... i) -> bool {
          pyref subobj = PyObject_GetItem(obj, pyref::make_tuple(PyLong_FromLong(i)...));
          return converter_T::is_convertible(subobj, false);
        };
        res = sum(nda::array_adapter{shape, l});

        if (!res and raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array. One element can not be converted to C++.");
        return res;
      }
    }

    // --------- PY -> C --------

    static nda::array<T, R> py2c(PyObject *obj) {

      // if obj is not an numpy, we make a numpy and rerun
      if (not PyArray_Check(obj) or (PyArray_Check(obj) and has_npy_type<T> and (PyArray_TYPE((PyArrayObject *)(obj)) != npy_type<T>))) {

        cpp2py::pyref numpy_obj = make_numpy(obj);
        EXPECTS(not PyErr_Occurred());
        return py2c(numpy_obj);
      }

      if constexpr (has_npy_type<T>) {
        if (not numpy_check_layout<R, nda::C_layout>(obj)) {
          cpp2py::pyref obj_c_order = make_numpy(obj);
          return nda::array<T, R>{converter_view_T::py2c(obj_c_order)};
        }
        return converter_view_T::py2c(obj);
      } else {
        auto p = make_numpy_proxy(obj);
        std::array<long, R> shape;
        for (int i = 0; i < R; ++i) shape[i] = p.extents[i];
        auto l = [obj](auto... i) {
          pyref subobj = PyObject_GetItem(obj, pyref::make_tuple(PyLong_FromLong(i)...));
          return converter_T::py2c(subobj);
        };
        nda::array<T, R> res = nda::array_adapter{shape, l};
        if (PyErr_Occurred()) PyErr_Print();
        return res;
      }
    }
  };

} // namespace cpp2py
