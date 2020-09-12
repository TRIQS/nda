#define NDA_ENFORCE_BOUNDCHECK

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include "numpy_proxy.hpp"

namespace pybind11::detail {

  // -----------------------------------
  // array_view
  // -----------------------------------

  template <typename T, int R, char Algebra>
  struct type_caster<nda::basic_array_view<T, R, nda::C_stride_layout, Algebra>> {
    using _type = nda::basic_array_view<T, R, nda::C_stride_layout, Algebra>;

    PYBIND11_TYPE_CASTER(_type, _("nda::array_view"));

    using U = std::decay_t<T>;
    static_assert(nda::python::has_npy_type<U>, "Logical Error");
    static_assert(not std::is_same_v<U, pybind11::object>, "Not implemented"); // would require to take care of the incref...
    // However, it works for PyObject *

    // --------- C->Py  --------

    static handle cast(nda::array_view<T, R> src, return_value_policy /* policy */, handle /* parent */) {
      _import_array();
      auto p = nda::python::make_numpy_proxy_from_array_or_view(src);
      return p.to_python();
    }

    // --------- PY -> C is possible --------

    static bool is_convertible(PyObject *obj, bool raise_python_exception, bool allow_lower_rank = false) {
      // has_npy_type<T> is true (static_assert at top)
      _import_array();
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
        if (nda::python::has_npy_type<U> && (PyArray_TYPE(arr) != nda::python::npy_type<U>)) {
          if (raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array_view : Type mismatch");
          return false;
        }
      }

      return true;
    }

    // --------- PY -> C --------

    static nda::array_view<T, R> py2c(PyObject *obj) {
      _import_array();
      auto p = nda::python::make_numpy_proxy(obj);
      EXPECTS(p.extents.size() >= R);
      EXPECTS(p.element_type == nda::python::npy_type<T> or p.extents.size() > R);

      std::array<long, R> extents, strides;
      for (int u = 0; u < R; ++u) {
        extents[u] = p.extents[u];
        strides[u] = p.strides[u] / sizeof(T);
      }
      return nda::array_view<T, R>{{extents, strides}, static_cast<T *>(p.data)};
    }

    // --------- PY -> C  --------

    bool load(handle src, bool) {
      PyObject *source = src.ptr();
      if (not is_convertible(source, false, false)) return false;
      value.rebind(py2c(source));
      if (PyErr_Occurred()) PyErr_Print();
      return true;
    }
  };

  // **********************************
  // array
  // **********************************

  template <typename T, int R, char Algebra>
  struct type_caster<nda::basic_array<T, R, nda::C_layout, Algebra, nda::heap>> {

    // T can be a npy type cpp2py::has_npy_type<T> == true or NOT (then we need to convert using cpp2py)
    static_assert(not std::is_same_v<T, pybind11::object>, "Not implemented");
    static_assert(not std::is_same_v<T, PyObject *>, "Not implemented");

    using converter_T             = type_caster<std::decay_t<T>>;
    using converter_view_T        = type_caster<nda::array_view<T, R>>;
    using converter_view_pyobject = type_caster<nda::array_view<PyObject *, R>>;

    // --------- C -> PY --------

    template <typename A>
    static PyObject *c2py(A &&src) {
      static_assert(std::is_same_v<std::decay_t<A>, nda::basic_array<T, R, nda::C_layout, Algebra, nda::heap>>,
                    "Logic Error in array c2py conversion");
      _import_array();
      auto p = nda::python::make_numpy_proxy_from_array_or_view(std::forward<A>(src));
      return p.to_python();
    }

    using _type = nda::basic_array<T, R, nda::C_layout, Algebra, nda::heap>;
    PYBIND11_TYPE_CASTER(_type, _("nda::array"));

    // FIXME RECHCEK
    template <typename A>
    static handle cast(A &&src, return_value_policy /* policy */, handle /* parent */) {
      //static handle cast(nda::array<T, R> src, return_value_policy /* policy */, handle /* parent */) {
      static_assert(std::is_same_v<std::decay_t<A>, nda::array<T, R>>, "Logic Error in array c2py conversion");
      _import_array();
      PRINT(" IN CAsT");

      auto p = nda::python::make_numpy_proxy_from_array_or_view(std::forward<A>(src));
      return p.to_python();
    }

    // --------- PY -> C is possible --------

    static PyObject *make_numpy(PyObject *obj) {
      return PyArray_FromAny(obj, PyArray_DescrFromType(nda::python::npy_type<T>), R, R, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY,
                             NULL); // new ref
    }

    static bool is_convertible(PyObject *obj, bool raise_python_exception) {
      // if obj is not an numpy, we try to make a numpy with the proper type
      if (not PyArray_Check(obj) or (PyArray_Check(obj) and nda::python::has_npy_type<T> and (PyArray_TYPE((PyArrayObject *)(obj)) != nda::python::npy_type<T>))) {
        auto numpy_obj = pybind11::reinterpret_steal<pybind11::object>(make_numpy(obj));
        if (PyErr_Occurred()) {
          if (!raise_python_exception) PyErr_Clear();
          return false;
        }
        return is_convertible(numpy_obj.ptr(), raise_python_exception);
      }

      if constexpr (nda::python::has_npy_type<T>) {
        // in this case, we convert to a view and then copy to the array, so the condition is the same
        return converter_view_T::is_convertible(obj, raise_python_exception);
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
        auto p = nda::python::make_numpy_proxy(obj);
        std::array<long, R> shape;
        for (int i = 0; i < R; ++i) shape[i] = p.extents[i];
        auto l = [obj](auto... i) -> bool {
          auto tupl   = pybind11::make_tuple(pybind11::reinterpret_steal<pybind11::object>(PyLong_FromLong(i))...);
          auto subobj = pybind11::reinterpret_steal<pybind11::object>(PyObject_GetItem(obj, tupl.ptr())); // PyObject_GetItem returns a new ref.
          converter_T c;
          bool b = c.load(subobj, true);
          return b;
          //return converter_T::is_convertible(subobj, false);
        };
        res = sum(nda::array_adapter{shape, l});

        if (!res and raise_python_exception) PyErr_SetString(PyExc_TypeError, "Cannot convert to array. One element can not be converted to C++.");
        return res;
      }
    }

    // --------- PY -> C --------

    static nda::array<T, R> py2c(PyObject *obj) {
      _import_array();

      // if obj is not an numpy, we make a numpy and rerun
      if (not PyArray_Check(obj) or (PyArray_Check(obj) and nda::python::has_npy_type<T> and (PyArray_TYPE((PyArrayObject *)(obj)) != nda::python::npy_type<T>))) {
        auto numpy_obj = pybind11::reinterpret_steal<pybind11::object>(make_numpy(obj));
        //cpp2py::pyref numpy_obj = make_numpy(obj);
        EXPECTS(not PyErr_Occurred());
        return py2c(numpy_obj.ptr());
      }

      if constexpr (nda::python::has_npy_type<T>) {
        return converter_view_T::py2c(obj);
      } else {
        auto p = nda::python::make_numpy_proxy(obj);
        std::array<long, R> shape;
        for (int i = 0; i < R; ++i) shape[i] = p.extents[i];
        auto l = [obj](auto... i) -> T {
          auto tupl   = pybind11::make_tuple(pybind11::reinterpret_steal<pybind11::object>(PyLong_FromLong(i))...);
          auto subobj = pybind11::reinterpret_steal<pybind11::object>(PyObject_GetItem(obj, tupl.ptr())); // PyObject_GetItem returns a new ref.
          //pyref subobj = PyObject_GetItem(obj, pyref::make_tuple(PyLong_FromLong(i)...));
          //converter_T c;
          //return converter_T::py2c(subobj.ptr());
          return subobj.template cast<T>();
        };
        nda::array<T, R> res = nda::array_adapter{shape, l};
        if (PyErr_Occurred()) PyErr_Print();
        return res;
      }
    }

    bool load(handle src, bool) {
      PyObject *source = src.ptr();
      if (not is_convertible(source, false)) return false;
      value = py2c(source);
      if (PyErr_Occurred()) PyErr_Print();
      return true;
    }
  };
} // namespace pybind11::detail
