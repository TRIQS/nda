#include "./nda_py_interface.hpp"

namespace nda::python {

  // Make a new view_info
  PyObject *to_python(view_info &v) {

    //_import_array();

#ifdef PYTHON_NUMPY_VERSION_LT_17
    int flags = NPY_BEHAVED & ~NPY_OWNDATA;
#else
    int flags = NPY_ARRAY_BEHAVED & ~NPY_ARRAY_OWNDATA;
#endif
    PyObject *result =
       PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(v.element_type), v.rank, v.extents.data(), v.strides.data(), v.data, flags, NULL);
    if (not result) return nullptr; // the Python error is set

    if (!PyArray_Check(result)) {
      PyErr_SetString(PyExc_RuntimeError, "The python object is not a numpy array");
      return nullptr;
    }

    PyArrayObject *arr = (PyArrayObject *)(result);
#ifdef PYTHON_NUMPY_VERSION_LT_17
    arr->base = v.base;
    assert(arr->flags == (arr->flags & ~NPY_OWNDATA));
#else
    int r     = PyArray_SetBaseObject(arr, v.base);
    EXPECTS(r == 0);
    EXPECTS(PyArray_FLAGS(arr) == (PyArray_FLAGS(arr) & ~NPY_ARRAY_OWNDATA));
#endif
    return result;
  }

  // ----------------------------------------------------------

  // Extract a view_info from python
  numpy_rank_and_type get_numpy_rank_and_type(PyObject *obj) {

    if (obj == NULL) return {};
    if (_import_array() != 0) return {};
    if (not PyArray_Check(obj)) return {};

    // extract strides and lengths
    PyArrayObject *arr = (PyArrayObject *)(obj);

#ifdef PYTHON_NUMPY_VERSION_LT_17
    int rank = arr->nd;
#else
    int rank = PyArray_NDIM(arr);
#endif
    return {arr, PyArray_TYPE(arr), rank};
  }

  // ----------------------------------------------------------

  // Extract a view_info from python
  view_info from_python(numpy_rank_and_type info) {

    view_info v;
    v.rank         = info.rank;
    v.element_type = info.element_type;
    v.extents.resize(v.rank);
    v.strides.resize(v.rank);
    v.data =PyArray_DATA(info.arr);

#ifdef PYTHON_NUMPY_VERSION_LT_17
    for (long i = 0; i < v.rank; ++i) {
      v.extents[i] = size_t(info.arr->dimensions[i]);
      v.strides[i] = std::ptrdiff_t(info.arr->strides[i]);
    }
#else
    for (size_t i = 0; i < v.rank; ++i) {
      v.extents[i] = size_t(PyArray_DIMS(info.arr)[i]);
      v.strides[i] = std::ptrdiff_t(PyArray_STRIDES(info.arr)[i]);
    }
#endif

    return v;
  }

  // ----------------------------------------------------------

  PyObject *make_numpy_copy(PyObject *obj, int rank, long element_type) {

    if (obj == nullptr) return nullptr;
    if (_import_array() != 0) return nullptr;

    // From obj, we ask the numpy library to make a numpy, and of the correct type.
    // This handles automatically the cases where :
    //   - we have list, or list of list/tuple
    //   - the numpy type is not the one we want.
    //   - adjust the dimension if needed
    // If obj is an array :
    //   - if Order is same, don't change it
    //   - else impose it (may provoque a copy).
    // if obj is not array :
    //   - Order = FortranOrder or SameOrder - > Fortran order otherwise C

    //bool ForceCast = false;// Unless FORCECAST is present in flags, this call will generate an error if the data type cannot be safely obtained from the object.
    int flags = 0; //(ForceCast ? NPY_FORCECAST : 0) ;// do NOT force a copy | (make_copy ?  NPY_ENSURECOPY : 0);
                   //if (!(PyArray_Check(obj) ))
    //flags |= ( IndexMapType::traversal_order == indexmaps::mem_layout::c_order(rank) ? NPY_C_CONTIGUOUS : NPY_F_CONTIGUOUS); //impose mem order
#ifdef PYTHON_NUMPY_VERSION_LT_17
    flags |= (NPY_C_CONTIGUOUS); //impose mem order
    flags |= (NPY_ENSURECOPY);
#else
    flags |= (NPY_ARRAY_C_CONTIGUOUS); // impose mem order
    flags |= (NPY_ARRAY_ENSURECOPY);
#endif
    return PyArray_FromAny(obj, PyArray_DescrFromType(element_type), rank, rank, flags, NULL); // new ref
  }

} // namespace nda::python
