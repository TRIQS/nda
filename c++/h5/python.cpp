#include <Python.h>
#include <numpy/arrayobject.h>

#include "./group.hpp"
#include "./scalar.hpp"
#include "./string.hpp"
#include "./array_interface.hpp"
#include "./python.hpp"

namespace h5 {

  struct h5_c_size_t {
    datatype hdf5_type; // type in hdf5
    int c_size;         // size of the corresponding C object
  };

  std::vector<h5_c_size_t> h5_c_size_table;

  static void init_h5_c_size_table() {
    h5_c_size_table = std::vector<h5_c_size_t>{
       {hdf5_type<bool>, sizeof(bool)},
       {hdf5_type<char>, sizeof(char)},
       {hdf5_type<signed char>, sizeof(signed char)},
       {hdf5_type<unsigned char>, sizeof(unsigned char)},
       {hdf5_type<short>, sizeof(short)},
       {hdf5_type<unsigned short>, sizeof(unsigned short)},
       {hdf5_type<int>, sizeof(int)},
       {hdf5_type<unsigned int>, sizeof(unsigned int)},
       {hdf5_type<long>, sizeof(long)},
       {hdf5_type<unsigned long>, sizeof(unsigned long)},
       {hdf5_type<long long>, sizeof(long long)},
       {hdf5_type<unsigned long long>, sizeof(unsigned long long)},
       {hdf5_type<float>, sizeof(float)},
       {hdf5_type<double>, sizeof(double)},
       {hdf5_type<long double>, sizeof(long double)},
       {hdf5_type<std::complex<float>>, sizeof(std::complex<float>)},
       {hdf5_type<std::complex<double>>, sizeof(std::complex<double>)},
       {hdf5_type<std::complex<long double>>, sizeof(std::complex<long double>)} //
    };
  }

  // h5 -> numpy type conversion
  //FIXME we could sort the table and use binary_search
  int h5_c_size(datatype t) {
    if (h5_c_size_table.empty()) init_h5_c_size_table();
    auto _end = h5_c_size_table.end();
    auto pos  = std::find_if(h5_c_size_table.begin(), _end, [](auto const &x) { return x.hdf5_type == t; });
    if (pos == _end) std::runtime_error("HDF5/Python Internal Error : can not find the numpy type from the HDF5 type");
    return pos->c_size;
  }

  //---------------------------------------

  struct h5_py_type_t {
    datatype hdf5_type;   // type in hdf5
    NPY_TYPES numpy_type; // For a Python object, we will always use the numpy type
  };

  std::vector<h5_py_type_t> h5_py_type_table;

  static void init_h5py() {
    h5_py_type_table = std::vector<h5_py_type_t>{
       {hdf5_type<bool>, NPY_BOOL, sizeof(bool)},
       {hdf5_type<char>, NPY_STRING, sizeof(char)},
       {hdf5_type<signed char>, NPY_BYTE, sizeof(signed char)},
       {hdf5_type<unsigned char>, NPY_UBYTE, sizeof(unsigned char)},
       {hdf5_type<short>, NPY_SHORT, sizeof(short)},
       {hdf5_type<unsigned short>, NPY_USHORT, sizeof(unsigned short)},
       {hdf5_type<int>, NPY_INT, sizeof(int)},
       {hdf5_type<unsigned int>, NPY_UINT, sizeof(unsigned int)},
       {hdf5_type<long>, NPY_LONG, sizeof(long)},
       {hdf5_type<unsigned long>, NPY_ULONG, sizeof(unsigned long)},
       {hdf5_type<long long>, NPY_LONGLONG, sizeof(long long)},
       {hdf5_type<unsigned long long>, NPY_ULONGLONG, sizeof(unsigned long long)},
       {hdf5_type<float>, NPY_FLOAT, sizeof(float)},
       {hdf5_type<double>, NPY_DOUBLE, sizeof(double)},
       {hdf5_type<long double>, NPY_LONGDOUBLE, sizeof(long double)},
       {hdf5_type<std::complex<float>>, NPY_CFLOAT, sizeof(std::complex<float>)},
       {hdf5_type<std::complex<double>>, NPY_CDOUBLE, sizeof(std::complex<double>)},
       {hdf5_type<std::complex<long double>>, NPY_CLONGDOUBLE, sizeof(std::complex<long double>)} //
    };
  }

  // h5 -> numpy type conversion
  NPY_TYPES h5_to_npy(datatype t) {
    if (h5_py_type_table.empty()) init_h5py();
    auto _end = h5_py_type_table.end();
    auto pos  = std::find_if(h5_py_type_table.begin(), _end, [](auto const &x) { return x.hdf5_type == t; });
    if (pos == _end) std::runtime_error("HDF5/Python Internal Error : can not find the numpy type from the HDF5 type");
    return pos->numpy_type;
  }

  // numpy -> h5 type conversion
  datatype npy_to_h5(NPY_TYPES t) {
    if (h5_py_type_table.empty()) init_h5py();
    auto _end = h5_py_type_table.end();
    auto pos  = std::find_if(h5_py_type_table.begin(), _end, [](auto const &x) { return x.numpy_type == t; });
    if (pos == _end) std::runtime_error("HDF5/Python Internal Error : can not find the numpy type from the HDF5 type");
    return pos->hdf5_type;
  }

  //--------------------------------------

  // Make a h5_array_view from the numpy object
  static h5_array_view make_av_from_py(PyArrayObject *arr_obj) {

#ifdef PYTHON_NUMPY_VERSION_LT_17
    int elementsType = arr_obj->descr->type_num;
    int rank         = arr_obj->nd;
#else
    int elementsType = PyArray_DESCR(arr_obj)->type_num;
    int rank         = PyArray_NDIM(arr_obj);
#endif
    datatype dt     = npy_to_h5(elementsType);
    bool is_complex = (elementsType == NPY_CDOUBLE) or (elementsType == NPY_CLONGDOUBLE) or (elementsType == NPY_FLOAT);

    h5_array_view res{dt, PyArray_DATA(arr_obj), rank};

    for (int i = 0; i < rank; ++i) {
#ifdef PYTHON_NUMPY_VERSION_LT_17
      res.slab.count[i] = size_t(arr_obj->dimensions[i]);
      res.slab.stride[i] = std::ptrdiff_t(arr_obj->strides[i]) / h5_c_size(dt);
#else
      res.slab.count[i] = size_t(PyArray_DIMS(arr_obj)[i]);
      res.slab.stride[i] = std::ptrdiff_t(PyArray_STRIDES(arr_obj)[i]) / h5_c_size(dt);
#endif
    }

    return res;
  }

  // -------------------------

  void write(group g, std::string const &name, PyObject *ob) {

    // if numpy
    if (PyArray_Check(ob)) {
      PyArrayObject *arr_obj = (PyArrayObject *)ob;
      write(g, name, make_av_from_py(arr_obj));
    } else if (PyFloat_Check(ob)) {
      h5_write(g, name, PyFloat_AsDouble(ob));
    } else if (PyLong_Check(ob)) {
      h5_write(g, name, PyLong_AsLong(ob));
    } else if (PyString_Check(ob)) {
      h5_write(g, name, (const char *)PyString_AsString(ob));
    } else if (PyComplex_Check(ob)) {
      h5_write(g, name, std::complex<double>{PyComplex_RealAsDouble(ob), PyComplex_ImagAsDouble(ob)});
    } else {
      // Error !
      PyErr_SetString(PyExc_RuntimeError, "The Python object can not be written in HDF5");
    }
  }

  // -------------------------

  PyObject *read(group g, std::string const &name) noexcept { // There should be no errors from h5 reading

    h5_lengths_type lt = get_lengths_and_h5type(g, name);

    if (lt.rank() == 0) { // it is a scalar
      if (lt.ty == hdf5_type<double>) {
        double x;
        h5_read(g, name, x);
        return PyFloat_FromDouble(x);
      }
      if (lt.ty == hdf5_type<long>) { // or other int ...
        long x;
        h5_read(g, name, x);
        return PyLong_FromLong(x);
      }
      if (lt.ty == hdf5_type<std::string>) {
        std::string x;
        h5_read(g, name, x);
        return PyString_FromString(x.c_str());
      }
      if (lt.ty == hdf5_type<std::complex<double>>) {
        std::complex<double> x;
        h5_read(g, name, x);
        return PyComplex_FromDoubles(x.real(), x.imag());
      }
      std::abort(); // WE SHOULD COVER all types
    }
    // it is an array
    std::vector<npy_intp> L(lt.rank());                         // check
    std::copy(lt.lengths.begin(), lt.lengths.end(), L.begin()); //npy_intp and size_t may differ, so I can not use =
    int elementsType = h5_to_npy(lt.ty);

    // make a new numpy array
    ob = PyArray_SimpleNewFromDescr(int(L.size()), &L[0], PyArray_DescrFromType(elementsType));
    // leave the error set up in Python if any

    // read from the file
    // CATCH error
    read(g, name, make_av_from_py(lt.ty, (PyArrayObject *)ob), lt);
  }

} // namespace h5
