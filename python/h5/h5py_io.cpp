#include <Python.h>
#include <numpy/arrayobject.h>

#include "h5py_io.hpp"
#include <h5/group.hpp>
#include <h5/scalar.hpp>
#include <h5/stl/string.hpp>
#include <h5/array_interface.hpp>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <algorithm>

namespace h5 {

  struct h5_c_size_t {
    datatype hdf5_type; // type in hdf5
    int c_size;         // size of the corresponding C object
  };

  std::vector<h5_c_size_t> h5_c_size_table;

  static void init_h5_c_size_table() {
    h5_c_size_table = std::vector<h5_c_size_t>{
       {hdf5_type<bool>(), sizeof(bool)},
       {hdf5_type<char>(), sizeof(char)},
       {hdf5_type<signed char>(), sizeof(signed char)},
       {hdf5_type<unsigned char>(), sizeof(unsigned char)},
       {hdf5_type<short>(), sizeof(short)},
       {hdf5_type<unsigned short>(), sizeof(unsigned short)},
       {hdf5_type<int>(), sizeof(int)},
       {hdf5_type<unsigned int>(), sizeof(unsigned int)},
       {hdf5_type<long>(), sizeof(long)},
       {hdf5_type<unsigned long>(), sizeof(unsigned long)},
       {hdf5_type<long long>(), sizeof(long long)},
       {hdf5_type<unsigned long long>(), sizeof(unsigned long long)},
       {hdf5_type<float>(), sizeof(float)},
       {hdf5_type<double>(), sizeof(double)},
       {hdf5_type<long double>(), sizeof(long double)},
       {hdf5_type<std::complex<float>>(), sizeof(std::complex<float>)},
       {hdf5_type<std::complex<double>>(), sizeof(std::complex<double>)},
       {hdf5_type<std::complex<long double>>(), sizeof(std::complex<long double>)} //
    };
  }

  // h5 -> numpy type conversion
  //FIXME we could sort the table and use binary_search
  int h5_c_size(datatype t) {
    if (h5_c_size_table.empty()) init_h5_c_size_table();
    auto _end = h5_c_size_table.end();
    auto pos  = std::find_if(h5_c_size_table.begin(), _end, [t](auto const &x) { return x.hdf5_type == t; });
    if (pos == _end) std::runtime_error("HDF5/Python Internal Error : can not find the numpy type from the HDF5 type");
    return pos->c_size;
  }

  //---------------------------------------

  struct h5_py_type_t {
    datatype hdf5_type; // type in hdf5
    int numpy_type;     // For a Python object, we will always use the numpy type
    size_t size;
  };

  //--------------------------------------

  std::vector<h5_py_type_t> h5_py_type_table;

  //--------------------------------------

  static void init_h5py() {
    h5_py_type_table = std::vector<h5_py_type_t>{
       //  {hdf5_type<bool>(), NPY_BOOL, sizeof(bool)},
       {hdf5_type<char>(), NPY_STRING, sizeof(char)},
       {hdf5_type<signed char>(), NPY_BYTE, sizeof(signed char)},
       {hdf5_type<unsigned char>(), NPY_UBYTE, sizeof(unsigned char)},
       {hdf5_type<short>(), NPY_SHORT, sizeof(short)},
       {hdf5_type<unsigned short>(), NPY_USHORT, sizeof(unsigned short)},
       {hdf5_type<int>(), NPY_INT, sizeof(int)},
       {hdf5_type<unsigned int>(), NPY_UINT, sizeof(unsigned int)},
       {hdf5_type<long>(), NPY_LONG, sizeof(long)},
       {hdf5_type<unsigned long>(), NPY_ULONG, sizeof(unsigned long)},
       {hdf5_type<long long>(), NPY_LONGLONG, sizeof(long long)},
       {hdf5_type<unsigned long long>(), NPY_ULONGLONG, sizeof(unsigned long long)},
       {hdf5_type<float>(), NPY_FLOAT, sizeof(float)},
       {hdf5_type<double>(), NPY_DOUBLE, sizeof(double)},
       {hdf5_type<long double>(), NPY_LONGDOUBLE, sizeof(long double)},
       {hdf5_type<std::complex<float>>(), NPY_CFLOAT, sizeof(std::complex<float>)},
       {hdf5_type<std::complex<double>>(), NPY_CDOUBLE, sizeof(std::complex<double>)},
       {hdf5_type<std::complex<long double>>(), NPY_CLONGDOUBLE, sizeof(std::complex<long double>)} //
    };
  }

  //--------------------------------------

  // h5 -> numpy type conversion
  int h5_to_npy(datatype t, bool is_complex) {

    if (h5_py_type_table.empty()) init_h5py();
    auto _end = h5_py_type_table.end();
    auto pos  = std::find_if(h5_py_type_table.begin(), _end, [t](auto const &x) { return H5Tequal(x.hdf5_type, t) > 0; });
    if (pos == _end) throw std::runtime_error("HDF5/Python Internal Error : can not find the numpy type from the HDF5 type");
    int res = pos->numpy_type;
    if (is_complex) {
      if (res == NPY_DOUBLE) res = NPY_CDOUBLE;
      if (res == NPY_FLOAT) res = NPY_CFLOAT;
      if (res == NPY_LONGDOUBLE) res = NPY_CLONGDOUBLE;
    }
    return res;
  }

  //--------------------------------------

  // numpy -> h5 type conversion
  datatype npy_to_h5(int t) {
    if (h5_py_type_table.empty()) init_h5py();
    auto _end = h5_py_type_table.end();
    auto pos  = std::find_if(h5_py_type_table.begin(), _end, [t](auto const &x) { return x.numpy_type == t; });
    if (pos == _end) std::runtime_error("HDF5/Python Internal Error : can not find the numpy type from the HDF5 type");
    return pos->hdf5_type;
  }

  //--------------------------------------

  // Make a h5_array_view from the numpy object
  static array_interface::h5_array_view make_av_from_py(PyArrayObject *arr_obj) {

#ifdef PYTHON_NUMPY_VERSION_LT_17
    int elementsType = arr_obj->descr->type_num;
    int rank         = arr_obj->nd;
#else
    int elementsType = PyArray_DESCR(arr_obj)->type_num;
    int rank         = PyArray_NDIM(arr_obj);
#endif
    datatype dt           = npy_to_h5(elementsType);
    const bool is_complex = (elementsType == NPY_CDOUBLE) or (elementsType == NPY_CLONGDOUBLE) or (elementsType == NPY_CFLOAT);

    array_interface::h5_array_view res{dt, PyArray_DATA(arr_obj), rank, is_complex};
    std::vector<long> c_strides(rank + is_complex, 0);
    long total_size = 1;

    for (int i = 0; i < rank; ++i) {
#ifdef PYTHON_NUMPY_VERSION_LT_17
      res.slab.count[i] = size_t(arr_obj->dimensions[i]);
      c_strides[i]      = std::ptrdiff_t(arr_obj->strides[i]) / h5_c_size(dt);
#else
      res.slab.count[i] = size_t(PyArray_DIMS(arr_obj)[i]);
      c_strides[i]      = std::ptrdiff_t(PyArray_STRIDES(arr_obj)[i]) / h5_c_size(dt);
#endif
      total_size *= res.slab.count[i];
    }

    // be careful to consider the last dim if complex, but do NOT copy it
    auto [Ltot, stri] = h5::array_interface::get_L_tot_and_strides_h5(c_strides.data(), rank + is_complex, total_size * (is_complex ? 2 : 1));
    for (int i = 0; i < rank; ++i) {
      res.L_tot[i]       = Ltot[i];
      res.slab.stride[i] = stri[i];
    }

    return res;
  }

  // -------------------------
  static void import_numpy() {
    static bool init = false;
    if (!init) {
      _import_array();
      init = true;
    }
  }
  // -------------------------

  void h5_write_bare(group g, std::string const &name, PyObject *ob) {

    import_numpy();

    // if numpy
    if (PyArray_Check(ob)) {
      PyArrayObject *arr_obj = (PyArrayObject *)ob;
      write(g, name, make_av_from_py(arr_obj), true);
    } else if (PyFloat_Check(ob)) {
      h5_write(g, name, PyFloat_AsDouble(ob));
    } else if (PyInt_Check(ob)) {
      h5_write(g, name, long(PyInt_AsLong(ob)));
    } else if (PyLong_Check(ob)) {
      h5_write(g, name, long(PyLong_AsLong(ob)));
    } else if (PyString_Check(ob)) {
      h5_write(g, name, (const char *)PyString_AsString(ob));
    } else if (PyComplex_Check(ob)) {
      h5_write(g, name, std::complex<double>{PyComplex_RealAsDouble(ob), PyComplex_ImagAsDouble(ob)});
    } else {
      PyErr_SetString(PyExc_RuntimeError, "The Python object can not be written in HDF5");
      return;
    }
  }

  // -------------------------

  PyObject *h5_read_bare(group g, std::string const &name) { // There should be no errors from h5 reading
    import_numpy();

    array_interface::h5_lengths_type lt = array_interface::get_h5_lengths_type(g, name);

    // First case, we have a scalar
    if (lt.rank() == 0) {
      if (H5Tget_class(lt.ty) == H5T_FLOAT) {
        //if (H5Tequal(lt.ty, hdf5_type<double>) > 0) {
        double x;
        h5_read(g, name, x);
        return PyFloat_FromDouble(x);
      }
      if (H5Tget_class(lt.ty) == H5T_INTEGER) {
        // if (H5Tequal(lt.ty, hdf5_type<long>) > 0) {
        long x;
        h5_read(g, name, x);
        return PyLong_FromLong(x);
      }
      if (H5Tget_class(lt.ty) == H5T_STRING) {
        std::string x;
        h5_read(g, name, x);
        return PyString_FromString(x.c_str());
      }
      // Default case : error, we can not read
      PyErr_SetString(PyExc_RuntimeError, "h5_read to Python: unknown scalar type");
      return NULL;
    }

    // A scalar complex is a special case
    if ((lt.rank() == 1) and lt.has_complex_attribute) {
      std::complex<double> z;
      h5_read(g, name, z);
      return PyComplex_FromDoubles(z.real(), z.imag());
    }

    // Last case : it is an array

    std::vector<npy_intp> L(lt.rank());                            // Make the lengths
    std::copy(lt.lengths.begin(), lt.lengths.end(), L.begin());    //npy_intp and size_t may differ, so I can not use =
    int elementsType = h5_to_npy(lt.ty, lt.has_complex_attribute); // element_type in Python from the hdf5 type and complex tag
    if (lt.has_complex_attribute)
      L.pop_back(); // remove the last dim which is 2 in complex case,
                    // since we are going to build a array of complex

    // make a new numpy array
    PyObject *ob = PyArray_SimpleNewFromDescr(int(L.size()), &L[0], PyArray_DescrFromType(elementsType));
    if (PyErr_Occurred()) return NULL;
    // in case of allocation error

    // read from the file
    read(g, name, make_av_from_py((PyArrayObject *)ob), lt);
    return ob;
  }

} // namespace h5
