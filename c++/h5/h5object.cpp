#include <type_traits>

#include <H5Ipublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>

// FIXME
//static_assert(std::is_same_v<hid_t, int64_t>, "Configuration error in HDF5. Check version.");

#include "./h5object.hpp"

namespace h5 {

  //static_assert(std::is_same<::hid_t, hid_t>::value, "Internal error");
  static_assert(std::is_same<::hsize_t, hsize_t>::value, "Internal error");

  // specializations for all basic types
  template <> hid_t hdf5_type<char>          = H5T_NATIVE_CHAR;
  template <> hid_t hdf5_type<signed char>   = H5T_NATIVE_SCHAR;
  template <> hid_t hdf5_type<unsigned char> = H5T_NATIVE_UCHAR;

  template <> hid_t hdf5_type<short>     = H5T_NATIVE_SHORT;
  template <> hid_t hdf5_type<int>       = H5T_NATIVE_INT;
  template <> hid_t hdf5_type<long>      = H5T_NATIVE_LONG;
  template <> hid_t hdf5_type<long long> = H5T_NATIVE_LLONG;

  template <> hid_t hdf5_type<unsigned short>     = H5T_NATIVE_USHORT;
  template <> hid_t hdf5_type<unsigned int>       = H5T_NATIVE_UINT;
  template <> hid_t hdf5_type<unsigned long>      = H5T_NATIVE_ULONG;
  template <> hid_t hdf5_type<unsigned long long> = H5T_NATIVE_ULLONG;

  template <> hid_t hdf5_type<float>       = H5T_NATIVE_FLOAT;
  template <> hid_t hdf5_type<double>      = H5T_NATIVE_DOUBLE;
  template <> hid_t hdf5_type<long double> = H5T_NATIVE_LDOUBLE;

  template <> hid_t hdf5_type<std::complex<double>> = H5T_NATIVE_DOUBLE;

  // bool. Use a lambda to initialize it.
  template <>
  hid_t hdf5_type<bool> = []() {
    datatype bool_enum_h5type = H5Tenum_create(H5T_NATIVE_CHAR);
    char val;
    H5Tenum_insert(bool_enum_h5type, "FALSE", (val = 0, &val));
    H5Tenum_insert(bool_enum_h5type, "TRUE", (val = 1, &val));
    return bool_enum_h5type;
  }();

  // -----------------------   Reference counting ---------------------------

  // xdecref, xincref manipulate the the ref, but ignore invalid (incl. 0) id.
  //  like XINC_REF and XDEC_REF in python
  inline void xdecref(hid_t id) {
    if (H5Iis_valid(id)) H5Idec_ref(id);
  }

  inline void xincref(hid_t id) {
    if (H5Iis_valid(id)) H5Iinc_ref(id);
  }

  // -----------------------  h5_object  ---------------------------

  h5_object::h5_object(h5_object const &x) : id(x.id) { xincref(id); } // a new copy, a new ref.

  // make an h5_object when the id is now owned (simply inc. the ref).
  h5_object h5_object::from_borrowed(hid_t id) {
    xincref(id);
    return h5_object(id);
  }

  h5_object &h5_object::operator=(h5_object &&x) noexcept { //steals the ref, after properly decref its own.
    xdecref(id);
    id                          = x.id;
    x.id                        = 0;
    return *this;
  }

  h5_object::~h5_object() {
    // debug : to check the ref counting. Ok in tests.
    //if (H5Iis_valid(id)) std::cerr << "closing h5 object id = " << id << " # ref = "<< H5Iget_ref(id) << std::endl;
    xdecref(id);
  }

  void h5_object::close() {
    xdecref(id);
    id = 0;
  } // e.g. to close a file explicitely.

  bool h5_object::is_valid() const { return H5Iis_valid(id) == 1; }

} // namespace h5
