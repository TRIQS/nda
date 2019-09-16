#pragma once
#include "./macros.hpp"

namespace h5 {

  // a class T has either :
  //  1- a static member hdf5_scheme -> std::string (or a constexpr char * ?)
  //  2- specializes hdf5_scheme_impl
  // user function is get_hdf5_scheme <T>() in all cases.
  // A claass which is not default constructible :
  //  -- 1 : implement static T h5_read_construct(gr, name) : rebuilt  a new T
  //  -- 2 : NOT IMPLEMENTED : if we want to make it non intrusive,
  //  specialize with a struct similarly to hdf5_scheme_impl
  // to be implemented if needed.

  template <typename T>
  struct hdf5_scheme_impl {
    static std::string invoke() { return T::hdf5_scheme(); }
  };

#define H5_SPECIALIZE_HDF5_SCHEME2(X, Y)                                                                                                             \
  template <>                                                                                                                                        \
  struct hdf5_scheme_impl<X> {                                                                                                                       \
    static std::string invoke() { return H5_AS_STRING(Y); }                                                                                          \
  };

#define H5_SPECIALIZE_HDF5_SCHEME(X) H5_SPECIALIZE_HDF5_SCHEME2(X, X)

  H5_SPECIALIZE_HDF5_SCHEME(bool);
  H5_SPECIALIZE_HDF5_SCHEME(int);
  H5_SPECIALIZE_HDF5_SCHEME(long);
  H5_SPECIALIZE_HDF5_SCHEME(long long);
  H5_SPECIALIZE_HDF5_SCHEME(unsigned int);
  H5_SPECIALIZE_HDF5_SCHEME(unsigned long);
  H5_SPECIALIZE_HDF5_SCHEME(unsigned long long);
  H5_SPECIALIZE_HDF5_SCHEME(float);
  H5_SPECIALIZE_HDF5_SCHEME(double);
  H5_SPECIALIZE_HDF5_SCHEME(long double);
  H5_SPECIALIZE_HDF5_SCHEME2(std::complex<double>, complex);

  template <typename T>
  std::string get_hdf5_scheme() {
    return hdf5_scheme_impl<T>::invoke();
  }

  template <typename T>
  std::string get_hdf5_scheme(T const &) {
    return hdf5_scheme_impl<T>::invoke();
  }

  // A few helper functions

 /* /// Write the triqs tag*/
  //void write_hdf5_scheme_as_string(group g, const char *a) { h5_write_attribute(g, "TRIQS_HDF5_data_scheme", a); }

  ///// Write the triqs tag of the group if it is an object.
  //template <typename T> void write_hdf5_scheme(group g, T const &) { write_hdf5_scheme_as_string(g, ::h5::get_hdf5_scheme<T>().c_str()); }

  ///// Read the triqs tag of the group if it is an object. Returns the empty string "" if attribute is not present
  ////std::string read_hdf5_scheme() const;

  ///// Asserts that the tag of the group is the same as for T. Throws H5_ERROR if
  //void assert_hdf5_scheme_as_string(const char *tag_expected, bool ignore_if_absent = false) const;

  ///// Asserts that the tag of the group is the same as for T. Throws H5_ERROR if
  //template <typename T>
  //void assert_hdf5_scheme(T const &, bool ignore_if_absent = false) const {
    //assert_hdf5_scheme_as_string(get_hdf5_scheme<T>().c_str(), ignore_if_absent);
  /*}*/

} // namespace h5
