#pragma once
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

  template <typename T> struct hdf5_scheme_impl {
    static std::string invoke() { return T::hdf5_scheme(); }
  };

#define H5_AS_STRING(X) H5_AS_STRING2(X)
#define H5_AS_STRING2(X) #X

#define TRIQS_SPECIALIZE_HDF5_SCHEME2(X, Y)                                                                                                          \
  template <> struct hdf5_scheme_impl<X> {                                                                                                           \
    static std::string invoke() { return H5_AS_STRING(Y); }                                                                                             \
  };

#define TRIQS_SPECIALIZE_HDF5_SCHEME(X) TRIQS_SPECIALIZE_HDF5_SCHEME2(X, X)

  TRIQS_SPECIALIZE_HDF5_SCHEME(bool);
  TRIQS_SPECIALIZE_HDF5_SCHEME(int);
  TRIQS_SPECIALIZE_HDF5_SCHEME(long);
  TRIQS_SPECIALIZE_HDF5_SCHEME(long long);
  TRIQS_SPECIALIZE_HDF5_SCHEME(unsigned int);
  TRIQS_SPECIALIZE_HDF5_SCHEME(unsigned long);
  TRIQS_SPECIALIZE_HDF5_SCHEME(unsigned long long);
  TRIQS_SPECIALIZE_HDF5_SCHEME(float);
  TRIQS_SPECIALIZE_HDF5_SCHEME(double);
  TRIQS_SPECIALIZE_HDF5_SCHEME(long double);
  TRIQS_SPECIALIZE_HDF5_SCHEME2(std::complex<double>, complex);

  template <typename T> std::string get_hdf5_scheme() { return hdf5_scheme_impl<T>::invoke(); }

}
