#pragma once

#include "./h5/file.hpp"
#include "./h5/group.hpp"
#include "./h5/scheme.hpp"
#include "./h5/scalar.hpp"
#include "./h5/generic.hpp"
#include "./h5/stl/string.hpp"
#include "./h5/stl/vector.hpp"

//#include "./h5/stl/map.hpp"
//#include "./h5/stl/pair.hpp"
//#include "./h5/stl/tuple.hpp"
//#include "./h5/stl/optional.hpp"
//#include "./h5/stl/variant.hpp"

// Correction of a bug in STL : 
namespace std { // has to be in the right namespace for ADL !

  template <typename T> std::complex<T> operator+(std::complex<T> const &a, long b) { return a + T(b); }
  template <typename T> std::complex<T> operator+(long a, std::complex<T> const &b) { return T(a) + b; }
  template <typename T> std::complex<T> operator-(std::complex<T> const &a, long b) { return a - T(b); }
  template <typename T> std::complex<T> operator-(long a, std::complex<T> const &b) { return T(a) - b; }
  template <typename T> std::complex<T> operator*(std::complex<T> const &a, long b) { return a * T(b); }
  template <typename T> std::complex<T> operator*(long a, std::complex<T> const &b) { return T(a) * b; }
  template <typename T> std::complex<T> operator/(std::complex<T> const &a, long b) { return a / T(b); }
  template <typename T> std::complex<T> operator/(long a, std::complex<T> const &b) { return T(a) / b; }
} // namespace std

// FIXME : Still needed ?
// for python code generator, we need to know what has to been included.
//#define TRIQS_INCLUDED_H5

// in some old version of hdf5 (Ubuntu 12.04 e.g.), the macro is not yet defined.
#ifndef H5_VERSION_GE

#define H5_VERSION_GE(Maj, Min, Rel)                                                                                                                 \
  (((H5_VERS_MAJOR == Maj) && (H5_VERS_MINOR == Min) && (H5_VERS_RELEASE >= Rel)) || ((H5_VERS_MAJOR == Maj) && (H5_VERS_MINOR > Min))               \
   || (H5_VERS_MAJOR > Maj))

#endif

