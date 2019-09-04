#pragma once
#include "./map.hpp"

namespace nda {

  // ---------------  a few additional functions --------------

  ///
  inline double conj_r(double x) { return x; }

  ///
  inline std::complex<double> conj_r(std::complex<double> x) { return std::conj(x); }

  ///
  inline double abs2(double x) { return x * x; }

  ///
  inline double abs2(std::complex<double> x) { return (std::conj(x) * x).real(); }

  // not for libc++ (already defined)
#if !defined(_LIBCPP_VERSION)
  // complex conjugation for integers
  inline std::complex<double> conj(int x) { return x; }
  inline std::complex<double> conj(long x) { return x; }
  inline std::complex<double> conj(long long x) { return x; }
  inline std::complex<double> conj(double x) { return x; }
#endif

  ///
  inline bool isnan(std::complex<double> const &z) { return std::isnan(z.real()) or std::isnan(z.imag()); }

  /// FIXME ??
  //  inline bool any(bool x) { return x; } // for generic codes

  /// pow for integer
  template <typename T>
  T pow(T x, int n) REQUIRES(std::is_integral_v<T>) {
    T r = 1;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
  }

  // ---------------  The Mapped function using map--------------

  // can not use a macro or I can not write the doc !

  /// Map pow on Ndarray
  template <typename A>
  auto pow(A &&a, int n) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([n](auto const &x) {
      using std::pow;
      return pow(x, n);
    })(std::forward<A>(a));
  }

  // the automatically generated once. Cf file, with vim commands
#include "./mapped_functions.hxx"

} // namespace nda
