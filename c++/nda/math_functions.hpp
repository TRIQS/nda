#pragma once
#include "./functional/map.hpp"
#include <boost/preprocessor/seq/for_each.hpp>
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

  ///
  inline bool any(bool x) { return x; } // for generic codes

  /// pow for integer
  template <typename T>
  T pow(T x, int n) REQUIRES(std::is_integral<T>) {
    T r = 1;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
  }

  // ---------------  a few additional functions --------------

  /// Map pow on Ndarray
  template <typename A>
  auto pow(A &&a, int n) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map(
       [n](auto const &x) {
         using std::pow;
         return pow(a, n);
       },
       std::forward<A>(a));
  }

  /// Map abs on Ndarray
  template <typename A>
  auto abs(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map(
       [](auto const &x) {
         using std::abs;
         return abs(a);
       },
       std::forward<A>(a));
  }

#define NDA_IMPL_MAP(X)                                                                                                                              \
  template <typename A>                                                                                                                              \
  auto abs(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {                                                                                          \
    return nda::map(                                                                                                                                 \
       [](auto const &x) {                                                                                                                           \
         using std::abs;                                                                                                                             \
         return abs(a);                                                                                                                              \
       },                                                                                                                                            \
       std::forward<A>(a));                                                                                                                          \
  }

#define NDA_IMPL_MAP_NOT_MATRIX(X)                                                                                                                   \
  template <typename A>                                                                                                                              \
  auto abs(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {                                                \
    return nda::map(                                                                                                                                 \
       [](auto const &x) {                                                                                                                           \
         using std::abs;                                                                                                                             \
         return abs(a);                                                                                                                              \
       },                                                                                                                                            \
       std::forward<A>(a));                                                                                                                          \
  }

#define NDA_IMPL_MAP_NO_STD(X)                                                                                                                       \
  template <typename A>                                                                                                                              \
  auto abs(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {                                                                                          \
    return nda::map([](auto const &x) { return abs(a); }, std::forward<A>(a));                                                                       \
  }

  NDA_IMPL_MAP_NO_STD(conj_r);
  NDA_IMPL_MAP_NO_STD(abs2);

  NDA_IMPL_MAP(abs);
  NDA_IMPL_MAP(real);
  NDA_IMPL_MAP(imag);
  NDA_IMPL_MAP(floor);
  NDA_IMPL_MAP(conj);
  NDA_IMPL_MAP(isnan);

  NDA_IMPL_MAP_NOT_MATRIX(exp);
  NDA_IMPL_MAP_NOT_MATRIX(cos);
  NDA_IMPL_MAP_NOT_MATRIX(sin);
  NDA_IMPL_MAP_NOT_MATRIX(tan);
  NDA_IMPL_MAP_NOT_MATRIX(cosh);
  NDA_IMPL_MAP_NOT_MATRIX(sinh);
  NDA_IMPL_MAP_NOT_MATRIX(tanh);
  NDA_IMPL_MAP_NOT_MATRIX(acos);
  NDA_IMPL_MAP_NOT_MATRIX(asin);
  NDA_IMPL_MAP_NOT_MATRIX(atan);
  NDA_IMPL_MAP_NOT_MATRIX(log);
  NDA_IMPL_MAP_NOT_MATRIX(sqrt);

} // namespace nda
