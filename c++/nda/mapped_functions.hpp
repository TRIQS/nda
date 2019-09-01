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

  ///
  inline bool any(bool x) { return x; } // for generic codes

  /// pow for integer
  template <typename T>
  T pow(T x, int n) REQUIRES(std::is_integral<T>) {
    T r = 1;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
  }

  // ---------------  The Mapped function using map--------------

  // can not use a macro or I can not write the doc !

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

  /*
   Vim macro to regenerate all the mappingi below with a clean code.
   Better than C macro (for error messages, no preproc, doc generation : otherwise no doc string ...)
   
  0- Make sure the }// namesapce is still the last line 
  1 -select the pattern of your choice 
     V}
  2- put it register t :
      "ty
  3- define the macro 
      :let @a = '"bywG"tPVG:s/X/\=@b/g'
      (yank word in "b; go to end; Paste "t; remplate X-> word from "b)
  4- select the list of function to map below (abs, ...) or only the one you want to add
     V}k
  6- use : and say
    :  <..> normal @a
 
  ---------  normal mapping -------

  /// Maps X onto the array
  template <typename A>                                 
  auto X(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) { 
    return nda::map(                                     
       [](auto const &x) {                              
         using std::X;                                 
         return X(a);                                 
       })(std::forward<A>(a));                         
  }

  ---------------

abs
real
imag
floor
conj
isnan

 ---------  same, no using std::-------

  /// Maps X onto the array
  template <typename A>                                 
  auto X(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) { 
    return nda::map(                                     
       [](auto const &x) {return X(a); },                                            
       std::forward<A>(a));                         
  }

  ---------------
 
conj_r
abs2

  ---------   mapping with matrix excluded -------

  /// Maps X onto the array
  template <typename A>                                 
  auto X(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) { 
    return nda::map(                                     
       [](auto const &x) {                              
         using std::X;                                 
         return X(a);                                 
       },                                            
       std::forward<A>(a));                         
  }

  ---------------

exp
cos
sin
tan
cosh
sinh
tanh
acos
asin
atan
log
sqrt

*/

  /// Maps abs onto the array
  template <typename A>
  auto abs(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([](auto const &x) {
      using std::abs;
      return abs(a);
    })(std::forward<A>(a));
  }

  /// Maps real onto the array
  template <typename A>
  auto real(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([](auto const &x) {
      using std::real;
      return real(a);
    })(std::forward<A>(a));
  }

  /// Maps imag onto the array
  template <typename A>
  auto imag(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([](auto const &x) {
      using std::imag;
      return imag(a);
    })(std::forward<A>(a));
  }

  /// Maps floor onto the array
  template <typename A>
  auto floor(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([](auto const &x) {
      using std::floor;
      return floor(a);
    })(std::forward<A>(a));
  }

  /// Maps conj onto the array
  template <typename A>
  auto conj(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([](auto const &x) {
      using std::conj;
      return conj(a);
    })(std::forward<A>(a));
  }

  /// Maps isnan onto the array
  template <typename A>
  auto isnan(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([](auto const &x) {
      using std::isnan;
      return isnan(a);
    })(std::forward<A>(a));
  }

  /// Maps conj_r onto the array
  template <typename A>
  auto conj_r(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([](auto const &x) { return conj_r(a); }, std::forward<A>(a));
  }

  /// Maps abs2 onto the array
  template <typename A>
  auto abs2(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return nda::map([](auto const &x) { return abs2(a); }, std::forward<A>(a));
  }

  /// Maps exp onto the array
  template <typename A>
  auto exp(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::exp;
         return exp(a);
       },
       std::forward<A>(a));
  }

  /// Maps cos onto the array
  template <typename A>
  auto cos(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::cos;
         return cos(a);
       },
       std::forward<A>(a));
  }

  /// Maps sin onto the array
  template <typename A>
  auto sin(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::sin;
         return sin(a);
       },
       std::forward<A>(a));
  }

  /// Maps tan onto the array
  template <typename A>
  auto tan(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::tan;
         return tan(a);
       },
       std::forward<A>(a));
  }

  /// Maps cosh onto the array
  template <typename A>
  auto cosh(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::cosh;
         return cosh(a);
       },
       std::forward<A>(a));
  }

  /// Maps sinh onto the array
  template <typename A>
  auto sinh(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::sinh;
         return sinh(a);
       },
       std::forward<A>(a));
  }

  /// Maps tanh onto the array
  template <typename A>
  auto tanh(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::tanh;
         return tanh(a);
       },
       std::forward<A>(a));
  }

  /// Maps acos onto the array
  template <typename A>
  auto acos(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::acos;
         return acos(a);
       },
       std::forward<A>(a));
  }

  /// Maps asin onto the array
  template <typename A>
  auto asin(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::asin;
         return asin(a);
       },
       std::forward<A>(a));
  }

  /// Maps atan onto the array
  template <typename A>
  auto atan(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::atan;
         return atan(a);
       },
       std::forward<A>(a));
  }

  /// Maps log onto the array
  template <typename A>
  auto log(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::log;
         return log(a);
       },
       std::forward<A>(a));
  }

  /// Maps sqrt onto the array
  template <typename A>
  auto sqrt(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return nda::map(
       [](auto const &x) {
         using std::sqrt;
         return sqrt(a);
       },
       std::forward<A>(a));
  }

} // namespace nda
