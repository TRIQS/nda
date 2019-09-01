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
  2- put it register t :
      "ty
  3- select the macro 
      mz"bywG"tPVG:s/X/\=@b/g'z
      (mark z; yank word in "b; go to end; Paste "t; remplate X-> word from "b; return to z)
  4- and put it in "a
      "ay
  5- select the list of function to map below (abs, ...) or only the one you want to add
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

} // namespace nda DO NOT add a line below this (cf vim macro above)
