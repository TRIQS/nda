/*
   Vim macro to regenerate all the mapping below with a clean code.
   Better than C macro (for error messages, no preproc, doc generation : otherwise no doc string ...)
   
  1 -select the pattern of your choice and put it in register t
      V}"ty
  2- define the macro (after :)
      let @a = '"byeG"tPVG:s/X/\=@b/g'
      (yank word in "b; go to end; Paste "t; remplate X-> word from "b)
  3- select the list of function to map below (abs, ...) or only the one you want to add
     V}k
  4- in normal mode, say (after :) to execute the macro on each selected line
    '<,'>  normal @a
 
  ---------  pattern 1 : normal mapping -------

  /// Maps X onto the array
  template <typename A>                                 
  auto X(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) { 
    return nda::map(                                     
       [](auto const &x) {                              
         using std::X;                                 
         return X(x);                                 
       })(std::forward<A>(a));                         
  }

  ---------------

abs
real
imag
floor
conj

 ---------  pattern 2 : same, no using std::-------

  /// Maps X onto the array
  template <typename A>                                 
  auto X(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) { 
    return nda::map(                                     
       [](auto const &x) {return X(x); },                                            
       std::forward<A>(a));                         
  }

  ---------------
 
conj_r
abs2
isnan

---------  pattern 3 : mapping with matrix excluded -------

  /// Maps X onto the array
  template <typename A>                                 
  auto X(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) { 
    return nda::map(                                     
       [](auto const &x) {                              
         using std::X;                                 
         return X(x);                                 
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
    return abs(x);
  })(std::forward<A>(a));
}

/// Maps real onto the array
template <typename A>
auto real(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
  return nda::map([](auto const &x) {
    using std::real;
    return real(x);
  })(std::forward<A>(a));
}

/// Maps imag onto the array
template <typename A>
auto imag(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
  return nda::map([](auto const &x) {
    using std::imag;
    return imag(x);
  })(std::forward<A>(a));
}

/// Maps floor onto the array
template <typename A>
auto floor(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
  return nda::map([](auto const &x) {
    using std::floor;
    return floor(x);
  })(std::forward<A>(a));
}

/// Maps conj onto the array
template <typename A>
auto conj(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
  return nda::map([](auto const &x) {
    using std::conj;
    return conj(x);
  })(std::forward<A>(a));
}

/// Maps isnan onto the array
template <typename A>
auto isnan(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
  return nda::map([](auto const &x) {
    return isnan(x);
  })(std::forward<A>(a));
}

/// Maps conj_r onto the array
template <typename A>
auto conj_r(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
  return nda::map([](auto const &x) { return conj_r(x); }, std::forward<A>(a));
}

/// Maps abs2 onto the array
template <typename A>
auto abs2(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
  return nda::map([](auto const &x) { return abs2(x); }, std::forward<A>(a));
}

/// Maps exp onto the array
template <typename A>
auto exp(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::exp;
       return exp(x);
     },
     std::forward<A>(a));
}

/// Maps cos onto the array
template <typename A>
auto cos(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::cos;
       return cos(x);
     },
     std::forward<A>(a));
}

/// Maps sin onto the array
template <typename A>
auto sin(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::sin;
       return sin(x);
     },
     std::forward<A>(a));
}

/// Maps tan onto the array
template <typename A>
auto tan(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::tan;
       return tan(x);
     },
     std::forward<A>(a));
}

/// Maps cosh onto the array
template <typename A>
auto cosh(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::cosh;
       return cosh(x);
     },
     std::forward<A>(a));
}

/// Maps sinh onto the array
template <typename A>
auto sinh(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::sinh;
       return sinh(x);
     },
     std::forward<A>(a));
}

/// Maps tanh onto the array
template <typename A>
auto tanh(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::tanh;
       return tanh(x);
     },
     std::forward<A>(a));
}

/// Maps acos onto the array
template <typename A>
auto acos(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::acos;
       return acos(x);
     },
     std::forward<A>(a));
}

/// Maps asin onto the array
template <typename A>
auto asin(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::asin;
       return asin(x);
     },
     std::forward<A>(a));
}

/// Maps atan onto the array
template <typename A>
auto atan(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::atan;
       return atan(x);
     },
     std::forward<A>(a));
}

/// Maps log onto the array
template <typename A>
auto log(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::log;
       return log(x);
     },
     std::forward<A>(a));
}

/// Maps sqrt onto the array
template <typename A>
auto sqrt(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
  return nda::map(
     [](auto const &x) {
       using std::sqrt;
       return sqrt(x);
     },
     std::forward<A>(a));
}
