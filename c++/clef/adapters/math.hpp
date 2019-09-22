#pragma once

#include "../clef.hpp"
#include <cmath>
#include <complex>

//#define TRIQS_CLEF_STD_MATH_FNT_TO_MAKE_LAZY (cos)(sin)(tan)(cosh)(sinh)(tanh)(acos)(asin)(atan)(exp)(log)(sqrt)(abs)(floor)(pow)(conj)

namespace clef {

#define TRIQS_CLEF_MAKE_STD_FNT_LAZY(name)                                                                                                           \
  using std::name;                                                                                                                                   \
  TRIQS_CLEF_MAKE_FNT_LAZY(name)

  // FIXME use vim generation
  TRIQS_CLEF_MAKE_STD_FNT_LAZY(cos)
  TRIQS_CLEF_MAKE_STD_FNT_LAZY(abs)

} // namespace clef
