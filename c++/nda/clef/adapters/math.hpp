#pragma once

#include "../clef.hpp"
#include <cmath>
#include <complex>

//#define CLEF_STD_MATH_FNT_TO_MAKE_LAZY (cos)(sin)(tan)(cosh)(sinh)(tanh)(acos)(asin)(atan)(exp)(log)(sqrt)(abs)(floor)(pow)(conj)

namespace nda::clef {

#define CLEF_MAKE_STD_FNT_LAZY(name)                                                                                                                 \
  using std::name;                                                                                                                                   \
  CLEF_MAKE_FNT_LAZY(name)

  // FIXME use vim generation
  CLEF_MAKE_STD_FNT_LAZY(cos)
  CLEF_MAKE_STD_FNT_LAZY(abs)

} // namespace nda::clef
