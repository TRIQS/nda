// Copyright (c) 2020-2021 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include "../clef.hpp"
#include <cmath>
#include <complex>

namespace nda::clef {

#define CLEF_MAKE_STD_FNT_LAZY(name)                                                                                                                 \
  using std::name;                                                                                                                                   \
  CLEF_MAKE_FNT_LAZY(name)

  // FIXME use vim generation
  CLEF_MAKE_STD_FNT_LAZY(cos)
  CLEF_MAKE_STD_FNT_LAZY(sin)
  CLEF_MAKE_STD_FNT_LAZY(tan)
  CLEF_MAKE_STD_FNT_LAZY(cosh)
  CLEF_MAKE_STD_FNT_LAZY(sinh)
  CLEF_MAKE_STD_FNT_LAZY(tanh)
  CLEF_MAKE_STD_FNT_LAZY(acos)
  CLEF_MAKE_STD_FNT_LAZY(asin)
  CLEF_MAKE_STD_FNT_LAZY(atan)
  CLEF_MAKE_STD_FNT_LAZY(exp)
  CLEF_MAKE_STD_FNT_LAZY(log)
  CLEF_MAKE_STD_FNT_LAZY(sqrt)
  CLEF_MAKE_STD_FNT_LAZY(abs)
  CLEF_MAKE_STD_FNT_LAZY(floor)
  CLEF_MAKE_STD_FNT_LAZY(pow)
  CLEF_MAKE_STD_FNT_LAZY(conj)

} // namespace nda::clef
