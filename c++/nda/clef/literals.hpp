// Copyright (c) 2019-2020 Simons Foundation
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

#pragma once
#include "./clef.hpp"

namespace nda::clef::literals {

  // Define literal placeholders starting from the end of the ph index spectrum
#define PH(I) placeholder<std::numeric_limits<int>::max() - I>{}

  constexpr auto i_ = PH(0);
  constexpr auto j_ = PH(1);
  constexpr auto k_ = PH(2);
  constexpr auto l_ = PH(3);

  constexpr auto bl_ = PH(4);

  constexpr auto w_ = PH(5);
  constexpr auto iw_ = PH(6);
  constexpr auto W_ = PH(7);
  constexpr auto iW_ = PH(8);
  constexpr auto t_ = PH(9);
  constexpr auto tau_ = PH(10);

#undef PH

} // namespace nda::clef::literals
