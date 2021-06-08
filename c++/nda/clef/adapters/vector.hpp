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
#include "../clef.hpp"
#include <vector>

namespace nda::clef {

  template <typename T, typename RHS, typename Tag, typename PhList, typename... CTArgs>
  void clef_auto_assign(std::vector<T> &v, RHS &&rhs, Tag, PhList phl, CTArgs...) {
    auto f = nda::clef::make_function(std::forward<RHS>(rhs), phl);
    static_assert(std::is_same_v<Tag, tags::subscript>, "() is not defined for a std::vector");
    for (size_t i = 0; i < v.size(); ++i) {
      if constexpr (sizeof...(CTArgs) > 0)
        clef_auto_assign(v[i], f(i), CTArgs{}...);
      else
        v[i] = f(i);
    }
  }

} // namespace nda::clef
