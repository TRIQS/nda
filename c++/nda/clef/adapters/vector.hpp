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
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include "../clef.hpp"
#include <vector>

namespace nda::clef {

  template <typename T, typename RHS>
  void clef_auto_assign__std_vector_impl(T &x, RHS &&rhs) {
    x = std::forward<RHS>(rhs);
  }

  template <typename Expr, int... Is, typename T>
  void clef_auto_assign__std_vector_impl(T &x, make_fun_impl<Expr, Is...> &&rhs) {
    clef_auto_assign_subscript(x, std::forward<make_fun_impl<Expr, Is...>>(rhs));
  }

  template <typename T, typename Fnt>
  void clef_auto_assign_subscript(std::vector<T> &v, Fnt f) {
    for (size_t i = 0; i < v.size(); ++i) clef_auto_assign__std_vector_impl(v[i], f(i));
  }

} // namespace nda::clef
