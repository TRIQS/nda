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
#include "clef.hpp"
#include <iostream>

namespace nda::clef {

  /* ---------------------------------------------------------------------------------------------------
  * Printing of the expressions
  *  --------------------------------------------------------------------------------------------------- */

  template <int N>
  std::ostream &operator<<(std::ostream &sout, placeholder<N>) {
    return sout << "_" << N;
  }
  template <typename T>
  std::ostream &operator<<(std::ostream &out, std::reference_wrapper<T> const &x) {
    return out << x.get();
  }

  template <typename Tag, typename... T>
  std::ostream &operator<<(std::ostream &sout, expr<Tag, T...> const &ex) {

    if constexpr (std::is_same_v<Tag, tags::function> or std::is_same_v<Tag, tags::subscript>) {
      bool is_fun = std::is_same_v<Tag, tags::function>;
      sout << "lambda" << (is_fun ? '(' : '[');
      auto print = [&sout](auto &&x, bool with_comma) mutable -> void {
        if (with_comma) sout << ", ";
        sout << x;
      };
      [&]<auto... Is>(std::index_sequence<Is...>) {
        (print(std::get<Is + 1>(ex.children), (Is > 0)), ...); // do not print arg 0, is it the function
      }
      (std::make_index_sequence<sizeof...(T) - 1>{});
      return sout << (is_fun ? ')' : ']');
    } else {
      if constexpr (sizeof...(T) == 2)
        return sout << "(" << std::get<0>(ex.children) << " " << get_tag_name(Tag{}) << " " << std::get<1>(ex.children) << ")";
      else
        return sout << "(" << get_tag_name(Tag{}) << " " << std::get<0>(ex.children) << ")";
    }
  }

  template <typename C, typename A, typename B>
  std::ostream &operator<<(std::ostream &sout, expr<tags::if_else, C, A, B> const &ex) {
    return sout << "(" << std::get<0>(ex.children) << "?" << std::get<1>(ex.children) << " : " << std::get<2>(ex.children) << ")";
  }

  template <typename T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::terminal, T> const &ex) {
    return sout << std::get<0>(ex.children);
  }

  template <typename T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::negate, T> const &ex) {
    return sout << "-(" << std::get<0>(ex.children) << ")";
  }

  template <typename Expr, int... Is>
  std::ostream &operator<<(std::ostream &sout, expr_as_function<Expr, Is...> const &x) {
    sout << "lazy function : (";
    (sout << ... << placeholder<Is>{});
    return sout << ") --> " << x.ex;
  }

} // namespace nda::clef
