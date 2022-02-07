// Copyright (c) 2019-2021 Simons Foundation
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
  //template<typename T> std::ostream & operator<<(std::ostream & out, std::reference_wrapper<T> const & x) { return out<< "["<<x.get()<<"]"; }

  inline std::ostream &variadic_print(std::ostream &out) { return out; }
  template <typename T0, typename... T>
  std::ostream &variadic_print(std::ostream &out, T0 &&t0, T &&... t) {
    out << t0 << (sizeof...(T) > 0 ? ", " : "");
    variadic_print(out, t...);
    return out;
  }

  template <int c, int m>
  struct print_tuple_impl {
    template <typename Tuple>
    void operator()(std::ostream &out, Tuple const &t) {
      out << std::get<c>(t) << (c != m - 1 ? ", " : "");
      print_tuple_impl<c + 1, m>()(out, t);
    }
  };
  template <int m>
  struct print_tuple_impl<m, m> {
    template <typename Tuple>
    void operator()(std::ostream &, Tuple const &) {}
  };
  template <typename Tuple>
  std::ostream &print_tuple(std::ostream &out, Tuple const &t) {
    print_tuple_impl<1, std::tuple_size<Tuple>::value>()(out, t);
    return out;
  }
  template <typename Tag, typename L>
  typename std::enable_if<std::is_base_of<tags::unary_op, Tag>::value, std::ostream &>::type operator<<(std::ostream &sout, expr<Tag, L> const &ex) {
    return sout << "(" << Tag::name() << " " << std::get<0>(ex.childs) << ")";
  }
  template <typename Tag, typename L, typename R>
  typename std::enable_if<std::is_base_of<tags::binary_op, Tag>::value, std::ostream &>::type operator<<(std::ostream &sout,
                                                                                                         expr<Tag, L, R> const &ex) {
    return sout << "(" << std::get<0>(ex.childs) << " " << Tag::name() << " " << std::get<1>(ex.childs) << ")";
  }
  template <typename C, typename A, typename B>
  std::ostream &operator<<(std::ostream &sout, expr<tags::if_else, C, A, B> const &ex) {
    return sout << "(" << std::get<0>(ex.childs) << "?" << std::get<1>(ex.childs) << " : " << std::get<2>(ex.childs) << ")";
  }
  template <typename... T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::function, T...> const &ex) {
    sout << "lambda"
         << "(";
    //sout << std::get<0>(ex.childs) << "(";
    print_tuple(sout, ex.childs);
    return sout << ")";
  }
  template <typename... T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::subscript, T...> const &ex) {
    sout << "lambda"
         << "[";
    //sout << std::get<0>(ex.childs) << "[";
    print_tuple(sout, ex.childs);
    return sout << "]";
  }

  template <typename T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::terminal, T> const &ex) {
    return sout << std::get<0>(ex.childs);
  }

  template <typename T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::subscript, T> const &ex) {
    return sout << std::get<0>(ex.childs) << "[" << std::get<1>(ex.childs) << "]";
  }

  template <typename T>
  std::ostream &operator<<(std::ostream &sout, expr<tags::negate, T> const &ex) {
    return sout << "-(" << std::get<0>(ex.childs) << ")";
  }

  template <typename Expr, int... Is>
  std::ostream &operator<<(std::ostream &sout, make_fun_impl<Expr, Is...> const &x) {
    sout << "lazy function : (";
    variadic_print(sout, placeholder<Is>()...);
    return sout << ") --> " << x.ex;
  }

} // namespace nda::clef
