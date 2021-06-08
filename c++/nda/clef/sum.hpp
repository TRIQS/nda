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

namespace nda {
  // forward : declared in basic_function
  template <typename A>
  auto make_regular(A &&x);
} // namespace nda

namespace nda::clef {

  //--------------------------------------------------------------------------------------------------
  //  sum of expressions
  // -------------------------------------------------------------------------------------------------

  // sum a function f on a domain D, using a simple foreach
  template <typename F, typename D>
  auto sum_f_domain_impl(F const &f, D const &d) {
    static_assert(not Lazy<F>);
    static_assert(not Lazy<D>);
    auto it  = d.begin();
    auto ite = d.end();
    if (it == ite) NDA_RUNTIME_ERROR << "Sum over an empty domain";
    auto res = make_regular(f(*it));
    ++it;
    for (; it != ite; ++it) res = res + f(*it);
    return res;
  }

  template <typename... A>
  requires((nda::clef::Lazy<A> or ...)) auto sum_f_domain_impl(A &&... a) {
    return make_expr_call(
       []<typename... T>(T && ... x)->decltype(auto) { return sum_f_domain_impl(std::forward<T>(x)...); }, std::forward<A>(a)...);
  }

  // sum( expression, i = domain)
  template <typename Expr, int N, typename D>
  decltype(auto) sum(Expr const &f, phvp<N, D> const &d) {
    return sum_f_domain_impl(make_function(f, clef::placeholder<N>()), d.rhs);
  }
  // warning : danger here : if the d is a temporary, the domain MUST be moved in case the Expr
  // is still lazy after eval, or we will obtain a dangling reference.
  template <typename Expr, int N, typename D>
  decltype(auto) sum(Expr const &f, phvp<N, D> &&d) {
    return sum_f_domain_impl(make_function(f, clef::placeholder<N>()), std::move(d.rhs));
  }

  // two or more indices : sum recursively
  template <typename Expr, typename A0, typename A1, typename... A>
  auto sum(Expr const &f, A0 &&a0, A1 &&a1, A &&... a) {
    return sum(sum(f, std::forward<A0>(a0)), std::forward<A1>(a1), std::forward<A>(a)...);
  }
} // namespace nda::clef
