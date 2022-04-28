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
// Authors: Nils Wentzell

#pragma once
#include "../layout/policies.hpp"
#include "../basic_array.hpp"
#include "../blas/dot.hpp"

namespace nda {

  /**
   * @tparam X
   * @tparam Y
   * @param x : lhs
   * @param y : rhs
   * @return the dot-product
   *   Implementation varies 
   */
  template <typename X, typename Y>
  auto dot(X &&l, Y &&r) {
    static constexpr auto L_adr_spc = mem::get_addr_space<X>;
    static constexpr auto R_adr_spc = mem::get_addr_space<Y>;
    static_assert(L_adr_spc != mem::None);
    static_assert(R_adr_spc != mem::None);

    using promoted_type = decltype(get_value_t<X>{} * get_value_t<Y>{});
    using vector_t      = basic_array<promoted_type, 1, C_layout, 'V', nda::heap<mem::combine<L_adr_spc, R_adr_spc>>>;

    if constexpr (is_blas_lapack_v<promoted_type>) {

      auto as_container = []<typename A>(A const &a) -> decltype(auto) {
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, promoted_type>)
          return a;
        else
          return vector_t{a};
      };

      return blas::dot(as_container(l), as_container(r));
    } else {
      return blas::dot_generic(l, r);
    }
  }

  /**
   * @tparam X
   * @tparam Y
   * @param x : lhs
   * @param y : rhs
   * @return The dotc-product
   *   Implementation varies 
   */
  template <typename X, typename Y>
  auto dotc(X &&l, Y &&r) {
    static constexpr auto L_adr_spc = mem::get_addr_space<X>;
    static constexpr auto R_adr_spc = mem::get_addr_space<Y>;
    static_assert(L_adr_spc != mem::None);
    static_assert(R_adr_spc != mem::None);

    using promoted_type = decltype(get_value_t<X>{} * get_value_t<Y>{});
    using vector_t      = basic_array<promoted_type, 1, C_layout, 'V', nda::heap<mem::combine<L_adr_spc, R_adr_spc>>>;

    if constexpr (is_blas_lapack_v<promoted_type>) {

      auto as_container = []<typename A>(A const &a) -> decltype(auto) {
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, promoted_type>)
          return a;
        else
          return vector_t{a};
      };

      return blas::dotc(as_container(l), as_container(r));
    } else {
      return blas::dotc_generic(l, r);
    }
  }

} // namespace nda
