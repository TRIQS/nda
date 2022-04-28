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
#include "../layout/policies.hpp"
#include "../basic_array.hpp"
#include "../blas/gemm.hpp"
#include "../blas/gemv.hpp"

namespace nda {

  /**
   * @tparam L NdArray with algebra 'M' 
   * @tparam R 
   * @param l : lhs
   * @param r : rhs
   * @return the matrix multiplication
   *   Implementation varies 
   */

  template <typename L, typename R>
  auto matmul(L &&l, R &&r) {
    EXPECTS_WITH_MESSAGE(l.shape()[1] == r.shape()[0], "Matrix product : dimension mismatch in matrix product " << l << " " << r);

    static constexpr auto L_adr_spc = mem::get_addr_space<L>;
    static constexpr auto R_adr_spc = mem::get_addr_space<R>;
    static_assert(L_adr_spc != mem::None);
    static_assert(R_adr_spc != mem::None);

    using promoted_type = decltype(get_value_t<L>{} * get_value_t<R>{});
    using matrix_t      = basic_array<promoted_type, 2, C_layout /*FIXME*/, 'M', nda::heap<mem::combine<L_adr_spc, R_adr_spc>>>;
    auto result         = matrix_t(l.shape()[0], r.shape()[1]);

    if constexpr (is_blas_lapack_v<promoted_type>) {

      auto as_container = []<typename A>(A const &a) -> decltype(auto) {
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, promoted_type>)
          return a;
        else
          return matrix_t{a};
      };

      // MSAN has no way to know that we are calling with beta = 0, hence
      // this is not necessaru
      // of course, in production code, we do NOT waste time to do this.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      result = 0;
#endif
#endif

      blas::gemm(1, as_container(l), as_container(r), 0, result);
    } else {
      blas::gemm_generic(1, l, r, 0, result);
    }
    return result;
  }

  /**
   * @tparam L NdArray with algebra 'M' 
   * @tparam R 
   * @param l : lhs
   * @param r : rhs
   * @return the matrix multiplication
   *   Implementation varies 
   */

  template <typename L, typename R>
  auto matvecmul(L &&l, R &&r) {
    EXPECTS_WITH_MESSAGE(l.shape()[1] == r.shape()[0], "Matrix Vector product : dimension mismatch in matrix product " << l << " " << r);

    static constexpr auto L_adr_spc = mem::get_addr_space<L>;
    static constexpr auto R_adr_spc = mem::get_addr_space<R>;
    static_assert(L_adr_spc != mem::None);
    static_assert(R_adr_spc != mem::None);

    using promoted_type = decltype(get_value_t<L>{} * get_value_t<R>{});
    vector<promoted_type, heap<L_adr_spc>> result(l.shape()[0]);

    if constexpr (is_blas_lapack_v<promoted_type>) {

      auto as_container = []<typename A>(A const &a) -> decltype(auto) {
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, promoted_type>)
          return a;
        else
          return basic_array<promoted_type, get_rank<A>, C_layout, 'A', heap<L_adr_spc>>{a};
      };

      // MSAN has no way to know that we are calling with beta = 0, hence
      // this is not necessaru
      // of course, in production code, we do NOT waste time to do this.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      result = 0;
#endif
#endif

      blas::gemv(1, as_container(l), as_container(r), 0, result);
    } else {
      blas::gemv_generic(1, l, r, 0, result);
    }
    return result;
  }

} // namespace nda
