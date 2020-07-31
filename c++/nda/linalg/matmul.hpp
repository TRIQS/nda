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
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    EXPECTS_WITH_MESSAGE(l.shape()[1] == r.shape()[0], "Matrix product : dimension mismatch in matrix product " << l << " " << r);

    using promoted_type = decltype(get_value_t<L_t>{} * get_value_t<R_t>{});
    matrix<promoted_type> result(l.shape()[0], r.shape()[1]);

    if constexpr (blas::is_blas_lapack_v<promoted_type>) {

      auto as_container = [](auto const &a) -> decltype(auto) {
        //FIXMEM C++20 LAMBDA
        using A = std::decay_t<decltype(a)>;
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, promoted_type>)
          return a;
        else
          return matrix<promoted_type>{a};
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
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    EXPECTS_WITH_MESSAGE(l.shape()[1] == r.shape()[0], "Matrix Vector product : dimension mismatch in matrix product " << l << " " << r);

    using promoted_type = decltype(get_value_t<L_t>{} * get_value_t<R_t>{});
    array<promoted_type, 1> result(l.shape()[0]);

    if constexpr (blas::is_blas_lapack_v<promoted_type>) {

      auto as_container = [](auto const &a) -> decltype(auto) {
        //FIXMEM C++20 LAMBDA
        using A = std::decay_t<decltype(a)>;
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, promoted_type>)
          return a;
        else
          return array<promoted_type, get_rank<A>>{a};
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
