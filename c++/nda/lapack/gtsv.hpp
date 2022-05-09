// Copyright (c) 2021 Simons Foundation
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

#include "../lapack.hpp"

namespace nda::lapack {

  /**
   * Computes the solution to the system of linear equations with a tridiagonal coefficient matrix A and multiple right-hand sides.
   *
   * The routine solves for X the system of linear equations A*X = B, where A is an n-by-n tridiagonal matrix.
   * The columns of matrix B are individual right-hand sides, and the columns of X are the corresponding solutions. 
   *
   * @tparam T Element type
   * @param dl
   * @param d
   * @param du
   * @param b 
   */
  template <typename V1, typename V2, typename V3, typename M>
  [[nodiscard]] int gtsv(V1 &dl, V2 &d, V3 &du, M &b) {

    static_assert(is_regular_or_view_v<V1> and (V1::rank == 1), "gtsv: V1 must be an array/view of rank 1");
    static_assert(is_regular_or_view_v<V2> and (V2::rank == 1), "gtsv: V2 must be an array/view of rank 1");
    static_assert(is_regular_or_view_v<V3> and (V3::rank == 1), "gtsv: V3 must be an array/view of rank 1");
    static_assert(is_regular_or_view_v<M> and (M::rank == 1 or M::rank == 2), "gtsv: M must be an matrix/array/view of rank  1 or 2");
    static_assert(have_same_value_type_v<V1, V2, V3, M>, "Arrays must have the same value-type");
    static_assert(is_double_or_complex_v<get_value_t<V1>>, "Arrays must have value-type double or complex");

    int N    = d.extent(0);
    int NRHS = (M::rank == 2 ? b.extent(1) : 1);
    EXPECTS(dl.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between sub-diagonal and diagonal vectors "
    EXPECTS(du.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between super-diagonal and diagonal vectors "
    EXPECTS(b.extent(0) == d.extent(0));      // "gtsv : dimension mismatch between diagonal vector and RHS matrix, "

    int info = 0;
    f77::gtsv(N, NRHS, dl.data(), d.data(), du.data(), b.data(), N, info);
    return info;
  }
} // namespace nda::lapack
