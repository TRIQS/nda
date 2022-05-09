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
   * Computes the inverse of an LU-factored general matrix.
   * The matrix m is modified during the operation.
   * The matrix is interpreted as FORTRAN orderedf getrf.
   *
   * @tparam M matrix, matrix_view, array, array_view of rank 2. M can be a temporary view
   * @param m  matrix to be LU decomposed. It is destroyed by the operation
   * @param ipiv  Gauss Pivot, cf lapack doc
   *
   */
  template <typename M, char Algebra, typename CP>
  [[nodiscard]] int getri(M &&m, basic_array<int, 1, C_layout, Algebra, CP> &ipiv) {
    static_assert(is_regular_or_view_v<M>, "getrf: M must be a matrix, matrix_view, array or array_view of rank 2");
    static_assert(std::decay_t<M>::rank == 2, "M must be of rank 2");
    static_assert(is_blas_lapack_v<get_value_t<M>>, "Matrices must have the same element type and it must be double, complex ...");

    EXPECTS(ipiv.size() >= std::min(m.extent(0), m.extent(1)));

    using T  = typename std::decay_t<M>::value_type;
    int info = 0;
    std::array<T, 2> work1{0, 0}; // always init for MSAN and clang-tidy ...

    // first call to get the optimal lwork
    f77::getri(get_n_rows(m), m.data(), get_ld(m), ipiv.data(), work1.data(), -1, info);
    int lwork;
    if constexpr (is_complex_v<T>)
      lwork = std::round(std::real(work1[0])) + 1;
    else
      lwork = std::round(work1[0]) + 1;

    array<T, 1> work(lwork);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    work = 0;
#endif
#endif

    // second call to do the job
    f77::getri(get_n_rows(m), m.data(), get_ld(m), ipiv.data(), work.data(), lwork, info);
    return info;
  }

} // namespace nda::lapack
