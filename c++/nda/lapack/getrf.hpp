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
   * LU decomposition of a matrix
   *
   * The matrix m is modified during the operation.
   * The matrix is interpreted as FORTRAN ordered, and then LU decomposed.
   * NB : for some operation, like det, inversion, it is fine to be transposed, 
   *      for some it may not be ... 
   *
   * @tparam M matrix, matrix_view, array, array_view of rank 2. M can be a temporary view
   * @param m  matrix to be LU decomposed. It is destroyed by the operation
   * @param ipiv  Gauss Pivot, cf lapack doc
   *
   */
  template <MemoryMatrix M, MemoryVector IPIV>
  [[nodiscard]] int getrf(M &&m, IPIV &ipiv) {
    static_assert(is_blas_lapack_v<get_value_t<M>>, "Matrix must have elements of type double or complex");
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Pivoting array must have elements of type int");

    if (!m.is_contiguous() or !ipiv.is_contiguous()) NDA_RUNTIME_ERROR << "Lapack routines require arrays with contiguous data";

    auto dm = std::min(m.extent(0), m.extent(1));
    if (ipiv.size() < dm) ipiv.resize(dm);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    ipiv = 0;
#endif
#endif

    int info = 0;
    f77::getrf(get_n_rows(m), get_n_cols(m), m.data(), get_ld(m), ipiv.data(), info);
    return info;
  }

} // namespace nda::lapack
