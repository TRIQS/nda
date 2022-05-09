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
   * getrs solves a system of linear equations
   *      A * X = B
   * with a general N-by-N matrix A using the LU factorization computed by getrf.
   *
   * @tparam A Contiguous array of rank 2 and shape (LDA,N)
   * @tparam B Contiguous array of rank 2 and shape (LDB,NRHS)
   * @param ipiv Integer array of shape (N) containing the pivot indices from getrf
   */
  template <ArrayOfRank<2> A, ArrayOfRank<2> B, ArrayOfRank<1> IPIV>
  [[nodiscard]] int getrs(A &&a, B &b, IPIV &ipiv) {
    static_assert(std::is_same_v<get_value_t<A>, get_value_t<B>>, "Matrices must have the same element type");
    static_assert(is_blas_lapack_v<get_value_t<A>>, "Matrices must have elements of type double or complex");
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Pivoting array must have elements of type int");

    if (!a.is_contiguous() or !b.is_contiguous() or !ipiv.is_contiguous()) NDA_RUNTIME_ERROR << "Lapack routines require arrays with contiguous data";

    EXPECTS(ipiv.size() >= std::min(a.extent(0), a.extent(1)));

    int info = 0;
    f77::getrs('N', get_n_cols(a), get_n_cols(b), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
    return info;
  }
} // namespace nda::lapack
