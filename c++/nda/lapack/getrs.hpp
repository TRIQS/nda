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
#include "../blas/tools.hpp"

namespace nda::lapack {

  /**
   * Solves a system of linear equations
   *    A * X = B,  A**T * X = B,  or  A**H * X = B
   * with a general N-by-N matrix A using the LU factorization computed
   * by getrf.
   *
   * [in]      a is real/complex array, dimension (LDA,N)
   *           The factors L and U from the factorization A = P*L*U
   *           as computed by ZGETRF.
   *
   * [in]      ipiv is INTEGER array, dimension (N)
   *           The pivot indices from ZGETRF; for 1<=i<=N, row i of the
   *           matrix was interchanged with row ipiv(i).
   *
   * [in,out]  b is real/complex array, dimension (LDB,NRHS)
   *           On entry, the right hand side matrix B.
   *           On exit, the solution matrix X.
   *
   * [return]  info is INTEGER
   *           = 0:  successful exit
   *           < 0:  if info = -i, the i-th argument had an illegal value
   */
  template <MemoryMatrix A, MemoryMatrix B, MemoryVector IPIV>
    requires(have_same_value_type_v<A, B> and mem::have_same_addr_space_v<A, B, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrs(A const &a, B &b, IPIV const &ipiv) {
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Pivoting array must have elements of type int");
    EXPECTS(ipiv.size() >= std::min(a.extent(0), a.extent(1)));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    static constexpr bool conj_A = blas::is_conj_array_expr<A>;
    char op_a                    = blas::get_op<conj_A, /*transpose =*/has_C_layout<A>>;

    int info = 0;
    if constexpr (mem::on_host<A>) {
      f77::getrs(op_a, a.extent(1), b.extent(1), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
    } else {
      cuda::getrs(op_a, a.extent(1), b.extent(1), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
    }
    return info;
  }
} // namespace nda::lapack
