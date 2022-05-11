// Copyright (c) 2020-2021 Simons Foundation
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

#include "../lapack.hpp"

namespace nda::lapack {

  /**
   * Computes the minimum norm solution to a real/complex linear
   * least squares problem:
   *
   * Minimize 2-norm(| b - A*x |).
   *
   * using the singular value decomposition (SVD) of A. A is an M-by-N
   * matrix which may be rank-deficient.
   *
   * Several right hand side vectors b and solution vectors x can be
   * handled in a single call; they are stored as the columns of the
   * M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix
   * X.
   *
   * The effective rank of A is determined by treating as zero those
   * singular values which are less than RCOND times the largest singular
   * value.
   *
   * [in,out]  a is real/complex array, dimension (LDA,N)
   *           On entry, the M-by-N matrix A.
   *           On exit, the first min(m,n) rows of A are overwritten with
   *           its right singular vectors, stored rowwise.
   *
   * [in,out]  b is real/complex array, dimension (LDB,NRHS)
   *           On entry, the M-by-NRHS right hand side matrix B.
   *           On exit, B is overwritten by the N-by-NRHS solution matrix X.
   *           If m >= n and RANK = n, the residual sum-of-squares for
   *           the solution in the i-th column is given by the sum of
   *           squares of the modulus of elements n+1:m in that column.
   *
   * [out]     s is a double array, dimension (min(M,N))
   *           The singular values of A in decreasing order.
   *           The condition number of A in the 2-norm = S(1)/S(min(m,n)).
   *
   * [in]      rcond is a double used to determine the effective rank of A.
   *           Singular values s(i) <= rcond*s(1) are treated as zero.
   *           If rcond < 0, machine precision is used instead.
   *
   * [out]     The effective rank of A, i.e., the number of singular values
   *           which are greater than RCOND*S(1).
   *
   * [return]  info is INTEGER
   *           = 0:  successful exit
   *           < 0:  if INFO = -i, the i-th argument had an illegal value.
   *           > 0:  the algorithm for computing the SVD failed to converge;
   *                 if INFO = i, i off-diagonal elements of an intermediate
   *                 bidiagonal form did not converge to zero.
   */
  template <MemoryMatrix A, MemoryMatrix B, MemoryVector S>
  requires(have_same_value_type_v<A, B> and mem::on_host<A, B, S> and is_blas_lapack_v<get_value_t<A>>)
  int gelss(A &a, B &b, S &s, double rcond, int &rank) {
    static_assert(has_F_layout<A> and has_F_layout<B>, "C order not implemented");

    using T = get_value_t<A>;
    auto dm    = std::min(a.extent(0), a.extent(1));
    auto rwork = array<double, 1>(5 * dm);
    if (s.size() < dm) s.resize(dm);

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(s.indexmap().min_stride() == 1);

    // First call to get the optimal bufferSize
    T bufferSize_T{};
    int info = 0;
    f77::gelss(a.extent(0), a.extent(1), b.extent(1), a.data(), get_ld(a), b.data(), get_ld(b), s.data(), rcond, rank, &bufferSize_T, -1,
               rwork.data(), info);
    int bufferSize = std::ceil(std::real(bufferSize_T));

    // Allocate work buffer and perform actual library call
    array<T, 1> work(bufferSize);
    f77::gelss(a.extent(0), a.extent(1), b.extent(1), a.data(), get_ld(a), b.data(), get_ld(b), s.data(), rcond, rank, work.data(), bufferSize,
               rwork.data(), info);

    if (info) NDA_RUNTIME_ERROR << "Error in gesvd : info = " << info;
    return info;
  }

} // namespace nda::lapack
