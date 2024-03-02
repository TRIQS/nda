// Copyright (c) 2020-2022 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell, Jason Kaye

#pragma once

#include "../lapack.hpp"
#include "nda/concepts.hpp"

namespace nda::lapack {

  /**
   * Computes a QR factorization with column pivoting of an M-by-N matrix A:
   *
   * A*P = Q*R
   *
   * using Level 3 BLAS.
   *
   * [in,out]  a is real/complex array, dimension (LDA,N)
   *           On entry, the M-by-N matrix A.  On exit, the upper triangle of a
   *           contains the min(M,N)-by-N upper triangular matrix R; the
   *           elements below the diagonal, together with the length min(M,N)
   *           vector tau, represent the unitary matrix Q as a product of
   *           min(M,N) elementary reflectors.
   *
   * [in,out]  jpvt is integer array, dimension (N)
   *           On entry, if jpvt(j).ne.0, the j-th column of A is permuted to
   *           the front of A*P (a leading column); if jpvt(j)=0, the j-th
   *           column of A is a free column.  On exit, if JPVT(j)=k, then the
   *           j-th column of A*P was the the k-th column of A.
   *
   * [out]     tau is a real/complex array, dimension (min(M,N))
   *           The scalar factors of the elementary reflectors.
   *
   * [return]  info is INTEGER
   *           = 0: successful exit. 
   *           < 0: if INFO = -i, the i-th argument had an illegal value.
   *
   * @note If one wishes to carry out the column pivoted QR algorithm, the array
   * \p jpvt must be initialized to zero; see the explanation above of the
   * argument jpvt.
   */
  template <MemoryMatrix A, MemoryVector JPVT, MemoryVector TAU>
    requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU>
             and mem::have_compatible_addr_space<A, JPVT, TAU>)
  int geqp3(A &&a, JPVT &&jpvt, TAU &&tau) {
    static_assert(has_F_layout<A>, "C order not implemented");
    static_assert(std::is_same_v<get_value_t<JPVT>, int>, "Pivoting array must have elements of type int");
    static_assert(mem::have_host_compatible_addr_space<A, JPVT, TAU>,
                  "geqp3 is only implemented on the CPU, but was provided non-host compatible array");

    using T     = get_value_t<A>;
    auto [m, n] = a.shape();
    auto rwork  = array<double, 1>(2 * n);
    EXPECTS(tau.size() >= std::min(m, n));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(jpvt.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // First call to get the optimal buffersize
    T bufferSize_T{};
    int info = 0;
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), &bufferSize_T, -1, rwork.data(), info);
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // Allocate work buffer and perform actual library call
    nda::array<T, 1, C_layout, heap<mem::get_addr_space<A>>> work(bufferSize);
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), work.data(), bufferSize, rwork.data(), info);

    if (info) NDA_RUNTIME_ERROR << "Error in geqp3 : info = " << info;
    return info;
  }

} // namespace nda::lapack
