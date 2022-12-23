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
   * Computes the singular value decomposition (SVD) of a real/complex
   * M-by-N matrix A, computing the left and/or right singular
   * vectors. The SVD is written
   *
   *      A = U * SIGMA * conjugate-transpose(V)
   *
   * where SIGMA is an M-by-N matrix which is zero except for its
   * min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
   * V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
   * are the singular values of A; they are real and non-negative, and
   * are returned in descending order.  The first min(m,n) columns of
   * U and V are the left and right singular vectors of A.
   *
   * Note that the routine calculates V**H, not V.
   *
   * [in,out]  A is real/complex array, dimension (LDA,N)
   *           On entry, the M-by-N matrix A.
   *           On exit,
   *
   * [out]     S is DOUBLE PRECISION array, dimension (min(M,N))
   *           The singular values of A, sorted so that S(i) >= S(i+1).
   *
   * [out]     U is real/complex array, dimension (LDU,M).
   *           U contains the M-by-M unitary matrix U;
   *
   * [out]     VT is real/complex array, dimension (LDVT,N)
   *           VT contains the N-by-N unitary matrix V**H
   *
   * [return]  INFO is INTEGER
   *           = 0:  successful exit.
   *           < 0:  if INFO = -i, the i-th argument had an illegal value.
   *           > 0:  if ZBDSQR did not converge, INFO specifies how many
   *                 superdiagonals of an intermediate bidiagonal form B
   *                 did not converge to zero. See the description of RWORK
   *                 above for details.
   */
  template <MemoryMatrix A, MemoryVector S, MemoryMatrix U, MemoryMatrix VT>
  requires(have_same_value_type_v<A, U, VT> and mem::have_same_addr_space_v<A, S, U, VT> and is_blas_lapack_v<get_value_t<A>>)
  int gesvd(A &&a, S &&s, U &&u, VT &&vt) {
    static_assert(has_F_layout<A> and has_F_layout<U> and has_F_layout<VT>, "C order not implemented");

    using T    = get_value_t<A>;
    auto dm    = std::min(a.extent(0), a.extent(1));
    auto rwork = array<double, 1, C_layout, heap<mem::get_addr_space<A>>>(5 * dm);
    if (s.size() < dm) s.resize(dm);

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(s.indexmap().min_stride() == 1);
    EXPECTS(u.indexmap().min_stride() == 1);
    EXPECTS(vt.indexmap().min_stride() == 1);

    // Call host/device implementation depending on input
    auto gesvd = []<typename... Ts>(Ts && ...args) {
      if constexpr (mem::on_host<A>) {
        lapack::f77::gesvd(std::forward<Ts>(args)...);
      } else {
#if defined(NDA_HAVE_DEVICE)
        lapack::device::gesvd(std::forward<Ts>(args)...);
#else
        static_assert(always_false<bool>," lapack on device without gpu support! Compile for GPU. ");
#endif
      }
    };

    // First call to get the optimal buffersize
    T bufferSize_T{};
    int info   = 0;
    gesvd('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), s.data(), u.data(), get_ld(u), vt.data(), get_ld(vt), &bufferSize_T, -1,
          rwork.data(), info);
    int bufferSize = std::ceil(std::real(bufferSize_T));

    // Allocate work buffer and perform actual library call
    array<T, 1, C_layout, heap<mem::get_addr_space<A>>> work(bufferSize);
    gesvd('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), s.data(), u.data(), get_ld(u), vt.data(), get_ld(vt), work.data(), bufferSize,
          rwork.data(), info);

    if (info) NDA_RUNTIME_ERROR << "Error in gesvd : info = " << info;
    return info;
  }

} // namespace nda::lapack
