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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include "../lapack.hpp"

namespace nda::lapack {

  /**
   * Computes an LU factorization of a general M-by-N matrix A
   * using partial pivoting with row interchanges.
   *
   * The factorization has the form
   *    A = P * L * U
   * where P is a permutation matrix, L is lower triangular with unit
   * diagonal elements (lower trapezoidal if m > n), and U is upper
   * triangular (upper trapezoidal if m < n).
   *
   * This is the right-looking Level 3 BLAS version of the algorithm.
   *
   * [in,out]  a is real/complex array, dimension (LDA,N)
   *           On entry, the M-by-N matrix to be factored.
   *           On exit, the factors L and U from the factorization
   *           a = P*l*u; the unit diagonal elements of L are not stored.
   *
   * [out]     ipiv is INTEGER array, dimension (min(M,N))
   *           The pivot indices; for 1 <= i <= min(M,N), row i of the
   *           matrix was interchanged with row ipiv(i).
   *
   * [return]  info is INTEGER
   *           = 0:  successful exit
   *           < 0:  if info = -i, the i-th argument had an illegal value
   *           > 0:  if info = i, U(i,i) is exactly zero. The factorization
   *                 has been completed, but the factor U is exactly
   *                 singular, and division by zero will occur if it is used
   *                 to solve a system of equations.
   */
  template <MemoryMatrix A, MemoryVector IPIV>
  requires(mem::have_same_addr_space_v<A, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrf(A &&a, IPIV &&ipiv) {
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Pivoting array must have elements of type int");

    auto dm = std::min(a.extent(0), a.extent(1));
    if (ipiv.size() < dm) ipiv.resize(dm);

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    ipiv = 0;
#endif
#endif

    int info = 0;
    if constexpr (mem::on_host<A>) {
      f77::getrf(a.extent(0), a.extent(1), a.data(), get_ld(a), ipiv.data(), info);
    } else {
#if defined(NDA_HAVE_DEVICE)
      device::getrf(a.extent(0), a.extent(1), a.data(), get_ld(a), ipiv.data(), info);
#else
      static_assert(always_false<bool>," lapack on device without gpu support! Compile for GPU. ");
#endif
    }
    return info;
  }

} // namespace nda::lapack
