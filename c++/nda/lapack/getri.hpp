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
   * Computes the inverse of a matrix using the LU factorization
   * computed by getrf.
   *
   * This method inverts U and then computes inv(A) by solving the system
   * inv(A)*L = inv(U) for inv(A).
   *
   * [in,out]  a is real/complex array, dimension (LDA,N)
   *           On entry, the factors L and U from the factorization
   *           A = P*L*U as computed by ZGETRF.
   *           On exit, if info = 0, the inverse of the original matrix A.
   *
   * [in]      ipiv, integer array of dimension (N)
   *           The pivot indices from ZGETRF; for 1<=i<=N, row i of the
   *           matrix was interchanged with row ipiv(i).
   *
   * [return]  info is INTEGER
   *           = 0:  successful exit
   *           < 0:  if info = -i, the i-th argument had an illegal value
   *           > 0:  if info = i, U(i,i) is exactly zero; the matrix is
   *                 singular and its inverse could not be computed.
   *
   */
  template <MemoryMatrix A, MemoryVector IPIV>
  requires(mem::have_compatible_addr_space_v<A, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getri(A &&a, IPIV const &ipiv) {
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Pivoting array must have elements of type int");

    using T  = get_value_t<A>;
    auto dm = std::min(a.extent(0), a.extent(1));
    if (ipiv.size() < dm) NDA_RUNTIME_ERROR << "In getri: Pivot index array size " << ipiv.size() << " smaller than required size " << dm;

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space_v<A,IPIV>) {
      device:getri(a.extent(0), a.data(), get_ld(a), ipiv.data(), NULL, 0, info);
    } else {
      // First call to get the optimal buffersize
      T bufferSize_T{};
      f77::getri(a.extent(0), a.data(), get_ld(a), ipiv.data(), &bufferSize_T, -1, info);
      int bufferSize = std::ceil(std::real(bufferSize_T));

      // Allocate work buffer and perform actual library call
      array<T, 1> work(bufferSize);
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      work = 0;
#endif
#endif
      f77::getri(a.extent(0), a.data(), get_ld(a), ipiv.data(), work.data(), bufferSize, info);
    }
    return info;
  }

} // namespace nda::lapack
