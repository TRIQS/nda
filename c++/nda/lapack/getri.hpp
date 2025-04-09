// Copyright (c) 2021-2024 Simons Foundation
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
// Authors: Thomas Hahn, Miguel Morales, Nils Wentzell

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `getri` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif // NDA_HAVE_DEVICE

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getri` routine.
   *
   * @details Computes the inverse of a matrix using the LU factorization computed by `getrf`.
   *
   * This method inverts \f$ \mathbf{U} \f$ and then computes \f$ \mathrm{inv}(\mathbf{A}) \f$ by solving the system
   * \f$ \mathrm{inv}(\mathbf{A}) L = \mathrm{inv}(\mathbf{U}) \f$ for \f$ \mathrm{inv}(\mathbf{A}) \f$.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the
   * factorization \f$ \mathbf{A} = \mathbf{P L U} \f$ as computed by `getrf`. On exit, if `INFO == 0`, the inverse of
   * the original matrix \f$ \mathbf{A} \f$.
   * @param ipiv Input vector. The pivot indices from `getrf`, i.e. for `1 <= i <= N`, row i of the matrix was
   * interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector IPIV>
    requires(mem::have_compatible_addr_space<A, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getri(A &&a, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getri: Pivoting array must have elements of type int");

    // check the dimensions of the input/output arrays/views and resize if necessary
    EXPECTS(a.extent(0) == a.extent(1));
    EXPECTS(ipiv.size() >= a.extent(0));

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    // perform the LAPACK calls
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      device::getri(a.extent(0), a.data(), get_ld(a), ipiv.data(), NULL, 0, info);
#else
      compile_error_no_gpu();
#endif
    } else {
      // first call to get the optimal buffersize
      using value_type = get_value_t<A>;
      value_type bufferSize_T{};
      f77::getri(a.extent(0), a.data(), get_ld(a), ipiv.data(), &bufferSize_T, -1, info);
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      // allocate work buffer and perform actual library call
      array<value_type, 1> work(bufferSize);
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
