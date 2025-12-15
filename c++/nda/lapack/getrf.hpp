// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `getrf` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"
#if defined(NDA_HAVE_DEVICE)
#include "../blas/interface/cxx_interface.hpp" // batched getrf is in cublas
#endif

#include <algorithm>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getrf` routine.
   *
   * @details Computes an LU factorization of a general m-by-n matrix \f$ \mathbf{A} \f$ using partial pivoting with row
   * interchanges.
   *
   * The factorization has the form
   * \f[
   *   \mathbf{A} = \mathbf{P L U}
   * \f]
   * where \f$ \mathbf{P} \f$ is a permutation matrix, \f$ \mathbf{L} \f$ is lower triangular with unit diagonal
   * elements (lower trapezoidal if `m > n`), and \f$ \mathbf{U} \f$ is upper triangular (upper trapezoidal if `m < n`).
   *
   * This is the right-looking Level 3 BLAS version of the algorithm.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the m-by-n matrix to be factored. On exit, the factors \f$ \mathbf{L} \f$
   * and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A} = \mathbf{P L U} \f$; the unit diagonal elements of
   * \f$ \mathbf{L} \f$ are not stored.
   * @param ipiv Output vector. The pivot indices from `getrf`, i.e. for `1 <= i <= n`, row i of the matrix was
   * interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector IPIV, MemoryVector W>
    requires(mem::have_compatible_addr_space<A, IPIV, W> and is_blas_lapack_v<get_value_t<A>> and std::is_same_v<get_value_t<A>, get_value_t<W>>)
  int getrf(A &&a, IPIV &&ipiv, [[maybe_unused]] W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrf: Pivoting array must have elements of type int");

    auto dm = std::min(a.extent(0), a.extent(1));
    if (ipiv.size() < dm) ipiv.resize(dm); // ipiv needs to be a regular array?

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    ipiv = 0;
#endif
#endif

    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      int buffer_size = device::getrf_bufferSize(a.extent(0), a.extent(1), a.data(), get_ld(a));
      if (work.size() < buffer_size) work.resize(buffer_size);
      device::getrf(a.extent(0), a.extent(1), a.data(), get_ld(a), work.data(), ipiv.data(), info);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::getrf(a.extent(0), a.extent(1), a.data(), get_ld(a), ipiv.data(), info);
    }
    return info;
  }

  template <MemoryMatrix A, MemoryVector IPIV>
    requires(mem::have_compatible_addr_space<A, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrf(A &&a, IPIV &&ipiv) {
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return getrf(std::forward<A>(a), std::forward<IPIV>(ipiv), work);
  }

  /// The slowest stride is always the batched one
  template <MemoryArrayOfRank<3> A, MemoryMatrix IPIV, MemoryVector W>
    requires(mem::have_compatible_addr_space<A, IPIV, W> and is_blas_lapack_v<get_value_t<A>> and std::is_same_v<get_value_t<A>, get_value_t<W>>)
  auto getrf(A &&a, IPIV &&ipiv, [[maybe_unused]] W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrf: Pivoting array must have elements of type int");
    static_assert(has_F_layout<A> or has_C_layout<A>, "Error in nda::lapack::getrf: Only C or Fortran layout allowed.");

    if constexpr (has_C_layout<A>) return getrf(transpose(a), ipiv, work);

    auto batchSize = a.extent(2);
    auto dm        = std::min(a.extent(0), a.extent(1));
    if constexpr (has_F_layout<IPIV>) {
      if (ipiv.extent(0) < dm or ipiv.extent(1) < batchSize) { ipiv.resize(dm, batchSize); }
    } else {
      if (ipiv.extent(1) < dm or ipiv.extent(0) < batchSize) { ipiv.resize(batchSize, dm); }
    }

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    ipiv = 0;
#endif
#endif

    array<int, 1> info(batchSize, 0);
    if constexpr (mem::have_device_compatible_addr_space<A, IPIV, W>) {
#if defined(NDA_HAVE_DEVICE)
      if (a.extent(0) == a.extent(1)) {
        using value_type = get_value_t<A>;
        // array of pointers
        array<value_type *, 1> ptr_h(batchSize);
        for (int b = 0; b < batchSize; ++b) ptr_h(b) = a(range::all, range::all, b).data();
        // copy to device, using container policy from work array
        using handle_t = typename std::decay_t<W>::container_policy_t;
        array<value_type *, 1, C_layout, handle_t> ptr_d(ptr_h);
        array<int, 1, C_layout, handle_t> info_d(batchSize, 0);
        // dispatch batched call
        blas::device::getrf_batched(a.extent(0), ptr_d.data(), a.strides()[1], ipiv.data(), info_d.data(), batchSize);
        info() = info_d();
      } else {
        int buffer_size = device::getrf_bufferSize(a.extent(0), a.extent(1), a.data(), a.strides()[1]);
        if (work.size() < buffer_size) work.resize(buffer_size);
        for (int b = 0; b < batchSize; ++b) {
          auto a_b = a(range::all, range::all, b);
          device::getrf(a_b.extent(0), a_b.extent(1), a_b.data(), get_ld(a_b), work.data(), ipiv.data() + b * get_ld(ipiv), info(b));
        }
      }
#else
      compile_error_no_gpu();
#endif
    } else {
      // Not using OpenMP yet, can use #pragma here!
      for (int b = 0; b < batchSize; ++b) {
        auto a_b = a(range::all, range::all, b);
        f77::getrf(a_b.extent(0), a_b.extent(1), a_b.data(), get_ld(a_b), ipiv.data() + b * get_ld(ipiv), info(b));
      }
    }
    return info;
  }

  template <MemoryArrayOfRank<3> A, MemoryMatrix IPIV>
    requires(mem::have_compatible_addr_space<A, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  auto getrf(A &&a, IPIV &&ipiv) {
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return getrf(std::forward<A>(a), std::forward<IPIV>(ipiv), work);
  }

} // namespace nda::lapack
