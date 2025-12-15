// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `geqp3` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `geqrf` routine.
   *
   * @details Computes a QR factorization with of a matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A} = \mathbf{Q R}
   * \f]
   * using Level 3 BLAS.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam TAU nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the m-by-n matrix \f$ \mathbf{A} \f$. On exit, On exit, 
   * the elements on and above the diagonal of the array contain the `min(M,N)`-by-N upper trapezoidal matrix R (R is
   * upper triangular if m >= n); the elements below the diagonal, with the array TAU, represent the unitary matrix Q 
   * as a product of min(m,n) elementary reflectors.
   * @param tau Output vector. The scalar factors of the elementary reflectors.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector TAU, MemoryVector W>
    requires(is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU, W> and mem::have_compatible_addr_space<A, TAU, W>)
  int geqrf(A &&a, TAU &&tau, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::geqrf: C order not supported");

    auto [m, n] = a.shape();
    EXPECTS(tau.size() >= std::min(m, n));

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffersize
    int info         = 0;
    using value_type = get_value_t<A>;
    if constexpr (mem::have_device_compatible_addr_space<A, TAU, W>) {
#if defined(NDA_HAVE_DEVICE)
      int buffer_size = device::geqrf_bufferSize(a.extent(0), a.extent(1), a.data(), get_ld(a));
      if (work.size() < buffer_size) work.resize(buffer_size);
      EXPECTS(work.indexmap().min_stride() == 1);
      device::geqrf(a.extent(0), a.extent(1), a.data(), get_ld(a), tau.data(), work.data(), buffer_size, info);
#else
      compile_error_no_gpu();
#endif
    } else {
      value_type bufferSize_T{};
      f77::geqrf(m, n, a.data(), get_ld(a), tau.data(), &bufferSize_T, -1, info);
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      // resize work buffer and perform actual library call
      if (work.size() < bufferSize) work.resize(bufferSize);
      EXPECTS(work.indexmap().min_stride() == 1);
      f77::geqrf(m, n, a.data(), get_ld(a), tau.data(), work.data(), bufferSize, info);
    }
    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::geqrf: info = " << info;
    return info;
  }

  template <MemoryMatrix A, MemoryVector TAU>
    requires(mem::have_compatible_addr_space<A, TAU> and is_blas_lapack_v<get_value_t<A>>)
  auto geqrf(A &&a, TAU &&tau) {
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return geqrf(std::forward<A>(a), std::forward<TAU>(tau), work);
  }

  // batched
  template <MemoryArrayOfRank<3> A, MemoryMatrix TAU, MemoryVector W>
    requires(is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU, W> and mem::have_compatible_addr_space<A, TAU, W>)
  auto geqrf(A &&a, TAU &&tau, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::geqrf: C order not supported");

    if constexpr (has_C_layout<TAU>) return geqrf(std::forward<A>(a), transpose(tau), work);

    auto [m, n, batchSize] = a.shape();
    EXPECTS(tau.extent(0) >= std::min(m, n));
    EXPECTS(tau.extent(1) >= batchSize);

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffersize
    array<int, 1> info(batchSize, 0);
    using value_type = get_value_t<A>;
    if constexpr (mem::have_device_compatible_addr_space<A, TAU, W>) {
#if defined(NDA_HAVE_DEVICE)
      // array of pointers
      array<value_type *, 1> ptr_h(2 * batchSize);
      for (int b = 0; b < batchSize; ++b) {
        ptr_h(b)             = a(range::all, range::all, b).data();
        ptr_h(b + batchSize) = tau(range::all, b).data();
      }
      // copy to device, using container policy from work array
      using handle_t = typename std::decay_t<W>::container_policy_t;
      array<value_type *, 1, C_layout, handle_t> ptr_d(ptr_h);
      // dispatch batched call
      blas::device::geqrf_batched(m, n, ptr_d.data(), a.strides()[1], ptr_d.data() + batchSize, info.data(), batchSize);
#else
      compile_error_no_gpu();
#endif
    } else {
      value_type bufferSize_T{};
      f77::geqrf(m, n, a.data(), a.strides()[1], tau.data(), &bufferSize_T, -1, info(0));
      if (info(0)) NDA_RUNTIME_ERROR << "Error in nda::lapack::geqrf: info = " << info(0);
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      // resize work buffer and perform actual library call
      if (work.size() < bufferSize) work.resize(bufferSize);
      EXPECTS(work.indexmap().min_stride() == 1);
      for (int b = 0; b < batchSize; ++b) {
        auto a_b = a(range::all, range::all, b);
        f77::geqrf(m, n, a_b.data(), get_ld(a_b), tau.data() + b * get_ld(tau), work.data(), bufferSize, info(b));
        if (info(b)) NDA_RUNTIME_ERROR << "Error in nda::lapack::geqrf: info = " << info(b);
      }
    }
    return info;
  }

  template <MemoryArrayOfRank<3> A, MemoryMatrix TAU>
    requires(mem::have_compatible_addr_space<A, TAU> and is_blas_lapack_v<get_value_t<A>>)
  auto geqrf(A &&a, TAU &&tau) {
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return geqrf(std::forward<A>(a), std::forward<TAU>(tau), work);
  }

} // namespace nda::lapack
