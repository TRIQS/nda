// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `orgqr` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#include <cmath>
#include <complex>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `orgqr` routine.
   *
   * @details Generates an m-by-n real matrix \f$ \mathbf{Q} \f$ with orthonormal columns, which is defined as the first
   * n columns of a product of k elementary reflectors of order m:
   * \f[
   *   \mathbf{Q} = \mathbf{H}(1) \mathbf{H}(2) \ldots \mathbf{H}(k) \; ,
   * \f]
   * as returned by `geqrf`.
   *
   * @tparam A nda::MemoryMatrix with float/double value type.
   * @tparam TAU nda::MemoryVector with float/double value type.
   * @param a Input/output matrix. On entry, the i-th column must contain the vector which defines the elementary
   * reflector \f$ H(i) \; , i = 1,2,...,k \f$, as returned by `geqrf` in the first k columns. On exit, the m-by-n
   * matrix \f$ \mathbf{Q} \f$.
   * @param tau Input vector. `tau(i)` must contain the scalar factor of the elementary reflector \f$ \mathbf{H}(i) \f$,
   * as returned by `geqrf`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector TAU, MemoryVector W>
    requires((std::is_same_v<double, get_value_t<A>> or std::is_same_v<float, get_value_t<A>>)
             and have_same_value_type_v<A, TAU, W> and mem::have_compatible_addr_space<A, TAU, W>)
  int orgqr(A &&a, TAU &&tau, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::orgqr: C order is not supported");
    using value_type = get_value_t<A>;

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    auto [m, n] = a.shape();
    auto k      = tau.size();

    // first call to get the optimal buffersize
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, TAU, W>) {
#if defined(NDA_HAVE_DEVICE)
      int buffer_size = device::orgqr_bufferSize(m, std::min(m, n), k, a.data(), get_ld(a), tau.data());
      if (work.size() < buffer_size) work.resize(buffer_size);
      EXPECTS(work.indexmap().min_stride() == 1);
      device::orgqr(m, std::min(m, n), k, a.data(), get_ld(a), tau.data(), work.data(), buffer_size, info);
#else
      compile_error_no_gpu();
#endif
    } else {
      value_type bufferSize_T{};
      lapack::f77::orgqr(m, std::min(m, n), k, a.data(), get_ld(a), tau.data(), &bufferSize_T, -1, info);
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      // resize work buffer and perform actual library call
      if (work.size() < bufferSize) work.resize(bufferSize);
      EXPECTS(work.indexmap().min_stride() == 1);
      lapack::f77::orgqr(m, std::min(m, n), k, a.data(), get_ld(a), tau.data(), work.data(), bufferSize, info);
    }

    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::orgqr: info = " << info;
    return info;
  }

  template <MemoryMatrix A, MemoryVector TAU>
    requires((std::is_same_v<double, get_value_t<A>> or std::is_same_v<float, get_value_t<A>>)
             and have_same_value_type_v<A, TAU> and mem::have_compatible_addr_space<A, TAU>)
  int orgqr(A &&a, TAU &&tau) { // NOLINT (temporary views are allowed here)
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return orgqr(std::forward<A>(a), std::forward<TAU>(tau), work);
  }

  template <MemoryArrayOfRank<3> A, MemoryMatrix TAU, MemoryVector W>
    requires((std::is_same_v<double, get_value_t<A>> or std::is_same_v<float, get_value_t<A>>)
             and have_same_value_type_v<A, TAU, W> and mem::have_compatible_addr_space<A, TAU, W>)
  int orgqr(A &&a, TAU &&tau, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::orgqr: C order is not supported");
    using value_type = get_value_t<A>;

    if constexpr (has_C_layout<TAU>) return orgqr(std::forward<A>(a), transpose(tau), work);

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    auto [m, n, batchSize] = a.shape();
    auto k                 = tau.extent(0);
    EXPECTS(tau.extent(1) >= batchSize);

    // first call to get the optimal buffersize
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, TAU, W>) {
#if defined(NDA_HAVE_DEVICE)
      int buffer_size = device::orgqr_bufferSize(m, std::min(m, n), k, a.data(), a.strides()[1], tau.data());
      if (work.size() < buffer_size) work.resize(buffer_size);
      EXPECTS(work.indexmap().min_stride() == 1);
      for (int b = 0; b < batchSize; ++b) {
        auto a_b = a(range::all, range::all, b);
        device::orgqr(m, std::min(m, n), k, a_b.data(), get_ld(a_b), tau.data() + b * get_ld(tau), work.data(), buffer_size, info);
      }
#else
      compile_error_no_gpu();
#endif
    } else {
      value_type bufferSize_T{};
      lapack::f77::orgqr(m, std::min(m, n), k, a.data(), a.strides()[1], tau.data(), &bufferSize_T, -1, info);
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      // resize work buffer and perform actual library call
      if (work.size() < bufferSize) work.resize(bufferSize);
      EXPECTS(work.indexmap().min_stride() == 1);
      for (int b = 0; b < batchSize; ++b) {
        auto a_b = a(range::all, range::all, b);
        lapack::f77::orgqr(m, std::min(m, n), k, a_b.data(), get_ld(a_b), tau.data() + b * get_ld(tau), work.data(), bufferSize, info);
      }
    }

    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::orgqr: info = " << info;
    return info;
  }

  template <MemoryArrayOfRank<3> A, MemoryMatrix TAU>
    requires((std::is_same_v<double, get_value_t<A>> or std::is_same_v<float, get_value_t<A>>)
             and have_same_value_type_v<A, TAU> and mem::have_compatible_addr_space<A, TAU>)
  int orgqr(A &&a, TAU &&tau) { // NOLINT (temporary views are allowed here)
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return orgqr(std::forward<A>(a), std::forward<TAU>(tau), work);
  }

} // namespace nda::lapack
