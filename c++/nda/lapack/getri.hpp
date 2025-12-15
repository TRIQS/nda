// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

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

#if defined(NDA_HAVE_DEVICE)
#include "../blas/interface/cxx_interface.hpp" // batched getri is in cublas
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getri` routine.
   *
   * @details Computes the inverse of a matrix using the LU factorization computed by `getri`.
   *
   * This method inverts \f$ \mathbf{U} \f$ and then computes \f$ \mathrm{inv}(\mathbf{A}) \f$ by solving the system
   * \f$ \mathrm{inv}(\mathbf{A}) L = \mathrm{inv}(\mathbf{U}) \f$ for \f$ \mathrm{inv}(\mathbf{A}) \f$.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the
   * factorization \f$ \mathbf{A} = \mathbf{P L U} \f$ as computed by `getri`. On exit, if `INFO == 0`, the inverse of
   * the original matrix \f$ \mathbf{A} \f$.
   * @param ipiv Input vector. The pivot indices from `getri`, i.e. for `1 <= i <= N`, row i of the matrix was
   * interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector IPIV, MemoryVector W>
    requires(mem::have_compatible_addr_space<A, IPIV, W> and is_blas_lapack_v<get_value_t<A>> and std::is_same_v<get_value_t<A>, get_value_t<W>>)
  int getri(A &&a, IPIV const &ipiv, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getri: Pivoting array must have elements of type int");
    auto dm = std::min(a.extent(0), a.extent(1));

    if (ipiv.size() < dm)
      NDA_RUNTIME_ERROR << "Error in nda::lapack::getri: Pivot index array size " << ipiv.size() << " smaller than required size " << dm;

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(a.extent(0) == a.extent(1));
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, IPIV, W>) {
#if defined(NDA_HAVE_DEVICE)
      int bufferSize = device::getri_bufferSize(a.extent(0), a.data(), get_ld(a));
      if (work.size() < bufferSize) work.resize(bufferSize);
      device::getri(a.extent(0), a.data(), get_ld(a), ipiv.data(), work.data(), bufferSize, info);
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
      if (work.size() < bufferSize) work.resize(bufferSize);
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      work = 0;
#endif
#endif
      f77::getri(a.extent(0), a.data(), get_ld(a), ipiv.data(), work.data(), bufferSize, info);
    }
    return info;
  }

  template <MemoryMatrix A, MemoryVector IPIV>
    requires(mem::have_compatible_addr_space<A, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getri(A &&a, IPIV const &ipiv) {
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return getri(std::forward<A>(a), ipiv, work);
  }

  // batched, the slowest stride is always the batched one
  template <MemoryArrayOfRank<3> A, MemoryMatrix IPIV, MemoryVector W>
    requires(mem::have_compatible_addr_space<A, IPIV, W> and is_blas_lapack_v<get_value_t<A>> and std::is_same_v<get_value_t<A>, get_value_t<W>>)
  auto getri(A &&a, IPIV const &ipiv, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getri: Pivoting array must have elements of type int");
    static_assert(has_F_layout<A> or has_C_layout<A>, "Error in nda::lapack::getri: Only C or Fortran layout allowed.");

    if constexpr (has_C_layout<A>) return getri(transpose(a), ipiv, work);

    auto batchSize = a.extent(2);
    auto dm        = std::min(a.extent(0), a.extent(1));
    if constexpr (has_C_layout<IPIV>) {
      if (ipiv.extent(1) < dm or ipiv.extent(0) < batchSize)
        NDA_RUNTIME_ERROR << "Error in nda::lapack::getri: Pivot index array dimensions (" << ipiv.extent(0) << "," << ipiv.extent(1)
                          << ") smaller than required size (" << batchSize << "," << dm << ")";
    } else {
      if (ipiv.extent(0) < dm or ipiv.extent(1) < batchSize)
        NDA_RUNTIME_ERROR << "Error in nda::lapack::getri: Pivot index array dimensions (" << ipiv.extent(0) << "," << ipiv.extent(1)
                          << ") smaller than required size (" << dm << "," << batchSize << ")";
    }

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(a.extent(0) == a.extent(1));
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    array<int, 1> info(batchSize, 0);
    if constexpr (mem::have_device_compatible_addr_space<A, IPIV, W>) {
#if defined(NDA_HAVE_DEVICE)
      using value_type = get_value_t<A>;
      // array of pointers
      array<value_type *, 1> ptr_h(2 * batchSize);
      if (work.size() < a.size()) work.resize(a.size());
      for (int b = 0; b < batchSize; ++b) {
        ptr_h(b)             = a(range::all, range::all, b).data();
        ptr_h(b + batchSize) = work.data() + b * a.extent(0) * a.extent(0);
      }
      // copy to device, using container policy from work array
      using handle_t = typename std::decay_t<W>::container_policy_t;
      array<value_type *, 1, C_layout, handle_t> ptr_d(ptr_h);
      array<int, 1, C_layout, handle_t> info_d(batchSize, 0);
      // dispatch batched call
      blas::device::getri_batched(a.extent(0), ptr_d.data(), a.strides()[1], ipiv.data(), ptr_d.data() + batchSize, a.extent(0), info_d.data(),
                                  batchSize);
      auto c = nda::cuarray_view<value_type, 3, F_layout>(a.shape(), work.data());
      a()    = c();
      info() = info_d();
#else
      compile_error_no_gpu();
#endif
    } else {
      // first call to get the optimal buffersize
      using value_type = get_value_t<A>;
      value_type bufferSize_T{};
      f77::getri(a.extent(0), a.data(), a.strides()[1], ipiv.data(), &bufferSize_T, -1, info(0));
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      // allocate work buffer and perform actual library call
      if (work.size() < bufferSize) work.resize(bufferSize);
      for (int b = 0; b < batchSize; ++b) {
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
        work = 0;
#endif
#endif
        auto a_b = a(range::all, range::all, b);
        f77::getri(a_b.extent(0), a_b.data(), get_ld(a_b), ipiv.data() + b * get_ld(ipiv), work.data(), bufferSize, info(b));
      }
    }
    return info;
  }

  template <MemoryArrayOfRank<3> A, MemoryMatrix IPIV>
    requires(mem::have_compatible_addr_space<A, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  auto getri(A &&a, IPIV const &ipiv) {
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return getri(std::forward<A>(a), ipiv, work);
  }

} // namespace nda::lapack
