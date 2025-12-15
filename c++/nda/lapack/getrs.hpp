// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `getrs` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#if defined(NDA_HAVE_DEVICE)
#include "../blas/interface/cxx_interface.hpp" // batched getrs is in cublas
#endif

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif // NDA_HAVE_DEVICE

#include <algorithm>
#include <type_traits>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `getrs` routine.
   *
   * @details Solves a system of linear equations
   *
   * - \f$ \mathbf{A X} = \mathbf{B} \f$,
   * - \f$ \mathbf{A}^T \mathbf{X} = \mathbf{B} \f$ or
   * - \f$ \mathbf{A}^H \mathbf{X} = \mathbf{B} \f$
   *
   * with a general n-by-n matrix \f$ \mathbf{A} \f$ using the LU factorization computed by `getrs`.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam B nda::MemoryMatrix type.
   * @tparam IPIV nda::MemoryVector type.
   * @param a Input matrix. The factors \f$ \mathbf{L} \f$ and \f$ \mathbf{U} \f$ from the factorization \f$ \mathbf{A}
   * = \mathbf{P L U} \f$ as computed by `getrs`.
   * @param b Input/output matrix. On entry, the right hand side matrix \f$ \mathbf{B} \f$. On exit, the solution matrix
   * \f$ \mathbf{X} \f$.
   * @param ipiv Input vector. The pivot indices from `getrs`, i.e. for `1 <= i <= n`, row i of the matrix was
   * interchanged with row `ipiv(i)`.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryMatrix B, MemoryVector IPIV>
    requires(have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  int getrs(A const &a, B &&b, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<B>, "Error in nda::lapack::getrs: B must have Fortran layout.");
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrs: Pivoting array must have elements of type int");
    EXPECTS(ipiv.size() >= std::min(a.extent(0), a.extent(1)));

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(ipiv.indexmap().min_stride() == 1);

    // check for lazy expressions
    static constexpr bool conj_A = is_conj_array_expr<A>;
    char op_a                    = get_op<conj_A, /* transpose = */ has_C_layout<A>>;

    // perform actual library call
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, B, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
      device::getrs(op_a, get_ncols(a), get_ncols(b), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::getrs(op_a, get_ncols(a), get_ncols(b), a.data(), get_ld(a), ipiv.data(), b.data(), get_ld(b), info);
    }
    return info;
  }

  namespace detail {

    /// helper function to simplify implementation. Decision to transpose is made outside this routine.
    template <bool transpose, MemoryArrayOfRank<3> A, MemoryArrayOfRank<3> B, MemoryMatrix IPIV, MemoryVector W>
      requires(have_same_value_type_v<A, B, W> and mem::have_compatible_addr_space<A, B, IPIV, W> and is_blas_lapack_v<get_value_t<A>>)
    auto getrs_impl(A const &a, B &&b, IPIV const &ipiv, [[maybe_unused]] W &&work) { // NOLINT (temporary views are allowed here)
      static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrs: Pivoting array must have elements of type int");
      static_assert(has_F_layout<A> and has_F_layout<B>, "Error in nda::lapack::detail::getrs_impl: Only Fortran layout allowed.");

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
      EXPECTS(b.indexmap().min_stride() == 1);
      EXPECTS(a.extent(0) == a.extent(1));
      EXPECTS(a.extent(0) == b.extent(0));
      EXPECTS(ipiv.indexmap().min_stride() == 1);

      // check for lazy expressions
      static constexpr bool conj_A = is_conj_array_expr<A>;
      char op_a                    = get_op<conj_A, transpose>;

      // perform actual library call
      array<int, 1> info(batchSize, 0);
      if constexpr (mem::have_device_compatible_addr_space<A, B, IPIV>) {
#if defined(NDA_HAVE_DEVICE)
        using value_type = get_value_t<A>;
        // array of pointers
        array<const value_type *, 1> ptr1_h(batchSize);
        array<value_type *, 1> ptr2_h(batchSize);
        for (int i = 0; i < batchSize; ++i) {
          ptr1_h(i) = a(range::all, range::all, i).data();
          ptr2_h(i) = b(range::all, range::all, i).data();
        }
        // copy to device, using container policy from work array
        using handle_t = typename std::decay_t<W>::container_policy_t;
        array<const value_type *, 1, C_layout, handle_t> ptr1_d(ptr1_h);
        array<value_type *, 1, C_layout, handle_t> ptr2_d(ptr2_h);
        //        array<int,1,C_layout,handle_t> info_d(batchSize,0);  // somehow, trs has info on host
        blas::device::getrs_batched(op_a, a.extent(0), b.extent(1), ptr1_d.data(), a.strides()[1], ipiv.data(), ptr2_d.data(), b.strides()[1],
                                    info.data(), batchSize);
//        info() = info_d();
#else
        compile_error_no_gpu();
#endif
      } else {
        for (int i = 0; i < batchSize; ++i) {
          auto a_b = a(range::all, range::all, i);
          auto b_b = b(range::all, range::all, i);
          f77::getrs(op_a, a.extent(0), b.extent(1), a_b.data(), get_ld(a_b), ipiv.data() + b * get_ld(ipiv), b_b.data(), get_ld(b_b), info(i));
        }
      }
      return info;
    }

  } // namespace detail

  /// batched. The slowest stride is the batched one.
  template <MemoryArrayOfRank<3> A, MemoryArrayOfRank<3> B, MemoryMatrix IPIV, MemoryVector W>
    requires(have_same_value_type_v<A, B, W> and mem::have_compatible_addr_space<A, B, IPIV, W> and is_blas_lapack_v<get_value_t<A>>)
  auto getrs(A const &a, B &&b, IPIV const &ipiv, [[maybe_unused]] W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(std::is_same_v<get_value_t<IPIV>, int>, "Error in nda::lapack::getrs: Pivoting array must have elements of type int");
    static_assert(has_F_layout<B>, "Error in nda::lapack::getrs: Only Fortran layout for B matrix allowed.");
    static_assert(has_F_layout<A> or has_C_layout<A>, "Error in nda::lapack::getrs: Only C or Fortran layout allowed.");

    if constexpr (has_C_layout<A>)
      return detail::getrs_impl<true>(transpose(a), std::forward<B>(b), ipiv, std::forward<W>(work));
    else
      return detail::getrs_impl<false>(a, std::forward<B>(b), ipiv, std::forward<W>(work));
  }

  template <MemoryArrayOfRank<3> A, MemoryArrayOfRank<3> B, MemoryMatrix IPIV>
    requires(have_same_value_type_v<A, B> and mem::have_compatible_addr_space<A, B, IPIV> and is_blas_lapack_v<get_value_t<A>>)
  auto getrs(A const &a, B &&b, IPIV const &ipiv) {
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return getrs(a, std::forward<B>(b), ipiv, work);
  }

} // namespace nda::lapack
