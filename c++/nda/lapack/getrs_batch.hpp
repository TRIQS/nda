// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to batched versions of the LAPACK/cuSOLVER `getrs` routine.
 */

#pragma once

#include "./getrs.hpp"
#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../blas/interface/cxx_interface.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../device.hpp"
#include "../layout_transforms.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>

namespace nda::lapack {

  namespace detail {

    // Implementation of the batched getrs routine.
    template <bool run_on_device>
    int getrs_batch_impl(auto const &a, auto &b, auto const &ipiv, char op) {
      // get underlying array in case it is given as a lazy conjugate expression
      auto &a_arr = get_array(a);

      // check the dimensions of the input/output arrays/views
      auto const [m, n, n_b]      = a_arr.shape();
      auto const [k, nrhs, n_b_2] = b.shape();
      EXPECTS(m == n);
      EXPECTS(n == k);
      EXPECTS(n_b == n_b_2);
      EXPECTS(ipiv.extent(0) == n);
      EXPECTS(ipiv.extent(1) == n_b);

      // arrays/views must be LAPACK compatible
      EXPECTS(a_arr.indexmap().min_stride() == 1);
      EXPECTS(b.indexmap().min_stride() == 1);
      EXPECTS(ipiv.indexmap().min_stride() == 1);

      // perform actual library call(s)
      int info = 0;
      if constexpr (run_on_device) {
        auto a_ptrs = to_device(batch_ptrs(a_arr));
        auto b_ptrs = to_device(batch_ptrs(b));
        blas::device::getrs_batch(op, n, nrhs, a_ptrs.data(), get_ld(a_arr(range::all, range::all, 0)), ipiv.data(), b_ptrs.data(),
                                  get_ld(b(range::all, range::all, 0)), info, n_b);
      } else {
        // for host, fall back to looping over batches
        for (int i = 0; i < n_b; ++i) {
          auto a_i       = a_arr(range::all, range::all, i);
          auto b_i       = b(range::all, range::all, i);
          auto ipiv_i    = ipiv(range::all, i);
          int local_info = 0;
          f77::getrs(op, get_ncols(a_i), get_ncols(b_i), a_i.data(), get_ld(a_i), ipiv_i.data(), b_i.data(), get_ld(b_i), local_info);
          if (local_info != 0 && info == 0) info = local_info;
        }
      }
      return info;
    }

  } // namespace detail

  /**
   * @ingroup linalg_lapack
   * @brief Interface to batched versions of the LAPACK/cuSOLVER `getrs` routine.
   *
   * @details This function solves systems of linear equations
   * \f[
   *   \mathbf{A}_i \mathbf{X}_i = \mathbf{B}_i \; , 
   * \f]
   * for batches of matrices indexed by \f$ i \in \{ 0, \ldots, N_b - 1 \} \f$. Here, \f$ N_b \f$ is the batch size. See
   * also nda::lapack::getrs.
   * 
   * A batch of matrices is just a 3-dimensional array in either nda::C_layout or nda::F_layout. For a Fortran (C) 
   * layout array, the last (first) dimension indexes the individual matrices such that `M(:,:,i)` (`M(i,:,:)`) 
   * corresponds to the \f$ i \f$-th matrix \f$ \mathbf{M}_i \f$ in the batch.
   * 
   * Depending on the input array types, the function does the following:
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, it calls cuBLAS's `cublasXgetrsBatched`.
   * - Otherwise, it simply loops over all matrices in the batch and calls nda::lapack::getrs.
   * 
   * @note \f$ \mathbf{A} \f$ is allowed to be a lazy conjugate expression (see nda::blas_lapack::is_conj_array_expr), 
   * in which case it is required to be in nda::C_layout. Otherwise, it must have nda::F_layout or nda::C_layout. \f$ 
   * \mathbf{B} \f$ is required to be in nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj<3> type.
   * @tparam B nda::blas_lapack::BlasArrayFor<A, 3> type.
   * @tparam IPIV nda::blas_lapack::PivotArrayFor<A, 2> type.
   * @param a Input array. The 3-dimensional array containing \f$ N_b \f$ LU factorized matrices \f$ \mathbf{A}_i \f$ of 
   * size \f$ n \times n \f$ as computed by nda::lapack::getrf_batch.
   * @param b Input/output array. On entry, the 3-dimensional array containing \f$ N_b \f$ right hand side matrices \f$ 
   * \mathbf{B}_i \f$. On exit, the corresponding solution matrices \f$ \mathbf{X}_i \f$.
   * @param ipiv Input matrix. The pivot indices from nda::lapack::getrf_batch. If the matrix is in Fortran (C) layout,
   * the \f$ i \f$-th column (row) contains the pivot indices from the factorization of the \f$ i \f$-th matrix.
   * @return Integer return code from the batched LAPACK/cuBLAS call(s). If zero, all calls were successful.
   */
  template <BlasArrayOrConj<3> A, BlasArrayFor<A, 3> B, PivotArrayFor<A, 2> IPIV>
    requires((has_F_layout<A> or has_C_layout<A>) and has_F_layout<B> and (not is_conj_array_expr<A> or has_C_layout<A>))
  int getrs_batch(A const &a, B &&b, IPIV const &ipiv) { // NOLINT (temporary views are allowed here)
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, B, IPIV>;

    // transpose ipiv array/view if necessary
    if constexpr (has_C_layout<IPIV>) return getrs_batch(a, b, transpose(ipiv));

    // transpose A array/view if necessary and call the implementation with the correct cuBLAS op flag
    constexpr char op = (has_C_layout<A> ? (is_conj_array_expr<A> ? 'C' : 'T') : 'N');
    if constexpr (has_C_layout<A>) {
      return detail::getrs_batch_impl<run_on_device>(transpose(a), std::forward<B>(b), ipiv, op);
    } else {
      return detail::getrs_batch_impl<run_on_device>(a, std::forward<B>(b), ipiv, op);
    }
  }

} // namespace nda::lapack
