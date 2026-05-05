// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to batched versions of the LAPACK/cuSOLVER `getrf` routine.
 */

#pragma once

#include "./getrf.hpp"
#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
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
#include <type_traits>
#include <utility>

namespace nda::lapack {

  namespace detail {

    // Implementation of the batched getrf routine.
    template <bool run_on_device>
    auto getrf_batch_impl(auto &&a, auto &&ipiv, [[maybe_unused]] auto &&work) {
      // check the dimensions of the input/output arrays/views and resize if necessary
      auto const m   = a.extent(0);
      auto const n   = a.extent(1);
      auto const n_b = a.extent(2);
      resize_or_check_if_view(ipiv, {std::min(m, n), n_b});

      // arrays/views must be LAPACK compatible
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(ipiv.indexmap().min_stride() == 1);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      ipiv = 0;
#endif
#endif

      // loop over batches and call getrf for each matrix
      auto loop_getrf = [n_b, &a, &ipiv, &work](auto &info) {
        for (int i = 0; i < n_b; ++i) {
          auto a_i    = a(range::all, range::all, i);
          auto ipiv_i = ipiv(range::all, i);
          info(i)     = getrf(a_i, ipiv_i, work);
        }
      };

      // perform actual library call(s)
      auto info = array<int, 1>(n_b, 0);
      if constexpr (run_on_device) {
        if (m == n) {
          // for square matrices on the device use cuBLAS
          using arr_t = std::remove_cvref_t<decltype(a)>;
          auto ptr_d  = to_device(batch_ptrs(a));
          auto info_d = vector<int, heap<mem::get_addr_space<arr_t>>>(n_b, 0);
          blas::device::getrf_batch(n, ptr_d.data(), get_ld(a(range::all, range::all, 0)), ipiv.data(), info_d.data(), n_b);
          info = info_d;
        } else {
          // for rectangular matrices on the device, fall back to looping over batches
          loop_getrf(info);
        }
      } else {
        // for host, fall back to looping over batches
        loop_getrf(info);
      }
      return info;
    }

  } // namespace detail

  /**
   * @ingroup linalg_lapack
   * @brief Interface to batched versions of the LAPACK/cuSOLVER `%getrf` routine.
   *
   * @details This function computes LU factorizations
   * \f[
   *   \mathbf{A}_i = \mathbf{P}_i \mathbf{L}_i \mathbf{U}_i \; ,
   * \f]
   * for a batch of matrices indexed by \f$ i \in \{ 0, \ldots, N_b - 1 \} \f$. Here, \f$ N_b \f$ is the batch size. See
   * also nda::lapack::getrf.
   * 
   * A batch of matrices is just a 3-dimensional array in either nda::C_layout or nda::F_layout. For a Fortran/C layout 
   * array, the last/first dimension indexes the individual matrices such that `A(:,:,i)`/`A(i,:,:)` corresponds to the 
   * \f$ i \f$-th matrix \f$ \mathbf{A}_i \f$ in the batch.
   * 
   * Depending on the input array types, the function does the following:
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space and if the individual matrices are 
   * square, it calls cuBLAS's `cublasXgetrfBatched`.
   * - Otherwise, it simply loops over all matrices in the batch and calls nda::lapack::getrf.
   * 
   * @note If \f$ \mathbf{A} \f$ is stored in nda::C_layout, the factorizations are actually performed on \f$ 
   * \mathbf{A}_i^T \f$. When the result is further used in nda::lapack::getrs_batch or nda::lapack::getri_batch, this 
   * is automatically taken into account and works as expected.
   *
   * @tparam A nda::blas_lapack::BlasArray<3> type.
   * @tparam IPIV nda::blas_lapack::PivotArrayFor<A, 2> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output array. On entry, the 3-dimensional array \f$ \mathbf{A} \f$ containing the matrices \f$
   * \mathbf{A}_i \f$ to be factored. On exit, the corresponding \f$ \mathbf{L}_i \f$ and \f$ \mathbf{U}_i \f$ matrices 
   * from the factorization.
   * @param ipiv Output matrix. If the matrix is in Fortran (C) layout, the \f$ i \f$-th column (row) contains the pivot
   * indices from the factorization of \f$ \mathbf{A}_i \f$.
   * @param work Ouput vector. Workspace array only used by the cuSOLVER routine.
   * @return nda::array of integer return codes from the LAPACK/cuBLAS/cuSOLVER call(s).
   */
  template <BlasArray<3> A, PivotArrayFor<A, 2> IPIV, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A> or has_C_layout<A>)
  auto getrf_batch(A &&a, IPIV &&ipiv, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, IPIV, W>;

    // transpose ipiv array/view if necessary
    if constexpr (has_C_layout<IPIV>) return getrf_batch(a, transpose(ipiv), work);

    // transpose A array/view if necessary and call the implementation
    if constexpr (has_C_layout<A>) {
      return detail::getrf_batch_impl<run_on_device>(transpose(a), std::forward<IPIV>(ipiv), std::forward<W>(work));
    } else {
      return detail::getrf_batch_impl<run_on_device>(std::forward<A>(a), std::forward<IPIV>(ipiv), std::forward<W>(work));
    }
  }

  /**
   * @ingroup linalg_lapack
   * @brief Generic-friendly overload of nda::lapack::getrf for batches stored as 3-dimensional arrays.
   *
   * @details It simply calls nda::lapack::getrf_batch and lets generic code call `%getrf(...)` regardless of whether 
   * the input is a single matrix (rank 2) or a batch (rank 3).
   */
  template <BlasArray<3> A, PivotArrayFor<A, 2> IPIV, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A> or has_C_layout<A>)
  auto getrf(A &&a, IPIV &&ipiv, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    return getrf_batch(std::forward<A>(a), std::forward<IPIV>(ipiv), std::forward<W>(work));
  }

} // namespace nda::lapack
