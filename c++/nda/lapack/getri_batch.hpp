// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to batched versions of the LAPACK/cuSOLVER `getri` routine.
 */

#pragma once

#include "./getri.hpp"
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
#include <type_traits>
#include <utility>

namespace nda::lapack {

  namespace detail {

    // Implementation of the batched getri routine.
    template <bool run_on_device>
    auto getri_batch_impl(auto &&a, auto const &ipiv, [[maybe_unused]] auto &&work) {
      // check the dimensions of the input/output arrays/views
      auto const [m, n, n_b] = a.shape();
      EXPECTS(m == n);
      EXPECTS(ipiv.extent(0) == n);
      EXPECTS(ipiv.extent(1) == n_b);

      // arrays/views must be LAPACK compatible
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(ipiv.indexmap().min_stride() == 1);

      // perform actual library call(s)
      auto info = array<int, 1>(n_b, 0);
      if constexpr (run_on_device) {
        using arr_t = std::remove_cvref_t<decltype(a)>;

        // resize/check work buffer
        resize_or_check_if_view(work, {a.size()});
        EXPECTS(work.indexmap().min_stride() == 1);

        // output buffer for inverted matrices
        auto c = cuarray_view<get_value_t<arr_t>, 3, F_layout>(a.shape(), work.data());

        auto a_ptrs = to_device(batch_ptrs(a));
        auto c_ptrs = to_device(batch_ptrs(c));
        auto info_d = vector<int, heap<mem::get_addr_space<arr_t>>>(n_b, 0);
        blas::device::getri_batch(n, a_ptrs.data(), get_ld(a(range::all, range::all, 0)), ipiv.data(), c_ptrs.data(),
                                  get_ld(c(range::all, range::all, 0)), info_d.data(), n_b);
        info = info_d;

        // copy result back to a
        a = c;
      } else {
        // for host, fall back to looping over batches
        for (int i = 0; i < n_b; ++i) {
          auto a_i    = a(range::all, range::all, i);
          auto ipiv_i = ipiv(range::all, i);
          info(i)     = getri(a_i, ipiv_i, work);
        }
      }
      return info;
    }

  } // namespace detail

  /**
   * @ingroup linalg_lapack
   * @brief Interface to batched versions of the LAPACK/cuSOLVER `%getri` routine.
   *
   * @details Computes the inverse of a batch of matrices indexed by \f$ i \in \{ 0, \ldots, N_b - 1 \} \f$. Here, 
   * \f$ N_b \f$ is the batch size. See also nda::lapack::getri.
   * 
   * A batch of matrices is just a 3-dimensional array in either nda::C_layout or nda::F_layout. For a Fortran/C layout 
   * array, the last/first dimension indexes the individual matrices such that `A(:,:,i)`/`A(i,:,:)` corresponds to the 
   * \f$ i \f$-th matrix \f$ \mathbf{A}_i \f$ in the batch.
   * 
   * Depending on the input array types, the function does the following:
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, it calls cuBLAS's `cublasXgetriBatched`.
   * - Otherwise, it simply loops over all matrices in the batch and calls nda::lapack::getri.
   * 
   * @note \f$ \mathbf{A} \f$ is required to have nda::F_layout or nda::C_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<3> type.
   * @tparam IPIV nda::blas_lapack::PivotArrayFor<A, 2> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output array. On entry, the 3-dimensional array \f$ \mathbf{A} \f$ containing LU factorized matrices 
   * \f$ \mathbf{A}_i \f$ as computed by nda::lapack::getrf_batch. On exit, the corresponding inverse matrices \f$ 
   * \mathbf{A}_i^{-1} \f$.
   * @param ipiv Input matrix. The pivot indices from nda::lapack::getrf_batch. If the matrix is in Fortran (C) layout, 
   * the \f$ i \f$-th column (row) contains the pivot indices from the factorization of \f$ \mathbf{A}_i \f$.
   * @param work Ouput vector. Workspace array used by the LAPACK/cuBLAS routine.
   * @return nda::array of integer return codes from the LAPACK/cuBLAS call(s).
   */
  template <BlasArray<3> A, PivotArrayFor<A, 2> IPIV, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A> or has_C_layout<A>)
  auto getri_batch(A &&a, IPIV const &ipiv, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    constexpr bool run_on_device = mem::have_device_compatible_addr_space<A, IPIV, W>;

    // transpose ipiv array/view if necessary
    if constexpr (has_C_layout<IPIV>) return getri_batch(a, transpose(ipiv), work);

    // transpose A array/view if necessary and call the implementation
    if constexpr (has_C_layout<A>) {
      return detail::getri_batch_impl<run_on_device>(transpose(a), ipiv, std::forward<W>(work));
    } else {
      return detail::getri_batch_impl<run_on_device>(std::forward<A>(a), ipiv, std::forward<W>(work));
    }
  }

  /**
   * @ingroup linalg_lapack
   * @brief Generic-friendly overload of nda::lapack::getri for batches stored as 3-dimensional arrays.
   *
   * @details It simply calls nda::lapack::getri_batch and lets generic code call `%getri(...)` regardless of whether 
   * the input is a single matrix (rank 2) or a batch (rank 3).
   */
  template <BlasArray<3> A, PivotArrayFor<A, 2> IPIV, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A> or has_C_layout<A>)
  auto getri(A &&a, IPIV const &ipiv, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    return getri_batch(std::forward<A>(a), ipiv, std::forward<W>(work));
  }

} // namespace nda::lapack
