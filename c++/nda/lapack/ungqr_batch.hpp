// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to batched versions of the LAPACK/cuSOLVER `%ungqr` routine.
 */

#pragma once

#include "./ungqr.hpp"
#include "../blas/tools.hpp"
#include "../macros.hpp"

#include <type_traits>
#include <utility>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to batched versions of the LAPACK/cuSOLVER `%ungqr` routine.
   *
   * @details This function generates the unitary matrix \f$ \mathbf{Q}_i \f$ for each matrix in a batch indexed by
   * \f$ i \in \{ 0, \ldots, N_b - 1 \} \f$ from the elementary reflectors and scalar factors produced by
   * nda::lapack::geqrf_batch. Here, \f$ N_b \f$ is the batch size. See also nda::lapack::ungqr.
   * 
   * A batch of matrices is just a 3-dimensional array in nda::F_layout where the last dimension indexes the individual
   * matrices such that `A(:,:,i)` corresponds to the \f$ i \f$-th matrix \f$ \mathbf{A}_i \f$ in the batch.
   *
   * No library-level batching is currently available for this function. Instead, the function simply loops over all
   * matrices in the batch and calls nda::lapack::ungqr on the first \f$ \min(m, n) \f$ columns of each slice. For wide
   * matrices (\f$ m < n \f$), this produces an \f$ m \times m \f$ unitary matrix in the first \f$ m \f$ columns of
   * each slice; the remaining columns are left untouched.
   *
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{T} \f$ are required to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArrayCplx<3> type.
   * @tparam TAU nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output array. On entry, the 3-dimensional array containing the elementary reflectors for each batch
   * (as returned by nda::lapack::geqrf_batch). On exit, the first \f$ \min(m, n) \f$ columns of each slice contain
   * \f$ \mathbf{Q}_i \f$.
   * @param tau Input matrix. The \f$ i \f$-th column contains the scalar factors of the elementary reflectors
   * representing \f$ \mathbf{Q}_i \f$.
   * @param work Workspace array used by the underlying single-matrix call (resized as needed).
   * @return First non-zero LAPACK/cuSOLVER info code observed across the batch (0 on success).
   */
  template <BlasArrayCplx<3> A, BlasArrayFor<A, 2> TAU, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A, TAU>)
  int ungqr_batch(A &&a, TAU &&tau, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    auto const n_b = a.extent(2);
    auto const k   = tau.extent(0);
    EXPECTS(tau.extent(1) == n_b);
    EXPECTS(k <= a.extent(0) && k <= a.extent(1));

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    int info = 0;
    for (int i = 0; i < n_b; ++i) {
      int local_info = ungqr(a(range::all, range(k), i), tau(range::all, i), work);
      if (local_info != 0 && info == 0) info = local_info;
    }
    return info;
  }

  /**
   * @ingroup linalg_lapack
   * @brief Generic-friendly overload of nda::lapack::ungqr for batches stored as 3-dimensional arrays.
   *
   * @details It simply calls nda::lapack::ungqr_batch and lets generic code call `%ungqr(...)` regardless of whether 
   * the input is a single matrix (rank 2) or a batch (rank 3).
   */
  template <BlasArrayCplx<3> A, BlasArrayFor<A, 2> TAU, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A, TAU>)
  int ungqr(A &&a, TAU &&tau, W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    return ungqr_batch(std::forward<A>(a), std::forward<TAU>(tau), std::forward<W>(work));
  }

} // namespace nda::lapack
