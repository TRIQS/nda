// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to batched versions of the LAPACK/cuSOLVER `geqrf` routine.
 */

#pragma once

#include "./geqrf.hpp"
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

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to batched versions of the LAPACK/cuSOLVER `%geqrf` routine.
   *
   * @details This function computes a QR factorization
   * \f[
   *   \mathbf{A}_i = \mathbf{Q}_i \mathbf{R}_i \; ,
   * \f]
   * for a batch of matrices indexed by \f$ i \in \{ 0, \ldots, N_b - 1 \} \f$. Here, \f$ N_b \f$ is the batch size. See
   * also nda::lapack::geqrf.
   *
   * A batch of matrices is just a 3-dimensional array in nda::F_layout where the last dimension indexes the individual
   * matrices such that `A(:,:,i)` corresponds to the \f$ i \f$-th matrix \f$ \mathbf{A}_i \f$ in the batch.
   *
   * Depending on the input array types, the function does the following:
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, it calls cuBLAS's `cublasXgeqrfBatched`.
   * - If the input arrays do not satisfy nda::mem::have_device_compatible_addr_space, it simply loops over all matrices
   * in the batch and calls nda::lapack::geqrf.
   *
   * @note \f$ \mathbf{A} \f$ and \f$ \mathbf{T} \f$ are required to have nda::F_layout.
   *
   * @tparam A nda::blas_lapack::BlasArray<3> type.
   * @tparam TAU nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @tparam W nda::blas_lapack::BlasArrayFor<A, 1> type.
   * @param a Input/output array. On entry, the 3-dimensional array \f$ \mathbf{A} \f$ containing the matrices \f$
   * \mathbf{A}_i \f$ to be factored. On exit, the corresponding upper trapezoidal matrices \f$ \mathbf{R}_i \f$ and the
   * elementary reflectors representing \f$ \mathbf{Q}_i \f$.
   * @param tau Output matrix \f$ \mathbf{T} \f$. The \f$ i \f$-th column contains the scalar factors of the elementary
   * reflectors representing \f$ \mathbf{Q}_i \f$.
   * @param work Ouput vector. Workspace array only used by the LAPACK routine.
   * @return Integer return code from the batched LAPACK/cuBLAS call(s). If zero, all calls were successful.
   */
  template <BlasArray<3> A, BlasArrayFor<A, 2> TAU, BlasArrayFor<A, 1> W = vector_value_t<A>>
    requires(has_F_layout<A, TAU>)
  int geqrf_batch(A &&a, TAU &&tau, [[maybe_unused]] W &&work = vector_value_t<A>{}) { // NOLINT (temporary views are allowed here)
    // check the dimensions of the input/output arrays/views and resize if necessary
    auto const [m, n, n_b] = a.shape();
    resize_or_check_if_view(tau, {std::min(m, n), n_b});

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    tau = get_value_t<A>{0};
#endif
#endif

    // perform actual library call(s)
    int info = 0;
    if constexpr (mem::have_device_compatible_addr_space<A, TAU, W>) {
      // get pointers to each matrix/vector in the batch
      auto a_ptrs   = to_device(batch_ptrs(a));
      auto tau_ptrs = to_device(batch_ptrs(tau));

      blas::device::geqrf_batch(m, n, a_ptrs.data(), get_ld(a(range::all, range::all, 0)), tau_ptrs.data(), info, n_b);
    } else {
      // for host, fall back to looping over batches
      for (int i = 0; i < n_b; ++i) {
        auto a_i       = a(range::all, range::all, i);
        auto tau_i     = tau(range::all, i);
        int local_info = geqrf(a_i, tau_i, work);
        if (local_info != 0 && info == 0) info = local_info;
      }
    }
    return info;
  }

} // namespace nda::lapack
