// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `gesvd` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../layout/policies.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif // NDA_HAVE_DEVICE

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <utility>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `gesvd` routine.
   *
   * @details Computes the singular value decomposition (SVD) of an \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. The
   * SVD is written as
   * \f[
   *   \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^H \; ,
   * \f]
   * where \f$ \mathbf{S} \f$ is an \f$ m \times n \f$ matrix which is zero except for its \f$ \min(m,n) \f$ diagonal
   * elements, \f$ \mathbf{U} \f$ is an \f$ m \times m \f$ unitary matrix, and \f$ \mathbf{V} \f$ is an \f$ n \times n
   * \f$ unitary matrix. The diagonal elements of \f$ \mathbf{S} \f$ are the singular values of \f$ \mathbf{A} \f$; they
   * are real and non-negative, and are returned in descending order. The first \f$ min(m,n) \f$ columns of \f$
   * \mathbf{U} \f$ and \f$ \mathbf{V} \f$ are the left and right singular vectors of \f$ \mathbf{A} \f$.
   *
   * Note that the routine returns \f$ \mathbf{V}^H \f$, not \f$ \mathbf{V} \f$.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam S nda::MemoryVector type.
   * @tparam U nda::MemoryMatrix type.
   * @tparam VT nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the \f$ m \times n \f$ matrix \f$ \mathbf{A} \f$. On exit, the contents of
   * \f$ \mathbf{A} \f$ are destroyed.
   * @param s Output vector. The singular values of \f$ \mathbf{A} \f$, sorted so that \f$ s_i \geq s_{i+1} \f$.
   * @param u Output matrix. It contains the \f$ m \times m \f$ unitary matrix \f$ \mathbf{U} \f$.
   * @param vt Output matrix. It contains the \f$ n \times n \f$ unitary matrix \f$ \mathbf{V}^H \f$.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector S, MemoryMatrix U, MemoryMatrix VT>
    requires(have_same_value_type_v<A, U, VT> and mem::have_compatible_addr_space<A, S, U, VT> and is_blas_lapack_v<get_value_t<A>>
             and have_same_value_type_v<get_fp_t<A>, S>)
  int gesvd(A &&a, S &&s, U &&u, VT &&vt) { // NOLINT (temporary views are allowed here)
    static_assert(has_C_layout<A> == has_C_layout<U> and has_C_layout<A> == has_C_layout<VT>,
                  "Error in nda::lapack::gesvd: Matrix layouts have to be the same");

    // check the dimensions of the output arrays/views and resize if necessary
    auto const [m, n] = a.shape();
    auto const k      = std::min(m, n);
    resize_or_check_if_view(s, {k});
    resize_or_check_if_view(u, {m, m});
    resize_or_check_if_view(vt, {n, n});

    // cusolverDn?gesvd only supports matrices with m >= n
    if constexpr (mem::have_device_compatible_addr_space<A, S, U, VT>) {
      if constexpr (has_C_layout<A>) {
        EXPECTS(n >= m);
      } else {
        EXPECTS(m >= n);
      }
    }

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(s.indexmap().min_stride() == 1);
    EXPECTS(u.indexmap().min_stride() == 1);
    EXPECTS(vt.indexmap().min_stride() == 1);

    // call host/device implementation depending on address space of input arrays/views
    auto gesvd_call = []<typename... Ts>(Ts &&...args) {
      if constexpr (mem::have_device_compatible_addr_space<A, S, U, VT>) {
#if defined(NDA_HAVE_DEVICE)
        lapack::device::gesvd(std::forward<Ts>(args)...);
#else
        compile_error_no_gpu();
#endif
      } else {
        lapack::f77::gesvd(std::forward<Ts>(args)...);
      }
    };

    // first call to get the optimal buffer size
    using value_type = get_value_t<A>;
    value_type tmp_lwork{};
    auto rwork = array<get_fp_t<A>, 1, C_layout, heap<mem::get_addr_space<A>>>(5 * k);
    int info   = 0;
    if constexpr (has_C_layout<A>) {
      gesvd_call('A', 'A', n, m, a.data(), get_ld(a), s.data(), vt.data(), get_ld(vt), u.data(), get_ld(u), &tmp_lwork, -1, rwork.data(), info);
    } else {
      gesvd_call('A', 'A', m, n, a.data(), get_ld(a), s.data(), u.data(), get_ld(u), vt.data(), get_ld(vt), &tmp_lwork, -1, rwork.data(), info);
    }
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // allocate work buffer and perform actual library call
    array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work(lwork);
    if constexpr (has_C_layout<A>) {
      gesvd_call('A', 'A', n, m, a.data(), get_ld(a), s.data(), vt.data(), get_ld(vt), u.data(), get_ld(u), work.data(), lwork, rwork.data(), info);
    } else {
      gesvd_call('A', 'A', m, n, a.data(), get_ld(a), s.data(), u.data(), get_ld(u), vt.data(), get_ld(vt), work.data(), lwork, rwork.data(), info);
    }

    return info;
  }

} // namespace nda::lapack
