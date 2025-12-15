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
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
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
#include <utility>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `gesvd` routine.
   *
   * @details Computes the singular value decomposition (SVD) of a complex m-by-n matrix \f$ \mathbf{A} \f$, optionally
   * computing the left and/or right singular vectors. The SVD is written
   * \f[
   *   \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^H
   * \f]
   * where \f$ \mathbf{S} \f$ is an m-by-n matrix which is zero except for its `min(m,n)` diagonal elements,
   * \f$ \mathbf{U} \f$ is an m-by-m unitary matrix, and \f$ \mathbf{V} \f$ is an n-by-n unitary matrix. The diagonal
   * elements of \f$ \mathbf{S} \f$ are the singular values of \f$ \mathbf{A} \f$; they are real and non-negative, and
   * are returned in descending order. The first `min(m,n)` columns of \f$ \mathbf{U} \f$ and \f$ \mathbf{V} \f$ are the
   * left and right singular vectors of \f$ \mathbf{A} \f$.
   *
   * Note that the routine returns \f$ \mathbf{V}^H \f$, not \f$ \mathbf{V} \f$.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam S nda::MemoryVector type.
   * @tparam U nda::MemoryMatrix type.
   * @tparam VT nda::MemoryMatrix type.
   * @param a Input/output matrix. On entry, the m-by-n matrix \f$ \mathbf{A} \f$. On exit, the contents of
   * \f$ \mathbf{A} \f$ are destroyed.
   * @param s Output vector. The singular values of \f$ \mathbf{A} \f$, sorted so that `s(i) >= s(i+1)`.
   * @param u Output matrix. It contains the m-by-m unitary matrix \f$ \mathbf{U} \f$.
   * @param vt Output matrix. It contains contains the n-by-n unitary matrix \f$ \mathbf{V}^H \f$.
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector S, MemoryMatrix U, MemoryMatrix VT, MemoryVector W>
    requires(have_same_value_type_v<A, U, VT, W> and mem::have_compatible_addr_space<A, S, U, VT, W> and is_blas_lapack_v<get_value_t<A>>)
  int gesvd(A &&a, S &&s, U &&u, VT &&vt, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A> and has_F_layout<U> and has_F_layout<VT>, "Error in nda::lapack::gesvd: C order not supported");

    auto dm = std::min(a.extent(0), a.extent(1));
    if (s.size() < dm) s.resize(dm);

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(s.indexmap().min_stride() == 1);
    EXPECTS(u.indexmap().min_stride() == 1);
    EXPECTS(vt.indexmap().min_stride() == 1);

    // call host/device implementation depending on input type
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

    // first call to get the optimal buffersize
    using value_type = get_value_t<A>;
    value_type bufferSize_T{};
    auto rwork = array<remove_complex_t<value_type>, 1, C_layout, heap<mem::get_addr_space<A>>>(5 * dm);
    int info   = 0;
    gesvd_call('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), s.data(), u.data(), get_ld(u), vt.data(), get_ld(vt), &bufferSize_T, -1,
               rwork.data(), info);
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // allocate work buffer and perform actual library call
    if (work.size() < bufferSize) work.resize(bufferSize);
    EXPECTS(work.indexmap().min_stride() == 1);
    gesvd_call('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), s.data(), u.data(), get_ld(u), vt.data(), get_ld(vt), work.data(), bufferSize,
               rwork.data(), info);

    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::gesvd: info = " << info;
    return info;
  }

  template <MemoryMatrix A, MemoryVector S, MemoryMatrix U, MemoryMatrix VT>
    requires(have_same_value_type_v<A, U, VT> and mem::have_compatible_addr_space<A, S, U, VT> and is_blas_lapack_v<get_value_t<A>>)
  int gesvd(A &&a, S &&s, U &&u, VT &&vt) { // NOLINT (temporary views are allowed here)
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return gesvd(std::forward<A>(a), std::forward<S>(s), std::forward<U>(u), std::forward<VT>(vt), work);
  }

} // namespace nda::lapack
