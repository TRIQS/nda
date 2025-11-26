// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `geev` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <cmath>
#include <complex>
#include <concepts>

namespace nda::lapack {

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `geev` routine for real matrices.
   *
   * @details Computes for an \f$ n \times n \f$ real nonsymmetric matrix \f$ \mathbf{A} \f$, the eigenvalues and,
   * optionally, the left and/or right eigenvectors.
   *
   * The right eigenvector \f$ \mathbf{v}_j \f$ of \f$ \mathbf{A} \f$ satisfies
   * \f[
   *   \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{v}_j
   * \f]
   * where \f$ \lambda_j \f$ is its eigenvalue.
   *
   * The left eigenvector \f$ \mathbf{u}_j \f$ of \f$ \mathbf{A} \f$ satisfies
   * \f[
   *   \mathbf{u}_j^T \mathbf{A} = \lambda_j \mathbf{u}_j^T
   * \f]
   * where \f$ \mathbf{u}_j^T \f$ denotes the conjugate-transpose of \f$ \mathbf{u}_j \f$.
   *
   * The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real.
   *
   * @note For real matrices, complex eigenvalues always occur in complex conjugate pairs and the corresponding 
   * eigenvectors are stored in a special packed format (see nda::linalg::get_geev_eigenvectors).
   *
   * @tparam A nda::MemoryMatrix with double value type.
   * @tparam WR nda::MemoryVector with double value type.
   * @tparam WI nda::MemoryVector with double value type.
   * @tparam VL nda::MemoryMatrix with double value type.
   * @tparam VR nda::MemoryMatrix with double value type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, \f$ \mathbf{A} \f$ is overwritten.
   * @param wr Output vector. The real parts of the computed eigenvalues.
   * @param wi Output vector. The imaginary parts of the computed eigenvalues.
   * @param vl Output matrix. If `jobvl = V`, the left eigenvectors (in packed format for complex pairs). If 
   * `jobvl = N`, `vl` is not referenced.
   * @param vr Output matrix. If `jobvr = V`, the right eigenvectors (in packed format for complex pairs). If
   * `jobvr = N`, `vr` is not referenced.
   * @param jobvl Character indicating whether to compute left eigenvectors ('V') or not ('N').
   * @param jobvr Character indicating whether to compute right eigenvectors ('V') or not ('N').
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector WR, MemoryVector WI, MemoryMatrix VL, MemoryMatrix VR>
    requires(mem::have_host_compatible_addr_space<A, WR, WI, VL, VR> and std::same_as<double, get_value_t<A>>
             and have_same_value_type_v<A, WR, WI, VL, VR>)
  int geev(A &&a, WR &&wr, WI &&wi, VL &&vl, VR &&vr, char jobvl = 'N', char jobvr = 'V') { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::geev: A must have Fortran layout");
    static_assert(has_F_layout<VL>, "Error in nda::lapack::geev: VL must have Fortran layout");
    static_assert(has_F_layout<VR>, "Error in nda::lapack::geev: VR must have Fortran layout");

    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    resize_or_check_if_view(wr, {n});
    resize_or_check_if_view(wi, {n});

    // check parameters
    EXPECTS(jobvl == 'V' or jobvl == 'N');
    EXPECTS(jobvr == 'V' or jobvr == 'N');

    // resize eigenvector matrices if needed
    int ldvl = 1;
    int ldvr = 1;
    if (jobvl == 'V') {
      resize_or_check_if_view(vl, {n, n});
      ldvl = get_ld(vl);
    }
    if (jobvr == 'V') {
      resize_or_check_if_view(vr, {n, n});
      ldvr = get_ld(vr);
    }

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(wr.indexmap().min_stride() == 1);
    EXPECTS(wi.indexmap().min_stride() == 1);
    EXPECTS(jobvl == 'N' or vl.indexmap().min_stride() == 1);
    EXPECTS(jobvr == 'N' or vr.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    double tmp_lwork{};
    int info = 0;
    lapack::f77::geev(jobvl, jobvr, n, a.data(), get_ld(a), wr.data(), wi.data(), vl.data(), ldvl, vr.data(), ldvr, &tmp_lwork, -1, info);
    int lwork = static_cast<int>(std::ceil(tmp_lwork));

    // allocate work buffer and perform actual library call
    array<double, 1> work(lwork);
    lapack::f77::geev(jobvl, jobvr, n, a.data(), get_ld(a), wr.data(), wi.data(), vl.data(), ldvl, vr.data(), ldvr, work.data(), lwork, info);

    return info;
  }

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `geev` routine for complex matrices.
   *
   * @details Computes for an \f$ n \times n \f$ complex nonsymmetric matrix \f$ \mathbf{A} \f$, the eigenvalues and,
   * optionally, the left and/or right eigenvectors.
   *
   * The right eigenvector \f$ \mathbf{v}_j \f$ of \f$ \mathbf{A} \f$ satisfies
   * \f[
   *   \mathbf{A} \mathbf{v}_j = \lambda_j \mathbf{v}_j
   * \f]
   * where \f$ \lambda_j \f$ is its eigenvalue.
   *
   * The left eigenvector \f$ \mathbf{u}_j \f$ of \f$ \mathbf{A} \f$ satisfies
   * \f[
   *   \mathbf{u}_j^H \mathbf{A} = \lambda_j \mathbf{u}_j^H
   * \f]
   * where \f$ \mathbf{u}_j^H \f$ denotes the conjugate-transpose of \f$ \mathbf{u}_j \f$.
   *
   * The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real.
   *
   * @tparam A nda::MemoryMatrix with complex value type.
   * @tparam W nda::MemoryVector with complex value type.
   * @tparam VL nda::MemoryMatrix with complex value type.
   * @tparam VR nda::MemoryMatrix with complex value type.
   * @param a Input/output matrix. On entry, the matrix \f$ \mathbf{A} \f$. On exit, \f$ \mathbf{A} \f$ is overwritten.
   * @param w Output vector. The computed eigenvalues.
   * @param vl Output matrix. If `jobvl = V`, the left eigenvectors. If `jobvl = N`, `vl` is not referenced.
   * @param vr Output matrix. If `jobvr = V`, the right eigenvectors. If `jobvl = N`, `vl` is not referenced.
   * @param jobvl Character indicating whether to compute left eigenvectors ('V') or not ('N').
   * @param jobvr Character indicating whether to compute right eigenvectors ('V') or not ('N').
   * @return Integer return code from the LAPACK call.
   */
  template <MemoryMatrix A, MemoryVector W, MemoryMatrix VL, MemoryMatrix VR>
    requires(mem::have_host_compatible_addr_space<A, W, VL, VR> and std::same_as<std::complex<double>, get_value_t<A>>
             and have_same_value_type_v<A, W, VL, VR>)
  int geev(A &&a, W &&w, VL &&vl, VR &&vr, char jobvl = 'N', char jobvr = 'V') { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::geev: A must have Fortran layout");
    static_assert(has_F_layout<VL>, "Error in nda::lapack::geev: VL must have Fortran layout");
    static_assert(has_F_layout<VR>, "Error in nda::lapack::geev: VR must have Fortran layout");

    // check the dimensions of the input/output arrays/views
    auto const [m, n] = a.shape();
    EXPECTS(m == n);
    resize_or_check_if_view(w, {n});

    // check parameters
    EXPECTS(jobvl == 'V' or jobvl == 'N');
    EXPECTS(jobvr == 'V' or jobvr == 'N');

    // resize eigenvector matrices if needed
    int ldvl = 1;
    int ldvr = 1;
    if (jobvl == 'V') {
      resize_or_check_if_view(vl, {n, n});
      ldvl = get_ld(vl);
    }
    if (jobvr == 'V') {
      resize_or_check_if_view(vr, {n, n});
      ldvr = get_ld(vr);
    }

    // arrays/views must be LAPACK compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(w.indexmap().min_stride() == 1);
    EXPECTS(jobvl == 'N' or vl.indexmap().min_stride() == 1);
    EXPECTS(jobvr == 'N' or vr.indexmap().min_stride() == 1);

    // first call to get the optimal buffer size
    array<double, 1> rwork(2 * n);
    std::complex<double> tmp_lwork{};
    int info = 0;
    lapack::f77::geev(jobvl, jobvr, n, a.data(), get_ld(a), w.data(), vl.data(), ldvl, vr.data(), ldvr, &tmp_lwork, -1, rwork.data(), info);
    int lwork = static_cast<int>(std::ceil(std::real(tmp_lwork)));

    // allocate work buffer and perform actual library call
    array<std::complex<double>, 1> work(lwork);
    lapack::f77::geev(jobvl, jobvr, n, a.data(), get_ld(a), w.data(), vl.data(), ldvl, vr.data(), ldvr, work.data(), lwork, rwork.data(), info);

    return info;
  }

} // namespace nda::lapack
