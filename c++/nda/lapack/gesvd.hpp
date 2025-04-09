// Copyright (c) 2020-2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Miguel Morales, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `gesvd` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../basic_functions.hpp"
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
#include <concepts>
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
  template <MemoryMatrix A, MemoryVector S, MemoryMatrix U, MemoryMatrix VT>
    requires(have_same_value_type_v<A, U, VT> and mem::have_compatible_addr_space<A, S, U, VT> and is_blas_lapack_v<get_value_t<A>>
             and std::same_as<double, get_value_t<S>>)
  int gesvd(A &&a, S &&s, U &&u, VT &&vt) { // NOLINT (temporary views are allowed here)
    static_assert(has_C_layout<A> == has_C_layout<U> and has_C_layout<A> == has_C_layout<VT>,
                  "Error in nda::lapack::gesvd: Matrix layouts have to be the same");

    // check the dimensions of the output arrays/views and resize if necessary
    auto dm = std::min(a.extent(0), a.extent(1));
    resize_or_check_if_view(s, {dm});
    resize_or_check_if_view(u, {a.extent(0), a.extent(0)});
    resize_or_check_if_view(vt, {a.extent(1), a.extent(1)});

    // arrays/views must be LAPACK compatible
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
    auto rwork = array<double, 1, C_layout, heap<mem::get_addr_space<A>>>(5 * dm);
    int info   = 0;
    if constexpr (has_C_layout<A>) {
      gesvd_call('A', 'A', a.extent(1), a.extent(0), a.data(), get_ld(a), s.data(), vt.data(), get_ld(vt), u.data(), get_ld(u), &bufferSize_T, -1,
                 rwork.data(), info);
    } else {
      gesvd_call('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), s.data(), u.data(), get_ld(u), vt.data(), get_ld(vt), &bufferSize_T, -1,
                 rwork.data(), info);
    }
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // allocate work buffer and perform actual library call
    array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work(bufferSize);
    if constexpr (has_C_layout<A>) {
      gesvd_call('A', 'A', a.extent(1), a.extent(0), a.data(), get_ld(a), s.data(), vt.data(), get_ld(vt), u.data(), get_ld(u), work.data(),
                 bufferSize, rwork.data(), info);
    } else {
      gesvd_call('A', 'A', a.extent(0), a.extent(1), a.data(), get_ld(a), s.data(), u.data(), get_ld(u), vt.data(), get_ld(vt), work.data(),
                 bufferSize, rwork.data(), info);
    }

    return info;
  }

} // namespace nda::lapack
