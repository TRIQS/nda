// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic outer product function.
 */

#pragma once

#include "../blas/ger.hpp"
#include "../blas/tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"
#include "../traits.hpp"

namespace nda::linalg {

  /**
   * @ingroup linalg_tools
   * @brief Outer product of two arrays/views.
   *
   * @details It calculates the outer product \f$ \mathbf{C} = \mathbf{A} \otimes \mathbf{B} \f$, such that
   * \f[
   *   \mathbf{C}_{i_1 \ldots i_k j_1 \ldots j_l} = \mathbf{A}_{i_1 \ldots i_k} \mathbf{B}_{j_1 \ldots j_l} \; .
   * \f]
   * Here, \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are the input arrays with shape \f$ (m_1, \ldots, m_k) \f$ and \f$
   * (n_1, \ldots, n_l) \f$, respectively. The resulting array \f$ \mathbf{C} \f$ has shape \f$ (m_1, \ldots, m_k, n_1,
   * \ldots, n_l) \f$.
   *
   * The outer product is performed by calling nda::blas::ger, which imposes various constraints on the supported input
   * arrays/views, e.g.
   * - their memory layouts have to be the same and either nda::C_layout or nda::F_layout,
   * - they have to be contiguous in memory,
   * - etc.
   *
   * See nda::blas::ger for more details.
   *
   * @tparam A nda::MemoryArray type.
   * @tparam B nda::MemoryArray type.
   * @param a Input array/view \f$ \mathbf{A} \f$.
   * @param b Input array/view \f$ \mathbf{B} \f$.
   * @return Outer product \f$ \mathbf{A} \otimes \mathbf{B} \f$.
   */
  template <MemoryArray A, MemoryArray B>
    requires((nda::blas::has_C_layout<A> or nda::blas::has_F_layout<A>) and nda::blas::has_C_layout<A> == nda::blas::has_C_layout<B>)
  auto outer_product(A const &a, B const &b) {
    // check the input arrays/views
    EXPECTS(a.is_contiguous());
    EXPECTS(b.is_contiguous());

    // get the return type
    auto constexpr rank    = get_rank<A> + get_rank<B>;
    auto constexpr algebra = []() {
      if constexpr (get_algebra<A> == 'V' and get_algebra<B> == 'V') {
        return 'M';
      } else {
        return 'A';
      }
    }();
    using layout_pol = typename A::layout_policy_t::contiguous_t;
    using cont_pol   = heap<nda::mem::common_addr_space<A, B>>;
    using return_t   = basic_array<get_value_t<A>, rank, layout_pol, algebra, cont_pol>;

    // use ger to calculate the outer product
    auto res   = return_t::zeros(stdutil::join(a.shape(), b.shape()));
    auto a_vec = reshape(a, std::array{a.size()});
    auto b_vec = reshape(b, std::array{b.size()});
    auto mat   = reshape(res, std::array{a.size(), b.size()});
    nda::blas::ger(1.0, a_vec, b_vec, mat);

    return res;
  }

} // namespace nda::linalg
