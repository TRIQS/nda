// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic dot product.
 */

#pragma once

#include "../blas/dot.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <complex>
#include <cstddef>

namespace nda::linalg {

  /**
   * @addtogroup linalg_tools
   * @{
   */

  namespace detail {

    // Implementation of a generic dot/dotc product.
    template <bool star, typename X, typename Y>
      requires(Scalar<get_value_t<X>> and Scalar<get_value_t<Y>> and mem::have_host_compatible_addr_space<X, Y>)
    auto dot_generic(X const &x, Y const &y) {
      // check the dimensions of the input arrays/views
      EXPECTS(x.size() == y.size());

      // conditional conjugation
      auto cond_conj = [](auto z) __attribute__((always_inline)) {
        if constexpr (star and is_complex_v<decltype(z)>) {
          return std::conj(z);
        } else {
          return z;
        }
      };

      // early return for zero-sized vectors
      long const N = x.size();
      if (N == 0) return decltype(cond_conj(x(0)) * y(0)){0};

      // loop over vectors and sum up element-wise products
      if constexpr (has_layout_smallest_stride_is_one<X> and has_layout_smallest_stride_is_one<Y>) {
        if constexpr (is_regular_or_view_v<X> and is_regular_or_view_v<Y>) {
          auto *__restrict px = x.data();
          auto *__restrict py = y.data();
          auto res            = cond_conj(px[0]) * py[0];
          for (size_t i = 1; i < N; ++i) res += cond_conj(px[i]) * py[i];
          return res;
        } else {
          auto res = cond_conj(x(_linear_index_t{0})) * y(_linear_index_t{0});
          for (long i = 1; i < N; ++i) res += cond_conj(x(_linear_index_t{i})) * y(_linear_index_t{i});
          return res;
        }
      } else {
        auto res = cond_conj(x(0)) * y(0);
        for (long i = 1; i < N; ++i) res += cond_conj(x(i)) * y(i);
        return res;
      }
    }

  } // namespace detail

  /**
   * @brief Compute the dot product of two nda::vector objects or the product of two scalars.
   *
   * @details The behaviour of this function is identical to nda::blas::dot, except that it allows
   * - the two input objects to be scalars,
   * - lazy expressions as input vectors,
   * - the value types of the input vectors to be different from each other and
   * - the value types of the input vectors to be different from nda::is_double_or_complex_v.
   *
   * For vectors, it calls nda::blas::dot if possible, otherwise it simply loops over the input arrays/views and sums up
   * the element-wise products.
   *
   * @note The first argument is never conjugated. Even for complex types. Use nda::linalg::dotc for that.
   *
   * @tparam X nda::Vector or nda::Scalar type.
   * @tparam Y nda::Vector or nda::Scalar type.
   * @param x Input vector/scalar.
   * @param y Input vector/scalar.
   * @return Result of the dot product.
   */
  template <typename X, typename Y>
    requires((Scalar<X> and Scalar<Y>) or (Vector<X> and Vector<Y>))
  auto dot(X const &x, Y const &y) {
    if constexpr (Scalar<X>) {
      return x * y;
    } else if constexpr (requires { nda::blas::dot(x, y); }) {
      return nda::blas::dot(x, y);
    } else {
      return detail::dot_generic<false>(x, y);
    }
  }

  /**
   * @brief Compute the dotc (LHS operand is conjugated) product of two nda::vector objects or the product of two
   * scalars.
   *
   * @details The behaviour of this function is identical to nda::blas::dotc, except that it allows
   * - the two input objects to be scalars,
   * - lazy expressions as input vectors,
   * - the value types of the input vectors to be different from each other and
   * - the value types of the input vectors to be different from nda::is_double_or_complex_v.
   *
   * For vectors, it calls nda::blas::dotc if possible, otherwise it simply loops over the input arrays/views and sums
   * up the element-wise products with the LHS operand conjugated.
   *
   * @tparam X nda::Vector or nda::Scalar type.
   * @tparam Y nda::Vector or nda::Scalar type.
   * @param x Input vector/scalar.
   * @param y Input vector/scalar.
   * @return Result of the dotc product.
   */
  template <typename X, typename Y>
    requires((Scalar<X> and Scalar<Y>) or (Vector<X> and Vector<Y>))
  auto dotc(X const &x, Y const &y) {
    if constexpr (Scalar<X>) {
      if constexpr (is_complex_v<X>) return std::conj(x) * y;
      return x * y;
    } else if constexpr (requires { nda::blas::dotc(x, y); }) {
      return nda::blas::dotc(x, y);
    } else {
      return detail::dot_generic<true>(x, y);
    }
  }

  /** @} */

} // namespace nda::linalg
