// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS `dot` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mapped_functions.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif

#include <complex>

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Interface to the BLAS `dot` routine.
   *
   * @details This function forms the dot product of two vectors. It calculates
   * - \f$ \mathbf{x}^T \mathbf{y} \f$ in case that both \f$ \mathbf{x} \f$ and \f$ \mathbf{y} \f$ are vectors,
   * - \f$ x \mathbf{y} \f$ in case that \f$ x \f$ is a scalar and \f$ \mathbf{y} \f$ is a vector,
   * - \f$ \mathbf{x} y \f$ in case that \f$ \mathbf{x} \f$ is a vector and \f$ y \f$ is a scalar or
   * - \f$ x y \f$ in case that both \f$ x \f$ and \f$ y \f$ are scalars.
   *
   * @tparam X nda::MemoryVector or nda::Scalar type.
   * @tparam Y nda::MemoryVector or nda::Scalar type.
   * @param x Input vector/scalar.
   * @param y Input vector/scalar.
   * @return Vector/scalar result of the dot product.
   */
  template <typename X, typename Y>
    requires((Scalar<X> or MemoryVector<X>) and (Scalar<Y> or MemoryVector<X>))
  auto dot(X const &x, Y const &y) {
    if constexpr (Scalar<X> or Scalar<Y>) {
      return x * y;
    } else {
      // compile-time checks
      static_assert(have_same_value_type_v<X, Y>, "Error in nda::blas::dot: Incompatible value types");
      static_assert(mem::have_compatible_addr_space<X, Y>, "Error in nda::blas::dot: Incompatible memory address spaces");
      static_assert(is_blas_lapack_v<get_value_t<X>>, "Error in nda::blas::dot: Value types incompatible with blas");

      // runtime check
      EXPECTS(x.shape() == y.shape());

      if constexpr (mem::have_device_compatible_addr_space<X, Y>) {
#if defined(NDA_HAVE_DEVICE)
        return device::dot(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
#else
        compile_error_no_gpu();
        return get_value_t<X>(0);
#endif
      } else {
        return f77::dot(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
      }
    }
  }

  /**
   * @brief Interface to the BLAS `dotc` routine.
   *
   * @details This function forms the dot product of two vectors. It calculates
   * - \f$ \mathbf{x}^H \mathbf{y} \f$ in case that both \f$ \mathbf{x} \f$ and \f$ \mathbf{y} \f$ are vectors,
   * - \f$ \bar{x} \mathbf{y} \f$ in case that \f$ x \f$ is a scalar and \f$ \mathbf{y} \f$ is a vector,
   * - \f$ \mathbf{x}^H y \f$ in case that \f$ \mathbf{x} \f$ is a vector and \f$ y \f$ is a scalar or
   * - \f$ \bar{x} y \f$ in case that both \f$ x \f$ and \f$ y \f$ are scalars.
   *
   * @tparam X nda::MemoryVector or nda::Scalar type.
   * @tparam Y nda::MemoryVector or nda::Scalar type.
   * @param x Input vector/scalar.
   * @param y Input vector/scalar.
   * @return Vector/scalar result of the dot product.
   */
  template <typename X, typename Y>
    requires((Scalar<X> or MemoryVector<X>) and (Scalar<Y> or MemoryVector<X>))
  auto dotc(X const &x, Y const &y) {
    if constexpr (Scalar<X> or Scalar<Y>) {
      return conj(x) * y;
    } else {
      // compile-time checks
      static_assert(have_same_value_type_v<X, Y>, "Error in nda::blas::dotc: Incompatible value types");
      static_assert(mem::have_compatible_addr_space<X, Y>, "Error in nda::blas::dotc: Incompatible memory address spaces");
      static_assert(is_blas_lapack_v<get_value_t<X>>, "Error in nda::blas::dotc: Value types incompatible with blas");

      // runtime check
      EXPECTS(x.shape() == y.shape());

      if constexpr (!is_complex_v<get_value_t<X>>) {
        return dot(x, y);
      } else if constexpr (mem::have_device_compatible_addr_space<X, Y>) {
#if defined(NDA_HAVE_DEVICE)
        return device::dotc(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
#else
        compile_error_no_gpu();
        return get_value_t<X>(0);
#endif
      } else {
        return f77::dotc(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
      }
    }
  }

  namespace detail {

    // Implementation of the nda::dot_generic and nda::dotc_generic functions.
    template <bool star, typename X, typename Y>
    auto _dot_impl(X const &x, Y const &y) {
      EXPECTS(x.shape() == y.shape());
      long N = x.shape()[0];

      auto _conj = [](auto z) __attribute__((always_inline)) {
        if constexpr (star and is_complex_v<decltype(z)>) {
          return std::conj(z);
        } else
          return z;
      };

      if constexpr (has_layout_smallest_stride_is_one<X> and has_layout_smallest_stride_is_one<Y>) {
        if constexpr (is_regular_or_view_v<X> and is_regular_or_view_v<Y>) {
          auto *__restrict px = x.data();
          auto *__restrict py = y.data();
          auto res            = _conj(px[0]) * py[0];
          for (size_t i = 1; i < N; ++i) { res += _conj(px[i]) * py[i]; }
          return res;
        } else {
          auto res = _conj(x(_linear_index_t{0})) * y(_linear_index_t{0});
          for (long i = 1; i < N; ++i) { res += _conj(x(_linear_index_t{i})) * y(_linear_index_t{i}); }
          return res;
        }
      } else {
        auto res = _conj(x(0)) * y(0);
        for (long i = 1; i < N; ++i) { res += _conj(x(i)) * y(i); }
        return res;
      }
    }

  } // namespace detail

  /**
   * @brief Generic implementation of nda::blas::dot for types not supported by BLAS/LAPACK.
   *
   * @tparam X Vector/Scalar type.
   * @tparam Y Vector/Scalar type.
   * @param x Input vector/scalar.
   * @param y Input vector/scalar.
   * @return Vector/scalar result of the dot product.
   */
  template <typename X, typename Y>
  auto dot_generic(X const &x, Y const &y) {
    if constexpr (Scalar<X> or Scalar<Y>) {
      return x * y;
    } else {
      return detail::_dot_impl<false>(x, y);
    }
  }

  /**
   * @brief Generic implementation of nda::blas::dotc for types not supported by BLAS/LAPACK.
   *
   * @tparam X Vector/Scalar type.
   * @tparam Y Vector/Scalar type.
   * @param x Input vector/scalar.
   * @param y Input vector/scalar.
   * @return Vector/scalar result of the dot product.
   */
  template <typename X, typename Y>
  auto dotc_generic(X const &x, Y const &y) {
    if constexpr (Scalar<X> or Scalar<Y>) {
      return conj(x) * y;
    } else {
      return detail::_dot_impl<true>(x, y);
    }
  }

  /** @} */

} // namespace nda::blas
