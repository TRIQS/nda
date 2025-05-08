// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS `dot`, `dotu` and `dotc` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif // NDA_HAVE_DEVICE

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Interface to the BLAS `dot` and `dotu` routine.
   *
   * @details This function forms the dot product of two vectors. It calculates
   * \f[
   *   \mathbf{x}^T \mathbf{y} \; .
   * \f]
   *
   * @note The first argument is never conjugated. Even for complex types. Use nda::blas::dotc for that.
   *
   * @tparam X nda::MemoryVector type.
   * @tparam Y nda::MemoryVector type.
   * @param x Input vector \f$ \mathbf{x} \f$.
   * @param y Input vector \f$ \mathbf{y} \f$.
   * @return Result of \f$ \mathbf{x}^T \mathbf{y} \f$.
   */
  template <MemoryVector X, MemoryVector Y>
    requires(have_same_value_type_v<X, Y> and mem::have_compatible_addr_space<X, Y> and is_blas_lapack_v<get_value_t<X>>)
  auto dot(X const &x, Y const &y) {
    // check the dimensions of the input/output arrays/views
    EXPECTS(x.size() == y.size());

    // perform actual library call
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

  /**
   * @brief Interface to the BLAS `dotc` routine.
   *
   * @details This function forms the dot product of two vectors. It calculates
   * \f[
   *   \mathbf{x}^H \mathbf{y} \; .
   * \f]
   *
   * If the value type of the input vectors is real, it calls nda::blas::dot and returns a real result.
   *
   * @tparam X nda::MemoryVector type.
   * @tparam Y nda::MemoryVector type.
   * @param x Input vector \f$ \mathbf{x} \f$.
   * @param y Input vector \f$ \mathbf{y} \f$.
   * @return Result of \f$ \mathbf{x}^H \mathbf{y} \f$.
   */
  template <MemoryVector X, MemoryVector Y>
    requires(have_same_value_type_v<X, Y> and mem::have_compatible_addr_space<X, Y> and is_blas_lapack_v<get_value_t<X>>)
  auto dotc(X const &x, Y const &y) {
    // check the dimensions of the input/output arrays/views
    EXPECTS(x.size() == y.size());

    // perform actual library call
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

  /** @} */

} // namespace nda::blas
