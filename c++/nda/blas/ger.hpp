// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the BLAS `ger`, `geru` and `gerc` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../layout_transforms.hpp"
#include "../layout/policies.hpp"
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
   * @brief Interface to the BLAS `ger` and `geru` routine.
   *
   * @details This function performs the rank 1 operation
   * \f[
   *   \mathbf{M} \leftarrow \alpha \mathbf{x} \mathbf{y}^T + \mathbf{M} \; ,
   * \f]
   * where \f$ \alpha \f$ is a scalar, \f$ \mathbf{x} \f$ is an \f$ m \f$ element vector, \f$ \mathbf{y} \f$ is an \f$ n
   * \f$ element vector and \f$ \mathbf{M} \f$ is an \f$ m \times n \f$ matrix.
   * 
   * @note The vector \f$ \mathbf{y} \f$ is never conjugated. Even for complex types. Use nda::blas::gerc for that.
   *
   * @tparam X nda::MemoryVector type.
   * @tparam Y nda::MemoryVector type.
   * @tparam M nda::MemoryMatrix type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param x Input vector \f$ \mathbf{x} \f$ of size \f$ m \f$.
   * @param y Input vector \f$ \mathbf{y} \f$  of size \f$ n \f$.
   * @param m Input/Output matrix \f$ \mathbf{M} \f$  of size \f$ m \times n \f$ to which the outer product is added.
   */
  template <MemoryVector X, MemoryVector Y, MemoryMatrix M>
    requires(have_same_value_type_v<X, Y, M> and mem::have_compatible_addr_space<X, Y, M> and is_blas_lapack_v<get_value_t<X>>)
  void ger(get_value_t<X> alpha, X const &x, Y const &y, M &&m) { // NOLINT (temporary views are allowed here)
    // for C-layout arrays/views, call ger with the transpose and swap x and y
    if constexpr (has_C_layout<M>) {
      ger(alpha, y, x, transpose(m));
      return;
    }

    // check the dimensions of the input/output arrays/views
    EXPECTS(m.extent(0) == x.size());
    EXPECTS(m.extent(1) == y.size());

    // arrays/views must be BLAS compatible
    EXPECTS(m.indexmap().min_stride() == 1);

    // perform actual library call
    if constexpr (mem::have_device_compatible_addr_space<X, Y, M>) {
#if defined(NDA_HAVE_DEVICE)
      device::ger(m.extent(0), m.extent(1), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::ger(m.extent(0), m.extent(1), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
    }
  }

  /**
   * @brief Interface to the BLAS `gerc` routine.
   * 
   * @details This function performs the rank 1 operation
   * \f[
   *   \mathbf{M} \leftarrow \alpha \mathbf{x} \mathbf{y}^H + \mathbf{M} \; ,
   * \f]
   * where \f$ \alpha \f$ is a scalar, \f$ \mathbf{x} \f$ is an \f$ m \f$ element vector, \f$ \mathbf{y} \f$ is an \f$ n
   * \f$ element vector and \f$ \mathbf{M} \f$ is an \f$ m \times n \f$ matrix.
   * 
   * If the value type of the input vectors/matrix is real, it calls nda::blas::ger.
   * 
   * @note \f$ \mathbf{M} \f$ has to be in nda::F_layout.
   *
   * @tparam X nda::MemoryVector type.
   * @tparam Y nda::MemoryVector type.
   * @tparam M nda::MemoryMatrix type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param x Input vector \f$ \mathbf{x} \f$ of size \f$ m \f$.
   * @param y Input vector \f$ \mathbf{y} \f$  of size \f$ n \f$.
   * @param m Input/Output matrix \f$ \mathbf{M} \f$  of size \f$ m \times n \f$ to which the outer product is added.
   */
  template <MemoryVector X, MemoryVector Y, MemoryMatrix M>
    requires(have_same_value_type_v<X, Y, M> and mem::have_compatible_addr_space<X, Y, M> and is_blas_lapack_v<get_value_t<X>>)
  void gerc(get_value_t<X> alpha, X const &x, Y const &y, M &&m) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<M>, "Error in nda::blas::gerc: M must be in Fortran layout");

    // check the dimensions of the input/output arrays/views
    EXPECTS(m.extent(0) == x.size());
    EXPECTS(m.extent(1) == y.size());

    // arrays/views must be BLAS compatible
    EXPECTS(m.indexmap().min_stride() == 1);

    // perform actual library call
    if constexpr (!is_complex_v<get_value_t<X>>) {
      return ger(alpha, x, y, m);
    } else if constexpr (mem::have_device_compatible_addr_space<X, Y, M>) {
#if defined(NDA_HAVE_DEVICE)
      device::gerc(m.extent(0), m.extent(1), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
#else
      compile_error_no_gpu();
#endif
    } else {
      f77::gerc(m.extent(0), m.extent(1), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
    }
  }

  /** @} */

} // namespace nda::blas
