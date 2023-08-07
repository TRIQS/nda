// Copyright (c) 2019-2024 Simons Foundation
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
 * @brief Provides a generic interface to the BLAS `ger` routine and an outer product routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../exceptions.hpp"
#include "../layout_transforms.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../stdutil/array.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif

#include <array>

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Interface to the BLAS `ger` routine.
   *
   * @details This function performs the rank 1 operation
   * \f[
   *   \mathbf{M} \leftarrow \alpha \mathbf{x} \mathbf{y}^H + \mathbf{M} ;,
   * \f]
   * where \f$ \alpha \f$ is a scalar, \f$ \mathbf{x} \f$ is an m element vector, \f$ \mathbf{y} \f$ is an n element
   * vector and \f$ \mathbf{M} \f$ is an m-by-n matrix.
   *
   * @tparam X nda::MemoryVector type.
   * @tparam Y nda::MemoryVector type.
   * @tparam M nda::MemoryMatrix type.
   * @param alpha Input scalar.
   * @param x Input left vector (column vector) of size m.
   * @param y Input right vector (row vector) of size n.
   * @param m Input/Output matrix of size m-by-n to which the outer product is added.
   */
  template <MemoryVector X, MemoryVector Y, MemoryMatrix M>
    requires(have_same_value_type_v<X, Y, M> and mem::have_compatible_addr_space<X, Y, M> and is_blas_lapack_v<get_value_t<X>>)
  void ger(get_value_t<X> alpha, X const &x, Y const &y, M &&m) { // NOLINT (temporary views are allowed here)
    EXPECTS(m.extent(0) == x.extent(0));
    EXPECTS(m.extent(1) == y.extent(0));

    // must be lapack compatible
    EXPECTS(m.indexmap().min_stride() == 1);

    // if in C, we need to call fortran with transposed matrix
    if (has_C_layout<M>) {
      ger(alpha, y, x, transpose(m));
      return;
    }

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
   * Calls ger on a matrix, matrix_view, array, array_view of rank 2
   *  m += alpha * x * ty
   *
   * @tparam X array, array_view of rank 1
   * @tparam Y array, array_view of rank 1
   * @tparam M matrix, matrix_view, array, array_view of rank 2
   * @param alpha
   * @param x 
   * @param y
   * @param m The result. Can be a temporary view. 
   *         
   * @StaticPrecondition : X, Y, M have the same value_type and it is complex<double> or double         
   * @Precondition : 
   *       * m has the correct dimension given a, b. 
   */
  template <MemoryVector X, MemoryVector Y, MemoryMatrix M>
  requires(have_same_value_type_v<X, Y, M> and mem::have_compatible_addr_space<X, Y, M> and is_blas_lapack_v<get_value_t<X>>)
  void gerc(get_value_t<X> alpha, X const &x, Y const &y, M &&m) {

    EXPECTS(m.extent(0) == x.extent(0));
    EXPECTS(m.extent(1) == y.extent(0));
    // Must be lapack compatible
    EXPECTS(m.indexmap().min_stride() == 1);

    // if in C, we need to call fortran with transposed matrix
    if (has_C_layout<M>) {
      gerc(alpha, y, x, transpose(m));
      return;
    }

    if constexpr (mem::have_device_compatible_addr_space<X,Y,M>) {
#if defined(NDA_HAVE_DEVICE)
      device::gerc(m.extent(0), m.extent(1), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
#else
      static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
#endif
    } else {
      f77::gerc(m.extent(0), m.extent(1), alpha, x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0], m.data(), get_ld(m));
    }
  }

  /**
   * @brief Calculate the outer product of two contiguous arrays/views/scalars.
   *
   * @details For general multidimensional arrays/views, it calculates their tensor outer product, i.e.
   * ```
   * c(i,j,k,...,u,v,w,...) = a(i,j,k,...) * b(u,v,w,...)
   * ```
   * If one of the arguments is a scalar, it multiplies each element of the other argument by the scalar which returns a
   * lazy nda::expr object.
   *
   * If both arguments are scalars, it returns their products.
   *
   * @tparam A nda::ArrayOrScalar type.
   * @tparam B nda::ArrayOrScalar type.
   * @param a Input array/scalar.
   * @param b Input array/scalar.
   * @return (Lazy) Outer product.
   */
  template <ArrayOrScalar A, ArrayOrScalar B>
  auto outer_product(A const &a, B const &b) {
    if constexpr (Scalar<A> or Scalar<B>) {
      return a * b;
    } else {
      if (not a.is_contiguous()) NDA_RUNTIME_ERROR << "Error in nda::blas::outer_product: First argument has non-contiguous layout";
      if (not b.is_contiguous()) NDA_RUNTIME_ERROR << "Error in nda::blas::outer_product: Second argument has non-contiguous layout";

      // use BLAS ger to calculate the outer product
      auto res   = zeros<get_value_t<A>, mem::common_addr_space<A, B>>(stdutil::join(a.shape(), b.shape()));
      auto a_vec = reshape(a, std::array{a.size()});
      auto b_vec = reshape(b, std::array{b.size()});
      auto mat   = reshape(res, std::array{a.size(), b.size()});
      ger(1.0, a_vec, b_vec, mat);

      return res;
    }
  }

  /** @} */

} // namespace nda::blas
