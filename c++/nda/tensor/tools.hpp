// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides various traits and utilities for the tensor interface.
 */

#pragma once

#include "../blas/tools.hpp"
#include "../exceptions.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

namespace nda::tensor {

  // Import tools from the blas_lapack namespace.
  using namespace nda::blas_lapack;

  /**
   * @addtogroup tensor_utils
   * @{
   */

  /// Constexpr variable that is true if nda is configured cuTENSOR support.
#if defined(NDA_HAVE_CUDA) && defined(NDA_HAVE_CUTENSOR)
  static constexpr bool have_cutensor = true;
#else
  static constexpr bool have_cutensor = false;
#endif // NDA_HAVE_CUTENSOR

  /// Constexpr variable that is true if nda is configured with TBLIS support.
#ifdef NDA_HAVE_TBLIS
  static constexpr bool have_tblis = true;
#else
  static constexpr bool have_tblis = false;
#endif // NDA_HAVE_TBLIS

  /// Data pointer type of an nda::blas_lapack::BlasArrayOrConj.
  template <BlasArrayOrConj A>
  using data_ptr_t = decltype(get_array(std::declval<A>()).data());

  /**
   * @brief Binary operations for tensor operations.
   *
   * @details The binary operations are mapped to TBLIS and cuTENSOR as follows:
   * - `binary_op::SUM` -> `REDUCE_SUM` or `CUTENSOR_OP_ADD`
   * - `binary_op::PROD` -> `CUTENSOR_OP_MUL` (N/A in TBLIS)
   * - `binary_op::SUM_ABS` -> `REDUCE_SUM_ABS` (N/A in cuTENSOR)
   * - `binary_op::MAX` -> `REDUCE_MAX` / `CUTENSOR_OP_MAX`
   * - `binary_op::MAX_ABS` -> `REDUCE_MAX_ABS` (N/A in cuTENSOR)
   * - `binary_op::MIN` -> `REDUCE_MIN` / `CUTENSOR_OP_MIN`
   * - `binary_op::MIN_ABS` -> `REDUCE_MIN_ABS` (N/A in cuTENSOR)
   * - `binary_op::NORM_2` -> `REDUCE_NORM_2` (N/A in cuTENSOR)
   */
  enum class binary_op : std::uint8_t { SUM, PROD, SUM_ABS, MAX, MAX_ABS, MIN, MIN_ABS, NORM_2 };

  // clang-format off
  /**
   * @brief Unary element-wise operations for tensor operations.
   *
   * @details The unary operations are mapped to cuTENSOR's element-wise operators (`cutensorOperator_t`) as follows:
   * - `unary_op::IDENTITY` -> `CUTENSOR_OP_IDENTITY` (elements are not changed)
   * - `unary_op::SQRT` -> `CUTENSOR_OP_SQRT` (square root)
   * - `unary_op::RELU` -> `CUTENSOR_OP_RELU` (rectified linear unit)
   * - `unary_op::CONJ` -> `CUTENSOR_OP_CONJ` (complex conjugate)
   * - `unary_op::RCP` -> `CUTENSOR_OP_RCP` (reciprocal)
   * - `unary_op::SIGMOID` -> `CUTENSOR_OP_SIGMOID` (y = 1/(1+exp(-x)))
   * - `unary_op::TANH` -> `CUTENSOR_OP_TANH` (y = tanh(x))
   * - `unary_op::EXP` -> `CUTENSOR_OP_EXP` (exponentiation)
   * - `unary_op::LOG` -> `CUTENSOR_OP_LOG` (log base e)
   * - `unary_op::ABS` -> `CUTENSOR_OP_ABS` (absolute value)
   * - `unary_op::NEG` -> `CUTENSOR_OP_NEG` (negation)
   * - `unary_op::SIN` -> `CUTENSOR_OP_SIN` (sine)
   * - `unary_op::COS` -> `CUTENSOR_OP_COS` (cosine)
   * - `unary_op::TAN` -> `CUTENSOR_OP_TAN` (tangent)
   * - `unary_op::SINH` -> `CUTENSOR_OP_SINH` (hyperbolic sine)
   * - `unary_op::COSH` -> `CUTENSOR_OP_COSH` (hyperbolic cosine)
   * - `unary_op::ASIN` -> `CUTENSOR_OP_ASIN` (inverse sine)
   * - `unary_op::ACOS` -> `CUTENSOR_OP_ACOS` (inverse cosine)
   * - `unary_op::ATAN` -> `CUTENSOR_OP_ATAN` (inverse tangent)
   * - `unary_op::ASINH` -> `CUTENSOR_OP_ASINH` (inverse hyperbolic sine)
   * - `unary_op::ACOSH` -> `CUTENSOR_OP_ACOSH` (inverse hyperbolic cosine)
   * - `unary_op::ATANH` -> `CUTENSOR_OP_ATANH` (inverse hyperbolic tangent)
   * - `unary_op::CEIL` -> `CUTENSOR_OP_CEIL` (ceiling)
   * - `unary_op::FLOOR` -> `CUTENSOR_OP_FLOOR` (floor)
   * - `unary_op::MISH` -> `CUTENSOR_OP_MISH` (mish, y = x*tanh(softplus(x)))
   * - `unary_op::SWISH` -> `CUTENSOR_OP_SWISH` (swish, y = x*sigmoid(x))
   * - `unary_op::SOFT_PLUS` -> `CUTENSOR_OP_SOFT_PLUS` (softplus, y = log(exp(x)+1))
   * - `unary_op::SOFT_SIGN` -> `CUTENSOR_OP_SOFT_SIGN` (softsign, y = x/(abs(x)+1))
   */
  enum class unary_op : std::uint8_t {
    IDENTITY, SQRT, RELU, CONJ, RCP, SIGMOID, TANH, EXP, LOG, ABS, NEG,
    SIN, COS, TAN, SINH, COSH, ASIN, ACOS, ATAN, ASINH, ACOSH, ATANH,
    CEIL, FLOOR, MISH, SWISH, SOFT_PLUS, SOFT_SIGN
  };
  // clang-format on

  namespace detail {

    // Apply a nda::tensor::binary_op to two scalar operands.
    template <typename T>
    T apply_binary(binary_op op, T x, T y) {
      switch (op) {
        case binary_op::SUM: return x + y;
        case binary_op::PROD: return x * y;
        case binary_op::SUM_ABS: return std::abs(x) + std::abs(y);
        case binary_op::MAX_ABS: return std::max(std::abs(x), std::abs(y));
        case binary_op::MIN_ABS: return std::min(std::abs(x), std::abs(y));
        case binary_op::NORM_2: return std::sqrt(std::norm(x) + std::norm(y));
        case binary_op::MAX:
        case binary_op::MIN:
          if constexpr (!is_complex_v<T>) {
            return (op == binary_op::MAX ? std::max(x, y) : std::min(x, y));
          } else {
            NDA_RUNTIME_ERROR << "nda::tensor: binary_op::MAX/MIN are unsupported for complex value types";
          }
      }
      return T{}; // unreachable
    }

  } // namespace detail

  /**
   * @brief A type-erased, non-owning view of an nda::MemoryArray or a conjugate lazy expression.
   *
   * @details It holds a pointer to the data, extents and strides of the viewed array or conjugate expression, as well
   * as its rank and an element-wise unary operation to apply to each element (see nda::tensor::unary_op).
   *
   * All pointers are non-owning and must remain valid for the lifetime of this object.
   *
   * @tparam T Value type of the tensor.
   */
  template <typename T>
  struct tensor_view {
    /// Value type of the tensor (can be const).
    using value_type = T;

    /// Pointer to the tensor data.
    T *data = nullptr;

    /// Pointer to the array of extents.
    const long *extents = nullptr;

    /// Pointer to the array of strides.
    const long *strides = nullptr;

    /// Number of dimensions (rank) of the tensor.
    int ndim = 0;

    /// Element-wise unary operation to apply.
    unary_op op = unary_op::IDENTITY;

    /// Default constructor initializes an empty view.
    tensor_view() = default;

    /**
     * @brief Construct a rank-0 tensor view from a pointer to a scalar value.
     *
     * @details This creates a tensor view with `ndim == 0` and `extents == strides == nullptr`, viewing the scalar as a
     * rank-0 tensor. The pointer must remain valid for the lifetime of this view.
     *
     * @param p Pointer to the scalar value.
     */
    tensor_view(T *p) : data(p) {}

    /**
     * @brief Construct a tensor view from an nda::MemoryArray and an nda::tensor::unary_op.
     *
     * @details The value type is deduced from the data pointer of the (underlying) array which can be const or
     * non-const.
     *
     * @tparam A nda::blas_lapack::BlasArrayOrConj type.
     * @param a Array or view to wrap.
     * @param op Unary operation to apply to each element.
     */
    template <BlasArray A>
      requires std::convertible_to<data_ptr_t<A>, T *>
    tensor_view(A &&a, unary_op op) // NOLINT
       : data(get_array(a).data()),
         extents(get_array(a).indexmap().lengths().data()),
         strides(get_array(a).indexmap().strides().data()),
         ndim(get_rank<decltype(get_array(a))>),
         op(op) {}

    /**
     * @brief Construct a tensor view from an nda::MemoryArray or a conjugate lazy expression.
     *
     * @details The value type is deduced from the data pointer of the (underlying) array which can be const or
     * non-const.
     *
     * @tparam A nda::blas_lapack::BlasArrayOrConj type.
     * @param a Array, view or conjugate expression to wrap.
     */
    template <BlasArrayOrConj A>
      requires std::convertible_to<data_ptr_t<A>, T *>
    tensor_view(A &&a) : tensor_view(get_array(a), is_conj_array_expr<A> ? unary_op::CONJ : unary_op::IDENTITY) {} // NOLINT

    /**
     * @brief Construct a tensor view from from another tensor view with a convertible value type.
     *
     * @details This enables implicit conversion from `tensor_view<T>` to `tensor_view<const T>`, analogous to `T*`
     * converting to `const T*`.
     *
     * @tparam U Value type of the source.
     * @param tv Source tensor view.
     */
    template <typename U>
      requires(!std::same_as<U, T> && std::convertible_to<U *, T *>)
    tensor_view(tensor_view<U> tv) // NOLINT
       : data(tv.data), extents(tv.extents), strides(tv.strides), ndim(tv.ndim), op(tv.op) {}
  };

  // Deduction guide: tensor_view(array) deduces T from the data pointer of the (underlying) array.
  template <BlasArray A>
  tensor_view(A &&, unary_op) -> tensor_view<std::remove_pointer_t<data_ptr_t<A>>>;

  template <BlasArrayOrConj A>
  tensor_view(A &&) -> tensor_view<std::remove_pointer_t<data_ptr_t<A>>>;

  /// Alias for a tensor_view with const value type.
  template <typename T>
  using const_tensor_view = tensor_view<const T>;

} // namespace nda::tensor
