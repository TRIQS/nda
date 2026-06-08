// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides lazy expressions for nda::Array types.
 */

#pragma once

#include "./concepts.hpp"
#include "./declarations.hpp"
#include "./linalg/inv.hpp"
#include "./linalg/matmul.hpp"
#include "./linalg/matvecmul.hpp"
#include "./macros.hpp"
#include "./stdutil/complex.hpp"
#include "./traits.hpp"
#include "./simd.hpp"

#include <functional>
#include <type_traits>
#include <utility>

#ifdef NDA_ENFORCE_BOUNDCHECK
#include "./exceptions.hpp"
#endif // NDA_ENFORCE_BOUNDCHECK

namespace nda {

  /**
   * @addtogroup av_ops
   * @{
   */

  /**
   * @brief Lazy unary expression for nda::Array types.
   *
   * @details A lazy unary expression contains a single operand and a unary operation. It fulfills the nda::Array
   * concept and can therefore be used in any other expression or function that expects an nda::Array type.
   *
   * The only supported unary operation is the negation operation ('-').
   *
   * @tparam OP Char representing the unary operation.
   * @tparam A nda::Array type.
   */
  template <char OP, Array A>
  struct expr_unary {
    static_assert(OP == '-', "Error in nda::expr_unary: Only negation is supported");

    /// nda::Array object.
    A a;

    /**
     * @brief Function call operator.
     *
     * @details Forwards the arguments to the nda::Array operand and negates the result.
     *
     * @tparam Args Types of the arguments.
     * @param args Function call arguments.
     * @return If the result of the forwarded function call is another nda::Array, a new lazy expression is returned.
     * Otherwise the result is negated and returned.
     */
    template <typename... Args>
    auto operator()(Args &&...args) const {
      return -a(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto load(auto simd_tag, Args &&...args) const {
      static_assert(std::is_same_v<decltype(simd_tag), simd::vectorize_t> or std::is_same_v<decltype(simd_tag), simd::emulate_t>,
                    "Load tag can only be vectorize or emulate");
      return -(a.load(simd_tag, std::forward<Args>(args)...));
    }

    /**
     * @brief Get the shape of the nda::Array operand.
     * @return `std::array<long, Rank>` object specifying the shape of the operand.
     */
    [[nodiscard]] constexpr auto shape() const { return a.shape(); }

    /**
     * @brief Get the total size of the nda::Array operand.
     * @return Number of elements contained in the operand.
     */
    [[nodiscard]] constexpr long size() const { return a.size(); }
  };

  /**
   * @brief Lazy binary expression for nda::ArrayOrScalar types.
   *
   * @details A lazy binary expression contains two operands and a binary operation. It fulfills the nda::Array
   * concept and can therefore be used in any other expression or function that expects an nda::Array type.
   *
   * The supported binary operations are addition ('+'), subtraction ('-'), multiplication ('*') and division ('/').
   *
   * @tparam OP Char representing the binary operation.
   * @tparam L nda::ArrayOrScalar type of left-hand side.
   * @tparam R nda::ArrayOrScalar type of right-hand side.
   */
  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  struct expr {
    /// nda::ArrayOrScalar left hand side operand.
    L l;

    /// nda::ArrayOrScalar right hand side operand.
    R r;

    /// Decay type of the left hand side operand.
    using L_t = std::decay_t<L>;

    /// Decay type of the right hand side operand.
    using R_t = std::decay_t<R>;

    // FIXME : we should use is_scalar_for_v but the trait needs work to accommodate scalar L or R
    /// Constexpr variable that is true if the left hand side operand is a scalar.
    static constexpr bool l_is_scalar = nda::is_scalar_v<L>;

    /// Constexpr variable that is true if the right hand side operand is a scalar.
    static constexpr bool r_is_scalar = nda::is_scalar_v<R>;

    /// Constexpr variable specifying the algebra of one of the non-scalar operands.
    static constexpr char algebra = (l_is_scalar ? get_algebra<R> : get_algebra<L>);

    /**
     * @brief Compute the layout information of the expression.
     * @return nda::layout_info_t object.
     */
    static constexpr layout_info_t compute_layout_info() {
      if (l_is_scalar)
        return get_layout_info<R>; //(algebra == 'A' ? get_layout_info<R> : layout_info_t{}); // 1 as an array has all flags, it is just 1
      if (r_is_scalar)
        return get_layout_info<L>; //(algebra == 'A' ? get_layout_info<L> : layout_info_t{}); // 1 as a matrix does not, as it is diagonal only.
      return get_layout_info<R> & get_layout_info<L>; // default case. Take the logical and of all flags
    }

    /**
     * @brief Get the shape of the expression (result of the operation).
     * @return `std::array<long, Rank>` object specifying the shape of the expression.
     */
    [[nodiscard]] constexpr decltype(auto) shape() const {
      if constexpr (l_is_scalar) {
        return r.shape();
      } else if constexpr (r_is_scalar) {
        return l.shape();
      } else {
        EXPECTS(l.shape() == r.shape());
        return l.shape();
      }
    }

    /**
     * @brief Get the total size of the expression (result of the operation).
     * @return Number of elements contained in the expression.
     */
    [[nodiscard]] constexpr long size() const {
      if constexpr (l_is_scalar) {
        return r.size();
      } else if constexpr (r_is_scalar) {
        return l.size();
      } else {
        EXPECTS(l.size() == r.size());
        return l.size();
      }
    }

    /**
     * @brief Function call operator.
     *
     * @details Forwards the arguments to the nda::Array operands and performs the binary operation.
     *
     * @tparam Args Types of the arguments.
     * @param args Function call arguments.
     * @return If the result of the forwarded function calls contains another nda::Array, a new lazy expression is
     * returned. Otherwise the result of the binary operation is returned.
     */
    template <typename... Args>
    auto operator()(Args const &...args) const {
      // addition
      if constexpr (OP == '+') {
        if constexpr (l_is_scalar) {
          // lhs is a scalar
          if constexpr (algebra == 'M')
            // rhs is a matrix
            return (std::equal_to{}(args...) ? l + r(args...) : r(args...));
          else
            // rhs is an array
            return l + r(args...);
        } else if constexpr (r_is_scalar) {
          // rhs is a scalar
          if constexpr (algebra == 'M')
            // lhs is a matrix
            return (std::equal_to{}(args...) ? l(args...) + r : l(args...));
          else
            // lhs is an array
            return l(args...) + r;
        } else
          // both are arrays or matrices
          return l(args...) + r(args...);
      }

      // subtraction
      if constexpr (OP == '-') {
        if constexpr (l_is_scalar) {
          // lhs is a scalar
          if constexpr (algebra == 'M')
            // rhs is a matrix
            return (std::equal_to{}(args...) ? l - r(args...) : -r(args...));
          else
            // rhs is an array
            return l - r(args...);
        } else if constexpr (r_is_scalar) {
          // rhs is a scalar
          if constexpr (algebra == 'M')
            // lhs is a matrix
            return (std::equal_to{}(args...) ? l(args...) - r : l(args...));
          else
            // lhs is an array
            return l(args...) - r;
        } else
          // both are arrays or matrices
          return l(args...) - r(args...);
      }

      // multiplication
      if constexpr (OP == '*') {
        if constexpr (l_is_scalar)
          // lhs is a scalar
          return l * r(args...);
        else if constexpr (r_is_scalar)
          // rhs is a scalar
          return l(args...) * r;
        else {
          // both are arrays (matrix product is not supported here)
          static_assert(algebra != 'M', "Error in nda::expr: Matrix algebra not supported");
          return l(args...) * r(args...);
        }
      }

      // division
      if constexpr (OP == '/') {
        if constexpr (l_is_scalar) {
          // lhs is a scalar
          static_assert(algebra != 'M', "Error in nda::expr: Matrix algebra not supported");
          return l / r(args...);
        } else if constexpr (r_is_scalar)
          // rhs is a scalar
          return l(args...) / r;
        else {
          // both are arrays (matrix division is not supported here)
          static_assert(algebra != 'M', "Error in nda::expr: Matrix algebra not supported");
          return l(args...) / r(args...);
        }
      }
    }

    /**
     * @brief Subscript operator.
     *
     * @details Simply forwards the argument to the function call operator.
     *
     * @tparam Arg Type of the argument.
     * @param arg Subscript argument.
     * @return Result of the corresponding function call.
     */
    template <typename Arg>
    auto operator[](Arg &&arg) const {
      static_assert(get_rank<expr> == 1, "Error in nda::expr: Subscript operator only available for expressions of rank 1");
      return operator()(std::forward<Arg>(arg));
    }

#ifdef NDA_HAVE_XSIMD
    private:
    template <typename Tag, typename... Args>
    auto _call_load(Args const &...args) const {
      using dispatch_t = Tag;
      /**
     * This lambda implements a critical optimization for matrix/scalar
     * binary operations (e.g., `M + s` or `s - M`) where `OP` is '+' or '-'.
     *
     * In an expression like `matrix + scalar`, the scalar value is only
     * added to the diagonal elements of the matrix.
     *
     * This lambda is called by the expression's `load` function, which loads
     * data one SIMD block at a time. The parameters `(i, j)` are the
     * row/column indices of the *start* of the SIMD block being loaded.
     *
     * This function's job is to:
     * 1. Detect if the current SIMD block contains a diagonal element.
     * 2. If it does, create a temporary SIMD vector that has the scalar
     * value *only* at that diagonal element's position (e.g., `[0, 0, s, 0]`).
     * 3. Add or subtract this temporary vector from the matrix data.
     * 4. If the block contains no diagonal element, just return the matrix data.
     */
      auto diagonal_simd = [this](long i, long j) {
        // This lambda is only valid for Matrix + Scalar operations. This if constexpr is needed so that we dont have compile errors for other cases.
        if constexpr (sizeof...(Args) == 2 and (Vectorizable<L_t> or Vectorizable<R_t>)) {

          // Calculate the 'diagonal index'.
          // For a C-layout (row-major) load starting at `(i, j)`, the vector
          // loads elements `(i, j), (i, j+1), (i, j+2), ...`.
          // A diagonal element `(i, i)` would be at index `k = i - j`
          // (i.e., we are at element `(i, j+k)` and we need `j+k == i`).
          long diff = i - j;

          // Handle F-layout (column-major).
          // For an F-layout load starting at `(i, j)`, the vector loads
          // `(i, j), (i+1, j), (i+2, j), ...`.
          // A diagonal element `(j, j)` would be at index `k = j - i`
          // (i.e., we are at element `(i+k, j)` and we need `i+k == j`).
          // This flips the sign of our 'diff', so we correct for it here.
          if constexpr ((l_is_scalar and get_layout_info<R>.stride_order == Fortran_stride_order<2>)
                        or (r_is_scalar and get_layout_info<L>.stride_order == Fortran_stride_order<2>)) {
            diff = -diff;
          }
          if constexpr (l_is_scalar) {
            using simd_t = native_simd<L_t>;
            // Check if the diagonal element's index `diff` is outside the bounds of our SIMD chunk
            if (diff < 0 or diff > simd_t::size - 1) return r.load(dispatch_t{}, i, j);
            // A diagonal element is in this block at index `diff`.
            // Create a temporary, zero-initialized array on the stack.
            alignas(simd_t::arch_type::alignment()) std::array<L_t, simd_t::size> tmp{};
            // Place the scalar value `l` only at the diagonal position.
            tmp[diff] = l;
            if constexpr (OP == '+') {
              return r.load(dispatch_t{}, i, j) + simd_t::load_aligned(tmp.data());
            } else {
              return r.load(dispatch_t{}, i, j) - simd_t::load_aligned(tmp.data());
            }
          } else if constexpr (r_is_scalar) {
            using simd_t = native_simd<R_t>;
            // Check if the diagonal element's index `diff` is outside the bounds of our SIMD chunk
            if (diff < 0 or diff > simd_t::size - 1) return l.load(dispatch_t{}, i, j);
            // A diagonal element is in this block at index `diff`.
            // Create a temporary, zero-initialized array on the stack.
            alignas(simd_t::arch_type::alignment()) std::array<R_t, simd_t::size> tmp{};
            // Place the scalar value `r` only at the diagonal position.
            tmp[diff] = r;
            if constexpr (OP == '+') {
              return l.load(dispatch_t{}, i, j) + simd_t::load_aligned(tmp.data());
            } else {
              return l.load(dispatch_t{}, i, j) - simd_t::load_aligned(tmp.data());
            }
          }
        }
      };
      // addition
      if constexpr (OP == '+') {
        if constexpr (l_is_scalar) {
          // lhs is a scalar
          if constexpr (algebra == 'M')
            // rhs is a matrix
            return diagonal_simd(args...);
          else
            // rhs is an array
            return native_simd<L_t>(l) + r.load(dispatch_t{}, args...);
        } else if constexpr (r_is_scalar) {
          // rhs is a scalar
          if constexpr (algebra == 'M') {
            // lhs is a matrix
            return diagonal_simd(args...);
          } else
            // lhs is an array
            return l.load(dispatch_t{}, args...) + native_simd<R_t>(r);
        } else
          // both are arrays or matrices
          return l.load(dispatch_t{}, args...) + r.load(dispatch_t{}, args...);
      }

      // subtraction
      if constexpr (OP == '-') {
        if constexpr (l_is_scalar) {
          // lhs is a scalar
          if constexpr (algebra == 'M')
            // rhs is a matrix
            return diagonal_simd(args...);
          else
            // rhs is an array
            return native_simd<L_t>(l) - r.load(dispatch_t{}, args...);
        } else if constexpr (r_is_scalar) {
          // rhs is a scalar
          if constexpr (algebra == 'M')
            // lhs is a matrix
            return diagonal_simd(args...);
          else
            // lhs is an array
            return l.load(dispatch_t{}, args...) - native_simd<R_t>(r);
        } else
          // both are arrays or matrices
          return l.load(dispatch_t{}, args...) - r.load(dispatch_t{}, args...);
      }

      // multiplication
      if constexpr (OP == '*') {
        if constexpr (l_is_scalar)
          // lhs is a scalar
          return native_simd<L_t>(l) * r.load(dispatch_t{}, args...);
        else if constexpr (r_is_scalar)
          // rhs is a scalar
          return l.load(dispatch_t{}, args...) * native_simd<R_t>(r);
        else {
          // both are arrays (matrix product is not supported here)
          static_assert(algebra != 'M', "Error in nda::expr: Matrix algebra not supported");
          return l.load(dispatch_t{}, args...) * r.load(dispatch_t{}, args...);
        }
      }

      // division
      if constexpr (OP == '/') {
        if constexpr (l_is_scalar) {
          // lhs is a scalar
          static_assert(algebra != 'M', "Error in nda::expr: Matrix algebra not supported");
          return native_simd<L_t>(l) / r.load(dispatch_t{}, args...);
        } else if constexpr (r_is_scalar)
          // rhs is a scalar
          return l.load(dispatch_t{}, args...) / native_simd<R_t>(r);
        else {
          // both are arrays (matrix division is not supported here)
          static_assert(algebra != 'M', "Error in nda::expr: Matrix algebra not supported");
          return l.load(dispatch_t{}, args...) / r.load(dispatch_t{}, args...);
        }
      }
    }

    public:
    template <typename... Args>
    auto load(auto simd_tag, Args const &...args) const {
      static_assert(std::is_same_v<decltype(simd_tag), simd::vectorize_t> or std::is_same_v<decltype(simd_tag), simd::emulate_t>,
                    "Load tag can only be vectorize or emulate");
      return _call_load<decltype(simd_tag)>(args...);
    }
#endif // NDA_HAVE_XSIMD
  };

  /**
   * @brief Unary minus operator for nda::Array types.
   *
   * @details It performs lazy elementwise negation.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array operand.
   * @return Lazy unary expression for the negation operation.
   */
  template <Array A>
  expr_unary<'-', A> operator-(A &&a) {
    return {std::forward<A>(a)};
  }

  /**
   * @brief Addition operator for two nda::Array types.
   *
   * @details It performs lazy elementwise addition.
   *
   * @tparam L nda::Array type of left hand side.
   * @tparam R nda::Array type of right hand side.
   * @param l nda::Array left hand side operand.
   * @param r nda::Array right hand side operand.
   * @return Lazy binary expression for the addition operation.
   */
  template <Array L, Array R>
  Array auto operator+(L &&l, R &&r) {
    static_assert(get_rank<L> == get_rank<R>, "Error in lazy nda::operator+: Rank mismatch");
    return expr<'+', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  /**
   * @brief Addition operator for an nda::Array and an nda::Scalar.
   *
   * @details Depending on the algebra of the nda::Array, it performs the following lazy operations:
   * - 'A': Elementwise addition.
   * - 'M': Addition of the nda::Scalar to the elements on the shorter diagonal of the matrix.
   *
   * @tparam A nda::Array type.
   * @tparam S nda::Scalar type.
   * @param a nda::Array left hand side operand.
   * @param s nda::Scalar right hand side operand.
   * @return Lazy binary expression for the addition operation.
   */
  template <Array A, Scalar S>
  Array auto operator+(A &&a, S &&s) { // NOLINT (S&& is mandatory for proper concept Array <: typename to work)
    return expr<'+', A, std::decay_t<S>>{std::forward<A>(a), s};
  }

  /**
   * @brief Addition operator for an nda::Scalar and an nda::Array.
   *
   * @details Depending on the algebra of the nda::Array, it performs the following lazy operations:
   * - 'A': Elementwise addition.
   * - 'M': Addition of the nda::Scalar to the elements on the shorter diagonal of the matrix.
   *
   * @tparam S nda::Scalar type.
   * @tparam A nda::Array type.
   * @param s nda::Scalar left hand side operand.
   * @param a nda::Array right hand side operand.
   * @return Lazy binary expression for the addition operation.
   */
  template <Scalar S, Array A>
  Array auto operator+(S &&s, A &&a) { // NOLINT (S&& is mandatory for proper concept Array <: typename to work)
    return expr<'+', std::decay_t<S>, A>{s, std::forward<A>(a)};
  }

  /**
   * @brief Subtraction operator for two nda::Array types.
   *
   * @details It performs lazy elementwise subtraction.
   *
   * @tparam L nda::Array type of left hand side.
   * @tparam R nda::Array type of right hand side.
   * @param l nda::Array left hand side operand.
   * @param r nda::Array right hand side operand.
   * @return Lazy binary expression for the subtraction operation.
   */
  template <Array L, Array R>
  Array auto operator-(L &&l, R &&r) {
    static_assert(get_rank<L> == get_rank<R>, "Error in lazy nda::operator-: Rank mismatch");
    return expr<'-', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  /**
   * @brief Subtraction operator for an nda::Array and an nda::Scalar.
   *
   * @details Depending on the algebra of the nda::Array, it performs the following lazy operations:
   * - 'A': Elementwise subtraction.
   * - 'M': Subtraction of the nda::Scalar from the elements on the shorter diagonal of the matrix.
   *
   * @tparam A nda::Array type.
   * @tparam S nda::Scalar type.
   * @param a nda::Array left hand side operand.
   * @param s nda::Scalar right hand side operand.
   * @return Lazy binary expression for the subtraction operation.
   */
  template <Array A, Scalar S>
  Array auto operator-(A &&a, S &&s) { // NOLINT (S&& is mandatory for proper concept Array <: typename to work)
    return expr<'-', A, std::decay_t<S>>{std::forward<A>(a), s};
  }

  /**
   * @brief Subtraction operator for an nda::Scalar and an nda::Array.
   *
   * @details Depending on the algebra of the nda::Array, it performs the following lazy operations:
   * - 'A': Elementwise subtraction.
   * - 'M': Subtraction of the elements on the shorter diagonal of the matrix from the nda::Scalar.
   *
   * @tparam S nda::Scalar type.
   * @tparam A nda::Array type.
   * @param s nda::Scalar left hand side operand.
   * @param a nda::Array right hand side operand.
   * @return Lazy binary expression for the subtraction operation.
   */
  template <Scalar S, Array A>
  Array auto operator-(S &&s, A &&a) { // NOLINT (S&& is mandatory for proper concept Array <: typename to work)
    return expr<'-', std::decay_t<S>, A>{s, std::forward<A>(a)};
  }

  /**
   * @brief Multiplication operator for two nda::Array types.
   *
   * @details The input arrays must have one of the following algebras:
   * - 'A' * 'A': Elementwise multiplication of two arrays returns a lazy nda::expr object.
   * - 'M' * 'M': Matrix-matrix multiplication calls nda::matmul and returns the result.
   * - 'M' * 'V': Matrix-vector multiplication calls nda::matvecmul and returns the result.
   *
   * Obvious restrictions on the ranks and shapes of the input arrays apply.
   *
   * @tparam L nda::Array type of left hand side.
   * @tparam R nda::Array type of right hand side.
   * @param l nda::Array left hand side operand.
   * @param r nda::Array right hand side operand.
   * @return Either a lazy binary expression for the multiplication operation ('A' * 'A') or the result
   * of the matrix-matrix or matrix-vector multiplication.
   */
  template <Array L, Array R>
  auto operator*(L &&l, R &&r) {
    // allowed algebras: A * A or M * M or M * V
    static constexpr char l_algebra = get_algebra<L>;
    static constexpr char r_algebra = get_algebra<R>;
    static_assert(l_algebra != 'V', "Error in nda::operator*: Can not multiply vector by an array or a matrix");

    // two arrays: A * A
    if constexpr (l_algebra == 'A') {
      static_assert(r_algebra == 'A', "Error in nda::operator*: Both types need to be arrays");
      static_assert(get_rank<L> == get_rank<R>, "Error in nda::operator*: Rank mismatch");
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.shape() != r.shape()) NDA_RUNTIME_ERROR << "Error in nda::operator*: Dimension mismatch: " << l.shape() << " != " << r.shape();
#endif
      return expr<'*', L, R>{std::forward<L>(l), std::forward<R>(r)};
    }

    // two matrices: M * M
    if constexpr (l_algebra == 'M') {
      static_assert(r_algebra != 'A', "Error in nda::operator*: Can not multiply a matrix by an array");
      if constexpr (r_algebra == 'M')
        // matrix * matrix
        return linalg::matmul(std::forward<L>(l), std::forward<R>(r));
      else
        // matrix * vector
        return linalg::matvecmul(std::forward<L>(l), std::forward<R>(r));
    }
  }

  /**
   * @brief Multiplication operator for an nda::Array and an nda::Scalar.
   *
   * @details It performs lazy elementwise multiplication.
   *
   * @tparam A nda::Array type.
   * @tparam S nda::Scalar type.
   * @param a nda::Array left hand side operand.
   * @param s nda::Scalar right hand side operand.
   * @return Lazy binary expression for the multiplication operation.
   */
  template <Array A, Scalar S>
  Array auto operator*(A &&a, S &&s) { // NOLINT (S&& is mandatory for proper concept Array <: typename to work)
    return expr<'*', A, std::decay_t<S>>{std::forward<A>(a), s};
  }

  /**
   * @brief Multiplication operator for an nda::Scalar and an nda::Array.
   *
   * @details It performs elementwise multiplication.
   *
   * @tparam S nda::Scalar type.
   * @tparam A nda::Array type.
   * @param s nda::Scalar left hand side operand.
   * @param a nda::Array right hand side operand.
   * @return Lazy binary expression for the multiplication operation.
   */
  template <Scalar S, Array A>
  Array auto operator*(S &&s, A &&a) { // NOLINT (S&& is mandatory for proper concept Array <: typename to work)
    return expr<'*', std::decay_t<S>, A>{s, std::forward<A>(a)};
  }

  /**
   * @brief Division operator for two nda::Array types.
   *
   * @details The input arrays must have one of the following algebras:
   * - 'A' / 'A': Elementwise division of two arrays returns a lazy nda::expr object.
   * - 'M' / 'M': Multiplies the lhs matrix with the inverse of the rhs matrix and returns the result.
   *
   * Obvious restrictions on the ranks and shapes of the input arrays apply.
   *
   * @tparam L nda::Array type of left hand side.
   * @tparam R nda::Array type of right hand side.
   * @param l nda::Array left hand side operand.
   * @param r nda::Array right hand side operand.
   * @return Either a lazy binary expression for the division operation ('A' * 'A') or the result
   * of the matrix-inverse matrix multiplication.
   */
  template <Array L, Array R>
  Array auto operator/(L &&l, R &&r) {
    // allowed algebras: A / A or M / M
    static constexpr char l_algebra = get_algebra<L>;
    static constexpr char r_algebra = get_algebra<R>;
    static_assert(l_algebra != 'V', "Error in nda::operator/: Can not divide a vector by an array or a matrix");

    // two arrays: A / A
    if constexpr (l_algebra == 'A') {
      static_assert(r_algebra == 'A', "Error in nda::operator/: Both types need to be arrays");
      static_assert(get_rank<L> == get_rank<R>, "Error in nda::operator/: Rank mismatch");
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.shape() != r.shape()) NDA_RUNTIME_ERROR << "Error in nda::operator/: Dimension mismatch: " << l.shape() << " != " << r.shape();
#endif
      return expr<'/', L, R>{std::forward<L>(l), std::forward<R>(r)};
    }

    // two matrices: M / M
    if constexpr (l_algebra == 'M') {
      static_assert(r_algebra == 'M', "Error in nda::operator/: Can not divide a matrix by an array/vector");
      return std::forward<L>(l) * linalg::inv(matrix<get_value_t<R>>{std::forward<R>(r)});
    }
  }

  /**
   * @brief Division operator for an nda::Array and an nda::Scalar.
   *
   * @details It performs lazy elementwise division.
   *
   * @tparam A nda::Array type.
   * @tparam S nda::Scalar type.
   * @param a nda::Array left hand side operand.
   * @param s nda::Scalar right hand side operand.
   * @return Lazy binary expression for the division operation.
   */
  template <Array A, Scalar S>
  Array auto operator/(A &&a, S &&s) { // NOLINT (S&& is mandatory for proper concept Array <: typename to work)
    return expr<'/', A, std::decay_t<S>>{std::forward<A>(a), s};
  }

  /**
   * @brief Division operator for an nda::Scalar and an nda::Array.
   *
   * @details Depending on the algebra of the nda::Array, it performs the following lazy operations:
   * - 'A': Elementwise division.
   * - 'M': Multiplication of the nda::Scalar with the inverse of the matrix.
   *
   * @tparam S nda::Scalar type.
   * @tparam A nda::Array type.
   * @param s nda::Scalar left hand side operand.
   * @param a nda::Array right hand side operand.
   * @return Lazy binary expression for the division operation (multiplication in case of a matrix).
   */
  template <Scalar S, Array A>
  Array auto operator/(S &&s, A &&a) { // NOLINT (S&& is mandatory for proper concept Array <: typename to work)
    static constexpr char algebra = get_algebra<A>;
    if constexpr (algebra == 'M')
      return s * linalg::inv(matrix<get_value_t<A>>{std::forward<A>(a)});
    else
      return expr<'/', std::decay_t<S>, A>{s, std::forward<A>(a)};
  }

  /** @} */

} // namespace nda
