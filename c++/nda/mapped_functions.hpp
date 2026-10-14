// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides some custom implementations of standard mathematical functions used for lazy, coefficient-wise array
 * operations.
 */

#pragma once

#include "./concepts.hpp"
#include "./macros.hpp"
#include "./map.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @addtogroup av_math
   * @{
   */

  namespace detail {

    // Get the real part of a scalar.
    template <nda::Scalar S>
    auto real(S x) {
      if constexpr (is_complex_v<S>) {
        return std::real(x);
      } else {
        return x;
      }
    }

    // Get the complex conjugate of a scalar.
    template <nda::Scalar S>
    auto conj(S x) {
      if constexpr (is_complex_v<S>) {
        return std::conj(x);
      } else {
        return x;
      }
    }

    // Get the squared absolute value with the return type of std::norm.
    FORCEINLINE auto abs2(auto x) { return std::norm(x); }

    // Check if a scalar or either component of a complex scalar is NaN.
    template <Scalar S>
    bool isnan(S const &x) {
      if constexpr (is_complex_v<S>) {
        return std::isnan(x.real()) or std::isnan(x.imag());
      } else {
        return std::isnan(x);
      }
    }

    // Functor for nda::detail::conj.
    struct conj_f {
      auto operator()(auto const &x) const { return conj(x); };
    };

  } // namespace detail

#ifdef NDA_DOXYGEN
  /**
   * @brief Function pow for nda::Scalar types (uses `std::pow`).
   */
  template <Scalar S, Scalar P>
  auto pow(S s, P p);
#endif
  using std::pow;

  /**
   * @brief Function pow for nda::Array types (lazy and coefficient-wise).
   *
   * @tparam A nda::Array type.
   * @tparam S nda::Scalar exponent type.
   * @param a nda::Array object.
   * @param p Exponent value.
   * @return A lazy nda::expr_call object.
   */
  template <Array A, Scalar S>
  auto pow(A &&a, S p) {
    return nda::map([p](auto const &x) {
      using std::pow;
      return pow(x, p);
    })(std::forward<A>(a));
  }

  /**
   * @brief Function conj for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types with a complex
   * value type).
   *
   * @tparam A nda::ArrayOrScalar type.
   * @param a nda::ArrayOrScalar object.
   * @return A lazy nda::expr_call object (nda::Array and complex valued), the forwarded input object (nda::Array and
   * not complex valued) or the complex conjugate of the scalar input.
   */
  template <ArrayOrScalar A>
  decltype(auto) conj(A &&a) {
    if constexpr (is_complex_v<get_value_t<A>>)
      return nda::map(detail::conj_f{})(std::forward<A>(a));
    else
      return std::forward<A>(a);
  }

  /**
   * @brief Reciprocal function for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   * 
   * @tparam A nda::ArrayOrScalar type.
   * @param a nda::ArrayOrScalar object.
   * @return A lazy nda::expr_call object (nda::Array) or the result of \f$ 1.0 / x \f$ applied to the object
   * (nda::Scalar).
   */
  template <ArrayOrScalar A>
  auto reciprocal(A &&a) {
    return nda::map([]<typename T>(T const &x) {
      if constexpr (Scalar<T>) {
        using real_t = remove_complex_t<T>;
        if constexpr (std::is_floating_point_v<real_t>) {
          return real_t{1} / x;
        } else {
          return 1.0 / x;
        }
      } else {
        return nda::reciprocal(x);
      }
    })(std::forward<A>(a));
  }

  /**
   * @brief Function max for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   *
   * @tparam A nda::ArrayOrScalar type.
   * @tparam B nda::ArrayOrScalar type.
   * @param a First operand.
   * @param b Second operand.
   * @return A lazy nda::expr_call object (nda::Array) or the result of `std::max` applied to the inputs (nda::Scalar).
   */
  template <ArrayOrScalar A, ArrayOrScalar B>
    requires(((Scalar<A> && Scalar<B>) || (Array<A> && Array<B> && get_rank<A> == get_rank<B>))
             && !is_complex_v<get_value_t<A>> && !is_complex_v<get_value_t<B>>)
  [[nodiscard]] auto max(A &&a, B &&b) {
    return nda::map([](auto const &x, auto const &y) {
      using std::max;
      return max(x, y);
    })(std::forward<A>(a), std::forward<B>(b));
  }

  /**
   * @brief Function min for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   *
   * @tparam A nda::ArrayOrScalar type.
   * @tparam B nda::ArrayOrScalar type.
   * @param a First operand.
   * @param b Second operand.
   * @return A lazy nda::expr_call object (nda::Array) or the result of `std::min` applied to the inputs (nda::Scalar).
   */
  template <ArrayOrScalar A, ArrayOrScalar B>
    requires(((Scalar<A> && Scalar<B>) || (Array<A> && Array<B> && get_rank<A> == get_rank<B>))
             && !is_complex_v<get_value_t<A>> && !is_complex_v<get_value_t<B>>)
  [[nodiscard]] auto min(A &&a, B &&b) {
    return nda::map([](auto const &x, auto const &y) {
      using std::min;
      return min(x, y);
    })(std::forward<A>(a), std::forward<B>(b));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function abs for nda::Scalar types (uses `std::abs`).
  template <Scalar S>
  auto abs(S s);
#endif
  using std::abs;

  /// \brief Function abs for nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto abs(A &&a)  {
    return nda::map(
       [](auto const &x) {
         using std::abs;
         return abs(x);
       })(std::forward<A>(a));
  }

  /// \brief Function imag for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type.
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `std::imag` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto imag(A &&a)  {
    return nda::map(
       [](auto const &x) {
         using std::imag;
         return imag(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function floor for nda::Scalar types (uses `std::floor`).
  template <Scalar S>
  auto floor(S s);
#endif
  using std::floor;

  /// \brief Function floor for nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto floor(A &&a)  {
    return nda::map(
       [](auto const &x) {
         using std::floor;
         return floor(x);
       })(std::forward<A>(a));
  }

  /// \brief Function real for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type..
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::real` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto real(A &&a) {
    return nda::map(
       [](auto const &x) {return detail::real(x); })(std::forward<A>(a));
  }

  /// \brief Function abs2 for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
  ///
  /// \tparam A nda::ArrayOrScalar type..
  /// \param a nda::ArrayOrScalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::abs2` applied to the object (nda::Scalar).
  template <ArrayOrScalar A>
  auto abs2(A &&a) {
    return nda::map(
       [](auto const &x) {return detail::abs2(x); })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function isnan for real nda::Scalar types (uses `std::isnan`).
  template <Scalar S>
  auto isnan(S s);
#endif
  using std::isnan;

  /// \brief Function isnan for nda::Array types and complex nda::Scalar types (lazy and coefficient-wise for nda::Array
  /// types).
  ///
  /// \details A complex scalar is NaN if either component is NaN.
  ///
  /// \tparam A nda::Array or complex nda::Scalar type.
  /// \param a nda::Array or complex nda::Scalar object.
  /// \return A lazy nda::expr_call object (nda::Array) or the result of `nda::detail::isnan` applied to the object
  /// (complex nda::Scalar).
  template <typename A>
    requires(Array<A> || (Scalar<std::remove_cvref_t<A>> && is_complex_v<std::remove_cvref_t<A>>))
  auto isnan(A &&a) {
    return nda::map(
       [](auto const &x) {return detail::isnan(x); })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function exp for nda::Scalar types (uses `std::exp`).
  template <Scalar S>
  auto exp(S s);
#endif
  using std::exp;

  /// \brief Function exp for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto exp(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::exp;
         return exp(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function cos for nda::Scalar types (uses `std::cos`).
  template <Scalar S>
  auto cos(S s);
#endif
  using std::cos;

  /// \brief Function cos for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto cos(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::cos;
         return cos(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function sin for nda::Scalar types (uses `std::sin`).
  template <Scalar S>
  auto sin(S s);
#endif
  using std::sin;

  /// \brief Function sin for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto sin(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::sin;
         return sin(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function tan for nda::Scalar types (uses `std::tan`).
  template <Scalar S>
  auto tan(S s);
#endif
  using std::tan;

  /// \brief Function tan for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto tan(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::tan;
         return tan(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function cosh for nda::Scalar types (uses `std::cosh`).
  template <Scalar S>
  auto cosh(S s);
#endif
  using std::cosh;

  /// \brief Function cosh for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto cosh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::cosh;
         return cosh(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function sinh for nda::Scalar types (uses `std::sinh`).
  template <Scalar S>
  auto sinh(S s);
#endif
  using std::sinh;

  /// \brief Function sinh for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto sinh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::sinh;
         return sinh(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function tanh for nda::Scalar types (uses `std::tanh`).
  template <Scalar S>
  auto tanh(S s);
#endif
  using std::tanh;

  /// \brief Function tanh for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto tanh(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::tanh;
         return tanh(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function acos for nda::Scalar types (uses `std::acos`).
  template <Scalar S>
  auto acos(S s);
#endif
  using std::acos;

  /// \brief Function acos for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto acos(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::acos;
         return acos(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function asin for nda::Scalar types (uses `std::asin`).
  template <Scalar S>
  auto asin(S s);
#endif
  using std::asin;

  /// \brief Function asin for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto asin(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::asin;
         return asin(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function atan for nda::Scalar types (uses `std::atan`).
  template <Scalar S>
  auto atan(S s);
#endif
  using std::atan;

  /// \brief Function atan for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto atan(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::atan;
         return atan(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function log for nda::Scalar types (uses `std::log`).
  template <Scalar S>
  auto log(S s);
#endif
  using std::log;

  /// \brief Function log for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto log(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::log;
         return log(x);
       })(std::forward<A>(a));
  }

#ifdef NDA_DOXYGEN
  /// \brief Function sqrt for nda::Scalar types (uses `std::sqrt`).
  template <Scalar S>
  auto sqrt(S s);
#endif
  using std::sqrt;

  /// \brief Function sqrt for non-matrix nda::Array types (lazy and coefficient-wise).
  ///
  /// \tparam A nda::Array type.
  /// \param a nda::Array object.
  /// \return A lazy nda::expr_call object.
  template <Array A>
  auto sqrt(A &&a) requires(get_algebra<A> != 'M') {
    return nda::map(
       [](auto const &x) {
         using std::sqrt;
         return sqrt(x);
       })(std::forward<A>(a));
  }

  /** @} */

} // namespace nda
