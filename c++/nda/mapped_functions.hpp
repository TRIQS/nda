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
#include "./map.hpp"
#include "./traits.hpp"

#include <cmath>
#include <complex>
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

    // Get the squared absolute value of a double.
    inline double abs2(double x) { return x * x; }

    // Get the squared absolute value of a std::complex<double>.
    inline double abs2(std::complex<double> z) { return (conj(z) * z).real(); }

    // Check if a std::complex<double> is NaN.
    inline bool isnan(std::complex<double> const &z) { return std::isnan(z.real()) or std::isnan(z.imag()); }

    // Functor for nda::detail::conj.
    struct conj_f {
      auto operator()(auto const &x) const { return conj(x); };
    };

  } // namespace detail

  /**
   * @brief Function pow for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types).
   *
   * @tparam A nda::ArrayOrScalar type..
   * @param a nda::ArrayOrScalar object.
   * @param p Exponent value.
   * @return A lazy nda::expr_call object (nda::Array) or the result of `std::pow` applied to the object (nda::Scalar).
   */
  template <ArrayOrScalar A>
  auto pow(A &&a, double p) {
    return nda::map([p](auto const &x) {
      using std::pow;
      return pow(x, p);
    })(std::forward<A>(a));
  }

  /**
   * @brief Function conj for nda::ArrayOrScalar types (lazy and coefficient-wise for nda::Array types with a complex
   * value type).
   *
   * @tparam A nda::ArrayOrScalar type..
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

  /** @} */

} // namespace nda
