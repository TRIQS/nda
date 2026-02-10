// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides the p-norm for general arrays/views of rank 1 and with scalar elements.
 */

#pragma once

#include "../algorithms.hpp"
#include "../basic_functions.hpp"
#include "../blas/dot.hpp"
#include "../concepts.hpp"
#include "../mapped_functions.hxx"
#include "../traits.hpp"

#include <cmath>
#include <complex>
#include <limits>

namespace nda::linalg {

  /**
   * @ingroup linalg_norms
   * @brief Calculate the \f$ p \f$-norm of a 1-dimensional array/view with scalar elements.
   *
   * @details The \f$ p \f$-norm is defined as
   * \f[
   *   || \mathbf{x} ||_p = \left( \sum_{i=0}^{N-1} |x_i|^p \right)^{1/p}
   * \f]
   * with the special cases (following `numpy.linalg.norm` convention)
   *
   * - \f$ || \mathbf{x} ||_0 = \text{number of non-zero elements} \f$,
   * - \f$ || \mathbf{x} ||_{\infty} = \max \{ |x_i| : i = 0, \dots, N - 1 \} \f$,
   * - \f$ || \mathbf{x} ||_{-\infty} = \min \{ |x_i| : i = 0, \dots, N - 1 \} \f$.
   * 
   * @note \f$ \mathbf{x} \f$ is required to have a value type that satisfies nda::Scalar.
   *
   * @tparam X nda::ArrayOfRank<1> type.
   * @param x 1-dimensional array \f$ \mathbf{x} \f$.
   * @param p Order of the norm.
   * @return \f$ p \f$-norm of the array/view.
   */
  template <ArrayOfRank<1> X>
    requires(Scalar<get_value_t<X>>)
  double norm(X const &x, double p = 2.0) {
    if (p == 2.0) [[likely]] {
      if constexpr (MemoryArray<X>)
        return std::sqrt(std::real(nda::blas::dotc(x, x)));
      else
        return norm(make_regular(x));
    } else if (p == 1.0) {
      return sum(abs(x));
    } else if (p == 0.0) {
      long count = 0;
      for (long i = 0; i < x.size(); ++i) {
        if (x(i) != get_value_t<X>{0}) ++count;
      }
      return double(count);
    } else if (p == std::numeric_limits<double>::infinity()) {
      return max_element(abs(x));
    } else if (p == -std::numeric_limits<double>::infinity()) {
      return min_element(abs(x));
    } else {
      return std::pow(sum(pow(abs(x), p)), 1.0 / p);
    }
  }

} // namespace nda::linalg
