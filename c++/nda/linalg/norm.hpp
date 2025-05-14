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
   * @ingroup linalg_tools
   * @brief Calculate the p-norm of an nda::ArrayOfRank<1> object \f$ \mathbf{x} \f$ with scalar values. The p-norm is
   * defined as
   * \f[
   *   || \mathbf{x} ||_p = \left( \sum_{i=0}^{N-1} |x_i|^p \right)^{1/p}
   * \f]
   * with the special cases (following `numpy.linalg.norm` convention)
   *
   * - \f$ || \mathbf{x} ||_0 = \text{number of non-zero elements} \f$,
   * - \f$ || \mathbf{x} ||_{\infty} = \max \{ |x_i| : i = 0, \dots, N - 1 \} \f$,
   * - \f$ || \mathbf{x} ||_{-\infty} = \min \{ |x_i| : i = 0, \dots, N - 1 \} \f$.
   *
   * @tparam A nda::ArrayOfRank<1> type.
   * @param a nda::ArrayOfRank<1> object \f$ \mathbf{x} \f$.
   * @param p Order of the norm.
   * @return p-norm of the array/view.
   */
  template <ArrayOfRank<1> A>
    requires(Scalar<get_value_t<A>>)
  double norm(A const &a, double p = 2.0) {
    if (p == 2.0) [[likely]] {
      if constexpr (MemoryArray<A>)
        return std::sqrt(std::real(nda::blas::dotc(a, a)));
      else
        return norm(make_regular(a));
    } else if (p == 1.0) {
      return sum(abs(a));
    } else if (p == 0.0) {
      long count = 0;
      for (long i = 0; i < a.size(); ++i) {
        if (a(i) != get_value_t<A>{0}) ++count;
      }
      return double(count);
    } else if (p == std::numeric_limits<double>::infinity()) {
      return max_element(abs(a));
    } else if (p == -std::numeric_limits<double>::infinity()) {
      return min_element(abs(a));
    } else {
      return std::pow(sum(pow(abs(a), p)), 1.0 / p);
    }
  }

} // namespace nda::linalg
