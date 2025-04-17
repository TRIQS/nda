// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to sum an expression over a single or multiple domains.
 */

#pragma once

#include "./clef.hpp"
#include "../basic_functions.hpp"
#include "../exceptions.hpp"

namespace nda::clef {

  /**
   * @addtogroup clef_expr
   * @{
   */

  namespace detail {

    // Helper function to sum a callable object over a domain using a simple for loop.
    template <typename F, typename D>
    auto sum_f_domain_impl(F const &f, D const &d)
      requires(not is_clef_expression<F, D>)
    {
      auto it  = d.begin();
      auto ite = d.end();
      if (it == ite) NDA_RUNTIME_ERROR << "Error in nda::clef::sum_f_domain_impl: Sum over an empty domain";
      auto res = make_regular(f(*it));
      ++it;
      for (; it != ite; ++it) res = res + f(*it);
      return res;
    }

    // Make sum_f_domain_impl lazy.
    CLEF_MAKE_FNT_LAZY(sum_f_domain_impl);

  } // namespace detail

  /**
   * @brief Sum an expression over a 1-dimensional domain.
   *
   * @details The following example sums the squared elements of a vector:
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * auto domain = std::vector{1, 2, 3};
   * auto ex = i_ * i_;
   * auto res = nda::clef::sum(ex, i_ = domain); // int res = 14;
   * @endcode
   *
   * @tparam Expr Type of the expression.
   * @tparam N Integer label of the placeholder to be replaced by the domain.
   * @tparam D Type of the domain.
   * @param ex Lazy expression.
   * @param d Pair containing the nda::clef::placeholder and the domain.
   * @return Either the result of the summation or a new lazy expression.
   */
  template <typename Expr, int N, typename D>
  decltype(auto) sum(Expr const &ex, clef::pair<N, D> d) {
    if constexpr (std::is_lvalue_reference_v<D>) {
      return detail::sum_f_domain_impl(make_function(ex, clef::placeholder<N>()), d.rhs);
    } else {
      return detail::sum_f_domain_impl(make_function(ex, clef::placeholder<N>()), std::move(d.rhs));
    }
  }

  /**
   * @brief Sum an expression over a multi-dimensional domain.
   *
   * @details The following example sums an expression over a 2-dimensional domain:
   *
   * @code{.cpp}
   * nda::clef::placeholder<0> i_;
   * nda::clef::placeholder<0> j_;
   * auto domain1 = std::vector{1, 2, 3};
   * auto domain2 = std::vector{4, 5, 6};
   * auto ex = i_ + j_;
   * auto res = nda::clef::sum(ex, i_ = domain1, j_ = domain2); // int res = 63;
   * @endcode
   *
   * @tparam Expr Type of the expression.
   * @tparam D0 nda::clef::pair type.
   * @tparam D1 nda::clef::pair type.
   * @tparam Ds Parameter pack of the remaining nda::clef::pair types.
   * @param ex Lazy expression.
   * @param d0 Pair containing an nda::clef::placeholder and a domain.
   * @param d1 Pair containing an nda::clef::placeholder and another domain.
   * @param ds Parameter pack of the remaining pairs.
   * @return Either the result of the summation or a new lazy expression.
   */
  template <typename Expr, typename D0, typename D1, typename... Ds>
  auto sum(Expr const &ex, D0 &&d0, D1 &&d1, Ds &&...ds) {
    return sum(sum(ex, std::forward<D0>(d0)), std::forward<D1>(d1), std::forward<Ds>(ds)...);
  }

  /** @} */

} // namespace nda::clef
