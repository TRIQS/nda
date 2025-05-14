// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a cross product for 3-dimensional vectors.
 */

#pragma once

#include "../basic_array.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../mem/policies.hpp"

namespace nda::linalg {

  /**
   * @ingroup linalg_tools
   * @brief Compute the cross product \f$ \mathbf{x} \times \mathbf{y} \f$ of two 3-dimensional vectors \f$ \mathbf{x} 
   * \f$ and \f$ \mathbf{y} \f$.
   *
   * @tparam X nda::Vector type.
   * @tparam Y nda::Vector type.
   * @param x Input vector \f$ \mathbf{x} \f$.
   * @param y Input vector \f$ \mathbf{y} \f$.
   * @return nda::vector containing the cross product of the two vectors.
   */
  template <Vector X, Vector Y>
    requires(nda::mem::have_host_compatible_addr_space<X, Y>)
  auto cross_product(X const &x, Y const &y) {
    EXPECTS(x.size() == 3 and y.size() == 3);
    auto res = vector<decltype(x(0) * y(0)), nda::heap<nda::mem::common_addr_space<X, Y>>>(3);
    res(0)   = x(1) * y(2) - y(1) * x(2);
    res(1)   = -x(0) * y(2) + y(0) * x(2);
    res(2)   = x(0) * y(1) - y(0) * x(1);
    return res;
  }

} // namespace nda::linalg
