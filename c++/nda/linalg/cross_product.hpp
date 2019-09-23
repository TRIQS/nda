/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#ifndef TRIQS_LINALG_CROSS_PRODUCT_H
#define TRIQS_LINALG_CROSS_PRODUCT_H
#include <triqs/utility/exceptions.hpp>

namespace triqs {
  namespace arrays {

    /** Cross product. Dim 3 only */
    template <typename VectorType>
    vector<typename std::remove_const<typename VectorType::value_type>::type> cross_product(VectorType const &A, VectorType const &B) {
      if (A.shape()[0] != 3)
        TRIQS_RUNTIME_ERROR << "arrays::linalg::cross_product : works only in d=3 while you gave a vector of size " << A.shape()[0];
      if (B.shape()[0] != 3)
        TRIQS_RUNTIME_ERROR << "arrays::linalg::cross_product : works only in d=3 while you gave a vector of size " << B.shape()[0];
      vector<typename std::remove_const<typename VectorType::value_type>::type> r(3);
      r(0) = A(1) * B(2) - B(1) * A(2);
      r(1) = -A(0) * B(2) + B(0) * A(2);
      r(2) = A(0) * B(1) - B(0) * A(1);
      return r;
    }

  } // namespace arrays
} // namespace triqs

#endif
