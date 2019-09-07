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
#pragma once

#include <iostream>

#define H5_AS_STRING(...) H5_AS_STRING2(__VA_ARGS__)
#define H5_AS_STRING2(...) #__VA_ARGS__

#ifdef __clang__
#define H5_REQUIRES(...) __attribute__((enable_if(__VA_ARGS__, H5_AS_STRING2(__VA_ARGS__))))
#elif __GNUC__
#define H5_REQUIRES(...) requires(__VA_ARGS__)
#endif

#define H5_PRINT(X) std::cerr << H5_AS_STRING(X) << " = " << X << "      at " << __FILE__ << ":" << __LINE__ << '\n'

#define H5_FORCEINLINE __inline__ __attribute__((always_inline))

#define H5_EXPECTS(X)                                                                                                                                \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Precondition " << H5_AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                          \
    std::terminate();                                                                                                                                \
  }
#define H5_ASSERT(X)                                                                                                                                 \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Assertion " << H5_AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                             \
    std::terminate();                                                                                                                                \
  }
#define H5_ENSURES(X)                                                                                                                                \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Postcondition " << H5_AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                         \
    std::terminate();                                                                                                                                \
  }
