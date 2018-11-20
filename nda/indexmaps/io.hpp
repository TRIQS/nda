/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 * author : O. Parcollet
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
#include <ostream>

namespace std {

  template <typename T, size_t N> std::ostream &operator<<(std::ostream &out, std::array<T, N> const &v) {
    out << "(";
    for (int i = 0; i < N; ++i) out << (i == 0 ? "" : " ") << int(v[i]);
    out << ")";
    return out;
  }

  /* template<typename T, size_t N> 
  std::string to_string(std::array<T,N> const &v) const {
    std::ostringstream fs;
    fs<<v;
    return fs.str();
  }
 */
} // namespace std

namespace nda {

   // layout_t
  template <int Rank> std::ostream &operator<<(std::ostream &out, layout_t<Rank> const &s) {
    out << "(";
    for (int i = 0; i < s.rank(); ++i) out << (i == 0 ? "" : " ") << int(s.as_permutation()[i]);
    out << ")";
    return out;
  }

  // idx_map
  template <int Rank> std::ostream &operator<<(std::ostream &out, idx_map<Rank> const &x) {
    return out << "  Lengths  : " << x.lengths() << "\n"
               << "  Strides  : " << x.strides() << "\n"
               << "  Offset   : " << x.offset() << "\n"
               << "  Layout   : " << x.layout().as_permutation() << "\n";
  }

} // namespace nda
