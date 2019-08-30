/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *  author : O. Parcollet
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
namespace nda {

    namespace layout {

    inline struct C_t {
    } C;
    inline struct Fortran_t { } Fortran; } // namespace layout

  // backward compat.
  //#define C_LAYOUT nda::layout::C
  //#define FORTRAN_LAYOUT nda::layout::Fortran

  /* The storage order is given by a permutation P
  *   P[0] : the slowest index,
  *   P[Rank-1] : the fastest index
  *   Example :
  *   210 : Fortran, the first index is the fastest
  *   012 : C the last index is the fastest
  *   120 : storage (i,j,k) is : index j is slowest, then k, then i
  */

  template <int Rank> class layout_t : _rank_injector<Rank> {
    vec_or_array<uint8_t, Rank> p;

    public:
    layout_t() = default;

    /// R is ignored in static case
    // Add a tag or ambiguous with the last constructor
    /*layout_t(int R) {
      if constexpr (Rank == -1) {
        this->_rank = R;
        p.resize(R);
      }
    }
*/

    layout_t(layout::C_t, int R = 0) : layout_t(R) {
      for (int u = 0; u < this->rank(); ++u) p[u] = u;
    }

    layout_t(layout::Fortran_t, int R = 0) : layout_t(R) {
      for (int u = 0; u < this->rank(); ++u) p[u] = this->rank() - u - 1;
    }

    template <typename... INT> layout_t(uint8_t i0, INT... in) : p{i0, uint8_t(in)...} {}

    bool operator==(layout_t const &ml) const { return p == ml.p; }
    bool operator!=(layout_t const &ml) const { return !operator==(ml); }

    uint8_t operator[](int u) const { return p[u]; }
    uint8_t &operator[](int u) { return p[u]; }

    vec_or_array<uint8_t, Rank> const &as_permutation() const { return p; }

  }; // namespace nda::layout

  // ------------------------- functions -------------------------------------

  template <int R> bool is_C(layout_t<R> const &l) {
    for (int u = 0; u < l.rank(); ++u) {
      if (l[u] != u) return false;
    }
    return true;
  }

  template <int R> bool is_Fortran(layout_t<R> const &l) {
    for (int u = 0; u < l.rank(); ++u) {
      if (l[u] != l.rank() - u - 1) return false;
    }
    return true;
  }

  template <int R> vec_or_array<int, R> memory_positions(layout_t<R> const &l) {
    vec_or_array<int, R> r;
    for (int u = 0; u < l.rank(); ++u) r[l[u]] = u;
    return r;
  }

  /// Apply a permutation to the indices
  template <int R> layout_t<R> transpose(layout_t<R> const &l, vec_or_array<int, R> const &perm) {
    auto r = l;
    for (int u = 0; u < l.rank(); ++u) r[u] = perm[l[u]];
    return r;
  }

  /* /// Make a memory_layout_t from the strides
  template <int Rank> std::array<int, Rank> layout_from_strides(std::array<long, Rank> const &strides) {
    std::array<int, Rank> c;
    for (int i = 0; i < Rank; ++i) c[i] = 0;
    for (int i = 0; i < Rank; ++i)
      for (int j = i + 1; j < Rank; ++j)
        if (strides[i] < strides[j])
          c[i]++;
        else
          c[j]++;
    return c;
  }
*/

} // namespace nda
