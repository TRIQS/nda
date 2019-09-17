/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet, P. Dumitrescu, N. Wentzell
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
#include <type_traits>
#include "../traits.hpp"

namespace nda {

  /**
   * A quick cache for object basic_array/view of rank 1 or 2
   * It makes a temporary copy if contiguity is not enough for lapack call.
   *
   * For vector and matrix : no copy
   * For matrix_view : maybe a copy
   */
  template <typename Aref>
  class _qcache {

    static_assert(std::is_reference_v<Aref>, "_qcache can only bind a reference");
    using A         = std::remove_reference_t<Aref>; // A may be const !
    using view_t    = typename A::view_t;
    using regular_t = typename A::regular_t;

    A &a;
    regular_t _copy; // empty by default
    view_t _view;
    bool need_copy = false, back_copy = false;

    static constexpr bool guarantee_no_copy = is_regular_v<A> or (A::rank == 1);
    static constexpr bool is_const          = std::is_const_v<A>;

    public:
    explicit _qcache(A &a, bool back_copy = false) : a(a), _view(a), back_copy(back_copy) {
      if constexpr (!guarantee_no_copy) {
        need_copy = (A::rank == 2 ? (a.indexmap().min_stride() != 1) : false);
        if (need_copy) {
          _copy = typename A::regular_t{a};
          _view.rebind(_copy);
        }
      }
    }

    ~_qcache() {
      if constexpr (!guarantee_no_copy and !is_const) {
        if (need_copy and back_copy) a = _copy;
      }
    }

    _qcache(_qcache const &) = delete;
    void operator=(_qcache const &) = delete;

    constexpr view_t operator()() const { return _view; }

    // for test only
    bool use_copy() const { return need_copy; }
  };

  template <typename A>
  _qcache<A &> qcache(A &a) {
    return _qcache<A &>{a, false};
  }

  template <typename A>
  _qcache<A &> reflexive_qcache(A &a) {
    return _qcache<A &>{a, true};
  }

} // namespace nda
