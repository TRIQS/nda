
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
#include <iterator>

namespace nda {

  // remap the idx_map iterator into an iterator
  // ValueType can be const
  template <typename ValueType, typename IdxMap>
  class iterator_adapter : public std::iterator<std::forward_iterator_tag, ValueType> {

    ValueType *_data = 0;
    typename IdxMap::iterator it;

    public:
    using value_type = ValueType;
    //using indexmap_iterator_type = typename IdxMap::iterator ;

    iterator_adapter()                         = default;
    iterator_adapter(iterator_adapter const &) = default;
    iterator_adapter(typename IdxMap::iterator const &it, ValueType *start) : _data(start), it(it) {}

    value_type &operator*() const { return _data[*it]; }
    value_type &operator->() const { return _data[*it]; }

    iterator_adapter &operator++() {
      ++it;
      return *this;
    }

    iterator_adapter operator++(int) {
      auto c = *this;
      ++it;
      return c;
    }

    // a little sentinel to test the end
    struct end_sentinel_t {};

    bool operator==(end_sentinel_t) const { return (it == typename IdxMap::iterator::end_sentinel_t{}); }
    bool operator!=(end_sentinel_t) const { return (it != typename IdxMap::iterator::end_sentinel_t{}); }

    bool operator==(iterator_adapter const &other) const { return (other.it == it); }
    bool operator!=(iterator_adapter const &other) const { return (!operator==(other)); }

    // not in forward iterator concept
    [[deprecated]] operator bool() const { return (*this) == end_sentinel_t{}; }

    // FIXME after bench
    decltype(auto) indices() const { return it.indices(); }

    typename IdxMap::iterator const &indexmap_iterator() const { return it; }
  };
} // namespace nda
