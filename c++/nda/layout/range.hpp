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
#include <ostream>

#include <nda/exceptions.hpp>

namespace nda {

  struct range_tag {};

  /**
  `range` mimics the Python `range`.
  */
  class range : range_tag {
    long first_ = 0, last_ = -1, step_ = 1;

    public:
    /**
     * Default constructor.
     * Equivalent to all, takes all indices
     *
     * As in Python, range() stands for the whole set of indices in the dimension (like `:` in python) ::    
     * A(range(), 0) // take the first column of A
     * */
    range() = default;

    /**
     *
     * @param first : first index
     * @param last : last index + 1 (as in Python or STL).
     * @param step : step, default 1
     *
     * @examples :
     *
     *      A(range (0,3), 0)  // means  A(0,0), A(1,0), A(2,0)
     *      A(range (0,4,2), 0) // means A(0,0), A(2,0)  
     * */
    range(long first__, long last__, long step__ = 1) noexcept : first_(first__), last_(last__), step_(step__) {}

    /**
     * Constructor
     *
     * @param i : last index + 1 (as in Python or STL).
     *
     * Equivalent to range(0,i,1)
     * */
    explicit range(long i) noexcept : range(0, i, 1) {}

    /// First index of the range
    [[nodiscard]] long first() const noexcept { return first_; }

    /// Last index of the range + 1 (as in Python or STL).
    [[nodiscard]] long last() const noexcept { return last_; }

    /// Step between two indices
    [[nodiscard]] long step() const noexcept { return step_; }

    /// Number of indices in the range
    [[nodiscard]] long size() const noexcept { return (last_ - first_) / step_; }

    friend inline std::ostream &operator<<(std::ostream &os, const range &r) noexcept {
      os << "range(" << r.first() << "," << r.last() << "," << r.step() << ")";
      return os;
    }

    // Iterator on the range (for for loop e.g.)
    class const_iterator {
      long last, pos, step;

      public:
      using value_type        = long;
      using iterator_category = std::forward_iterator_tag;
      using pointer           = value_type *;
      using difference_type   = std::ptrdiff_t;
      using reference         = value_type const &;

      const_iterator(range const *r, bool atEnd) noexcept {
        last = r->last();
        step = r->step();
        pos  = (atEnd ? last : r->first());
      }

      const_iterator &operator++() noexcept {
        pos += step;
        return *this;
      }

      const_iterator operator++(int) noexcept {
        const_iterator c = *this;
        pos += step;
        return c;
      }

      bool operator==(const_iterator const &other) const noexcept { return (other.pos == this->pos); }
      bool operator!=(const_iterator const &other) const noexcept { return (!operator==(other)); }

      long operator*() const noexcept { return pos; }
      long operator->() const noexcept { return operator*(); }
    };

    //---------
    [[nodiscard]] const_iterator begin() const noexcept { return const_iterator(this, false); }
    [[nodiscard]] const_iterator end() const noexcept { return const_iterator(this, true); }
    [[nodiscard]] const_iterator cbegin() const noexcept { return const_iterator(this, false); }
    [[nodiscard]] const_iterator cend() const noexcept { return const_iterator(this, true); }
  };

  // ------------------   foreach   --------------------------------------

  /// Calls f for each element of the range
 // useless ?
  //template <typename F>
  //void foreach (range const &r, F && f) noexcept {
    //long i = r.first(), last = r.last(), step = r.step();
    //for (; i < last; i += step) std::forward<F>(f)(i);
  //}

  // ------------------  range_all   --------------------------------------

  /// Equivalent to range, but quicker (no operation).
  struct range_all : range_tag {};

  /// Ellipsis can be provided in place of [[range]], as in python. The type `ellipsis` is similar to [[range_all]] except that it is implicitly repeated to as much as necessary.
  struct ellipsis : range_all {};

  inline std::ostream &operator<<(std::ostream &os, range_all) noexcept { return os << "_"; }
  inline std::ostream &operator<<(std::ostream &os, ellipsis) noexcept { return os << "___"; }

  namespace vars {
    static inline constexpr range_all _;
    static inline constexpr ellipsis ___;
  } // namespace vars

  // ---------------------- details  --------------------------------

  // impl details : detects ellipsis in a argument pack
  template <typename... T>
  constexpr bool ellipsis_is_present = ((std::is_same_v<T, ellipsis> ? 1 : 0) + ... + 0); // +0 because it can be empty

} // namespace nda
