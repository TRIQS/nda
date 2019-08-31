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

namespace nda {

  /**
   * A range of indices.
   * Mimics the Python `range`.
  * */
  class range {
    long first_ = 0, last_ = -1, step_ = 1;

    public:
    using index_type = long;

    /**
     * Default constructor.
     * Equivalent to all, takes all indices
     *
     * As in Python, range() stands for the whole set of indices in the dimension (like `:` in python) ::    
     * A(range(), 0) // take the first column of A
     * */
    range() = default;

    /**
     * Constructor
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
    range(long first, long last, long step = 1) : first_(first), last_(last), step_(step) {}

    /**
     * Constructor
     *
     * @param i : last index + 1 (as in Python or STL).
     *
     * Equivalent to range(0,i,1)
     * */
    explicit range(long i) : range(0, i, 1) {}

    /// First index of the range
    long first() const { return first_; }

    /// Last index of the range + 1 (as in Python or STL).
    long last() const { return last_; }

    /// Step between two indices
    long step() const { return step_; }

    /// Number of indices in the range
    size_t size() const {
      long r = (last_ - first_) / step_;
      if (r < 0) TRIQS_RUNTIME_ERROR << " range with negative size";
      return size_t(r);
    }

    /// FIXME : deprecate ?
    //range operator+(long shift) const { return range(first_ + shift, last_ + shift, step_); }

    // Iterator on the range (for for loop e.g.)
    class const_iterator {
      long last, pos, step;

      public:
      using value_type        = long;
      using iterator_category = std::forward_iterator_tag;
      using pointer           = value_type *;
      using difference_type   = std::ptrdiff_t;
      using reference         = value_type const &;

      const_iterator(range const *r, bool atEnd) {
        last = r->last();
        step = r->step();
        pos  = (atEnd ? last : r->first());
      }

      const_iterator &operator++() {
        pos += step;
        return *this;
      }

      const_iterator operator++(int) {
        Iter c = self();
        pos += step;
        return *this;
      }

      bool operator==(Iter const &other) const { return (other.pos == this->pos); }
      bool operator!=(Iter const &other) const { return (!operator==(other)); }

      long operator*() const { return pos; }
      long operator->() const { return operator*(); }
    };

    //---------

    const_iterator begin() const { return const_iterator(this, false); }
    const_iterator end() const { return const_iterator(this, true); }
    const_iterator cbegin() const { return const_iterator(this, false); }
    const_iterator cend() const { return const_iterator(this, true); }
  };

  /// Apply the lambda F to
  template <typename F>
  void foreach (range const &r, F const &f) {
    long i = r.first(), last = r.last(), step = r.step();
    for (; i < last; i += step) f(i);
  }

  /**
   * Ellipsis is equivalent to any number >= 0 of r
   *
   * */
  class ellipsis : public range {
    public:
    // deprecated
    //ellipsis(long first, long last, long step = 1) : range(first, last, step) {}
    ellipsis() : range() {}
  };

  // FIXME : why is this here ?
  // for the case A(i, ellipsis) where A is of dim 1...
  //inline int operator*(ellipsis, int) { return 0; }
  //inline int operator*(int, ellipsis) { return 0; }
} // namespace nda
