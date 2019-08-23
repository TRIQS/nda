#pragma once
namespace nda {

  namespace traversal {

    inline struct C_t {
    } C;
    inline struct Fortran_t {
    } Fortran;
    inline struct Dynamical_t { } Dynamical; } // namespace traversal

  template <int Rank> class idx_map; //forward

  /**
    * Iterator on the idx_map.
    * Forward Iterator, with 3 possible traversal orders.
    * Output can be
    * It is given by a permutation, with the same convention as IndexOrder.
    */
  template <int Rank, bool WithIndices = false, typename TraversalOrder = traversal::C_t> class idx_map_iterator {
    idx_map<Rank> const *im = nullptr;
    using idx_t             = vec_or_array<long, Rank>;
    idx_t idx;
    long pos = 0;

    public:
    idx_map_iterator() = default;
    idx_map_iterator(idx_map<Rank> const *im_ptr) : im(im_ptr), pos(im->offset()) {
      for (int u = 0; u < im->rank(); ++u) idx[u] = 0;
    }

    using difference_type   = std::ptrdiff_t;
    using value_type        = std::conditional_t<WithIndices, std::pair<long, idx_t const &>, long>;
    using iterator_category = std::forward_iterator_tag;
    using pointer           = value_type *;
    using reference         = value_type &;

    // a little sentinel to test the end
    struct end_sentinel_t {};

    bool operator==(end_sentinel_t) const {
      if constexpr (Rank == 1) // faster implementation
        return idx[0] == im->lengths()[0];
      else
        return pos == -1;
    }
    bool operator==(idx_map_iterator const &other) const { return (other.pos == pos); }

    bool operator!=(end_sentinel_t other) const { return (!operator==(other)); }
    bool operator!=(idx_map_iterator const &other) const { return (!operator==(other)); }

    auto operator*() const {
      if constexpr (WithIndices) {
        return value_type{pos, idx};
      } else {
        return pos;
      }
    }
    auto operator-> () const { return operator*(); }

    idx_map_iterator &operator++() {

      if constexpr (Rank == 1) { // faster implementation
        ++(idx[0]);
        pos += im->strides()[0];
        return *this;
      } else {

        // traverse : fastest index first, then slowest..
        for (int v = im->rank() - 1; v >= 0; --v) {

          int p;
          if constexpr (std::is_same_v<TraversalOrder, traversal::C_t>) {
            p = v;
          } else if constexpr (std::is_same_v<TraversalOrder, traversal::Fortran_t>) {
            p = im->rank() - v - 1;
          } else if constexpr (std::is_same_v<TraversalOrder, traversal::Dynamical_t>) {
            p = im->layout()[v];
          }

          if (idx[p] < im->lengths()[p] - 1) {
            ++(idx[p]);
            pos += im->strides()[p];
            return *this;
          }
          idx[p] = 0;
          pos -= (im->lengths()[p] - 1) * im->strides()[p];
        }
        pos = -1;
        return *this;
      }
    }

    idx_map_iterator operator++(int) {
      idx_map_iterator c = *this;
      ++(*this);
      return c;
    }

    idx_t const &indices() const { return idx; }
  };

} // namespace nda
