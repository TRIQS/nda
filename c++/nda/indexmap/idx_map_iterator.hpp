#pragma once
namespace nda {

  /**
    * Iterator on the idx_map.
    * Forward Iterator, with 3 possible traversal orders.
    * Output can be
    * It is given by a permutation, with the same convention as IndexOrder.
    */
  template <typename IdxMap, bool WithIndices = false>
  class idx_map_iterator {
    IdxMap const *im          = nullptr;
    static constexpr int Rank = IdxMap::rank();
    using idx_t               = std::array<long, Rank>;
    idx_t idx;
    long pos = 0;
    std::array<long, Rank> len, strides;

    public:
    idx_map_iterator() = default;
    idx_map_iterator(IdxMap const *im_ptr) : im(im_ptr), pos(im->offset()), len(im->lengths()), strides(im->strides()) {
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
        return idx[0] == len[0];
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
        pos += strides[0];
        return *this;
      } else {

        // traverse : fastest index first, then slowest..
        for (int v = im->rank() - 1; v >= 0; --v) {

          int p = v;
          if (idx[p] < len[p] - 1) {
            ++(idx[p]);
            pos += strides[p];
            return *this;
          }
          idx[p] = 0;
          pos -= (len[p] - 1) * strides[p];
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
