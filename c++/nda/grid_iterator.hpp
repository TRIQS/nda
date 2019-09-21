#pragma once
#include <iterator>
namespace nda {

  /**
    * Iterator on a rectangular grid (in C traversal order)
    */
  template <int Rank>
  class grid_iterator {
    long stri   = 0;
    long pos    = 0;
    long offset = 0;

    using sub_iterator_t = std::conditional_t<(Rank > 1) : grid_iterator<Rank - 1>, void *>;
    sub_iterator_t it, it_begin, it_end;

    public:
    grid_iterator() = default;
    grid_iterator(long const *lengths, long const *strides, bool at_end)
       : stri(strides[0]),
         pos(at_end ? lengths[0] : 0),
         offset(pos * stri),
         it(lengths + 1, strides + 1, false),
         it_begin(it),
         it_end(lengths + 1, strides + 1, true) {}

    using difference_type   = std::ptrdiff_t;
    using value_type        = long;
    using iterator_category = std::forward_iterator_tag;
    using pointer           = value_type *;
    using reference         = value_type &;

    //-----------------

    [[nodiscard]] value_type &operator*() const {
      if constexpr (Rank == 1) {
        return pos;
      } else {
        return pos + it.get_offset();
      }
    }

    value_type &operator->() const { return operator*(); }

    //-----------------

    bool operator==(grid_iterator const &other) const { return (other.pos == pos); }
    bool operator!=(grid_iterator const &other) const { return (other.pos != pos); }

    //-----------------

    [[nodiscard]] long get_offset() const { return pos; }

    //-----------------

    grid_iterator &operator++() {
      if constexpr (Rank == 1) {
        offset += stri;
        ++pos;
      } else {
        if (it != it_end) //FIXME [[likely]]
          ++it;
        else {
          ++pos;
          offset += stri;
          it = it_begin;
        }
      }
      return *this;
    }

    grid_iterator operator++(int) {
      auto c = *this;
      ++(*this);
      return c;
    }
  };

  // The iterator for the array and array_view container.
  // Makes an iterator of rank Rank on a pointer of type T.
  // e.g. for a strided_1d, we use Rank == 1, whatever the real array is
  // T can be const
  template <int Rank, typename T, bool Restrict>
  class array_iterator : public std::iterator<std::forward_iterator_tag, T> {

    T *data = nullptr;
    std::array<long, Rank> len, stri;
    grid_iterator<Rank> iter;

    public:
    using value_type = T;

    array_iterator()                       = default;
    array_iterator(array_iterator const &) = default;

    array_iterator(std::array<long, Rank> const &len, std::array<long, Rank> const &strides, T *start, bool at_end)
       : data(start), len(lengths), stri(strides), iter(len.data(), stri.data(), at_end) {}

    value_type &operator*() const {
      if constexpr (Restrict)
        return ((T * __restrict) data)[*iter];
      else
        return data[*iter];
    }

    value_type &operator->() const { return operator*(); }

    array_iterator &operator++() {
      ++it;
      return *this;
    }

    array_iterator operator++(int) {
      auto c = *this;
      ++it;
      return c;
    }

    bool operator==(array_iterator const &other) const { return (other.it == it); }
    bool operator!=(array_iterator const &other) const { return (!operator==(other)); }
  };

} // namespace nda
