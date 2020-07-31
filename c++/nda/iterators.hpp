// Copyright (c) 2019-2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <iterator>
namespace nda {

  template <int Rank>
  class grid_iterator;

  /*
   * Iterator on a rectangular grid (in C traversal order)
   *
   */

  // -------------------------------
  // Rank = 1 is a special case
  template <>
  class grid_iterator<1> : public std::iterator<std::forward_iterator_tag, long> {
    long stri   = 0;
    long pos    = 0;
    long offset = 0;

    public:
    grid_iterator() = default;
    grid_iterator(long const *lengths, long const *strides, bool at_end) : stri(strides[0]), pos(at_end ? lengths[0] : 0), offset(pos * stri) {}

    [[nodiscard]] long operator*() const { return offset; }
    long operator->() const { return operator*(); }

    bool operator==(grid_iterator const &other) const { return (other.pos == pos); }
#if __cplusplus > 201703L
    bool operator!=(grid_iterator const &other) const { return (other.pos != pos); }
#endif

    grid_iterator &operator++() {
      offset += stri;
      ++pos;
      return *this;
    }

    grid_iterator operator++(int) {
      auto c = *this;
      ++(*this);
      return c;
    }
  };

  // -------------------------------
  // Rank >1 : general case
  template <int Rank>
  class grid_iterator : public std::iterator<std::forward_iterator_tag, long> {
    long stri   = 0;
    long pos    = 0;
    long offset = 0;
    grid_iterator<Rank - 1> it_begin, it_end, it;

    public:
    grid_iterator() = default;

    grid_iterator(long const *lengths, long const *strides, bool at_end)
       : stri(strides[0]),
         pos(at_end ? lengths[0] : 0),
         offset(pos * stri),
         it_begin(lengths + 1, strides + 1, false),
         it_end(lengths + 1, strides + 1, true),
         it(it_begin) {} // NB always it_begin, even if at_end. The end iterator correspond to pos = (length -1) + 1, but it is at its *beginning*

    [[nodiscard]] long operator*() const { return offset + *it; }
    long operator->() const { return operator*(); }

    bool operator==(grid_iterator const &other) const { return ((other.pos == pos) and (other.it == it)); }
    bool operator!=(grid_iterator const &other) const { return not operator==(other); }

    grid_iterator &operator++() {
      ++it;
      if (it == it_end) { //FIXME [[unlikely]]
        ++pos;
        offset += stri;
        it = it_begin;
      }
      return *this;
    }

    grid_iterator operator++(int) {
      auto c = *this;
      ++(*this);
      return c;
    }
  };

  //-----------------------------------------------------------------------

  // The iterator for the array and array_view container.
  // Makes an iterator of rank Rank on a pointer of type T.
  // e.g. for a strided_1d, we use Rank == 1, whatever the real array is
  // T can be const
  template <int Rank, typename T, typename Pointer>
  class array_iterator : public std::iterator<std::forward_iterator_tag, T> {
    T *data = nullptr;
    std::array<long, Rank> len, stri;
    grid_iterator<Rank> iter;

    public:
    using value_type = T;

    array_iterator()                       = default;
    array_iterator(array_iterator const &) = default;

    array_iterator(std::array<long, Rank> const &lengths, std::array<long, Rank> const &strides, T *start, bool at_end)
       : data(start), len(lengths), stri(strides), iter(len.data(), stri.data(), at_end) {}

    value_type &operator*() const { return ((Pointer)data)[*iter]; }

    value_type &operator->() const { return operator*(); }

    array_iterator &operator++() {
      ++iter;
      return *this;
    }

    array_iterator operator++(int) {
      auto c = *this;
      ++iter;
      return c;
    }

    bool operator==(array_iterator const &other) const { return (other.iter == iter); }
    bool operator!=(array_iterator const &other) const { return (!operator==(other)); }
  };

} // namespace nda
