// Copyright (c) 2018-2021 Simons Foundation
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
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include "permutation.hpp"

namespace nda {

  // ----------------  for_each  -------------------------

  /*  // C style*/
  //template <int I, typename F, size_t R>
  //FORCEINLINE void for_each_impl(std::array<long, R> idx_lengths, F &&f) {
  //if constexpr (I == R)
  //f();
  //else {
  //long imax = idx_lengths[I];
  //for (int i = 0; i < imax; ++i) {
  //for_each_impl<I + 1>(
  //idx_lengths, [ i, f ](auto &&... x) __attribute__((always_inline)) { return f(i, x...); });
  //}
  //}
  //}

  /////
  //template <typename F, size_t R>
  //FORCEINLINE void for_each2(std::array<long, R> idx_lengths, F &&f) {
  //for_each_impl<0>(idx_lengths, f);
  /*}*/

  namespace details {

    // return the i th index in Strider
    template <int R>
    constexpr int index_from_stride_order(uint64_t StrideOrder, int i) {
      if (StrideOrder == 0) return i;             // default C order
      auto stride_order = decode<R>(StrideOrder); // FIXME C++20
      return stride_order[i];
    }

    // ----------------  get_extent  -------------------------

    template <int I, int R, uint64_t StaticExtents, std::integral Int = long>
    long get_extent(std::array<Int, R> const &l) {
      if constexpr (StaticExtents == 0)
        return l[I]; // quick exit, no computation of
      else {
        constexpr auto static_extents = decode<R>(StaticExtents); // FIXME C++20
        if constexpr (static_extents[I] == 0)
          return l[I];
        else
          return static_extents[I];
      }
    }

    // ----------------  for_each

    template <int I, uint64_t StaticExtents, uint64_t StrideOrder, typename F, size_t R, std::integral Int = long>
    FORCEINLINE void for_each_static_impl(std::array<Int, R> const &idx_lengths, F &&f) {
      if constexpr (I == R)
        f();
      else {
        static constexpr int J = details::index_from_stride_order<R>(StrideOrder, I);
        const long imax        = details::get_extent<J, R, StaticExtents>(idx_lengths);
        for (long i = 0; i < imax; ++i) {
          for_each_static_impl<I + 1, StaticExtents, StrideOrder>(
             idx_lengths,
             [ i, &f ](auto &&... x)
// Great: clang and gcc want the lambda mutable and attribute in a different order !:
#ifdef __clang__
                __attribute__((always_inline)) mutable
#else
                mutable __attribute__((always_inline))
#endif
             { return f(i, x...); }); // mutable since f itself can be mutable !
        }
      }
    }
  } // namespace details

  // ----------------  for_each  -------------------------

  ///
  template <uint64_t StaticExtents, uint64_t StrideOrder, typename F, auto R, std::integral Int = long>
  FORCEINLINE void for_each_static(std::array<Int, R> const &idx_lengths, F &&f) {
    details::for_each_static_impl<0, StaticExtents, StrideOrder>(idx_lengths, f);
  }

  /// A loop in C order
  template <typename F, auto R, std::integral Int = long>
  FORCEINLINE void for_each(std::array<Int, R> const &idx_lengths, F &&f) {
    details::for_each_static_impl<0, 0, 0>(idx_lengths, f);
  }

} // namespace nda
