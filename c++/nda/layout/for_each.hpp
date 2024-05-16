// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides for_each functions for multi-dimensional arrays.
 */

#pragma once

#include "./permutation.hpp"

#include <array>
#include <concepts>
#include <cstdint>

namespace nda {

  namespace details {

    // Get the i-th slowest moving dimension from a given encoded stride order.
    template <int R>
    constexpr int index_from_stride_order(uint64_t stride_order, int i) {
      if (stride_order == 0) return i;                 // default C-order
      auto stride_order_arr = decode<R>(stride_order); // FIXME C++20
      return stride_order_arr[i];
    }

    // Get the extent of an array along its i-th dimension.
    template <int I, int R, uint64_t StaticExtents, std::integral Int = long>
    constexpr long get_extent(std::array<Int, R> const &shape) {
      if constexpr (StaticExtents == 0) {
        // dynamic extents
        return shape[I];
      } else {
        // static extents
        constexpr auto static_extents = decode<R>(StaticExtents); // FIXME C++20
        if constexpr (static_extents[I] == 0)
          return shape[I];
        else
          return static_extents[I];
      }
    }

    // Apply a callable object recursively to all possible index values of a given shape.
    template <int I, uint64_t StaticExtents, uint64_t StrideOrder, typename F, size_t R, std::integral Int = long>
    FORCEINLINE void for_each_static_impl(std::array<Int, R> const &shape, std::array<long, R> &idxs, F &f) {
      if constexpr (I == R) {
        // end of recursion
        std::apply(f, idxs);
      } else {
        // get the dimension over which to iterate and its extent
        static constexpr int J = details::index_from_stride_order<R>(StrideOrder, I);
        const long imax        = details::get_extent<J, R, StaticExtents>(shape);

        // loop over all indices of the current dimension
        for (long i = 0; i < imax; ++i) {
          // recursive call for the next dimension
          for_each_static_impl<I + 1, StaticExtents, StrideOrder>(shape, idxs, f);
          ++idxs[J];
        }
        idxs[J] = 0;
      }
    }

  } // namespace details

  /**
   * @brief Loop over all possible index values of a given shape and apply a function to them.
   *
   * @details It traverses all possible indices in the order given by the encoded `StrideOrder` parameter.
   * The shape is either specified at runtime (`StaticExtents == 0`) or, partially or fully, at compile time (`StaticExtents != 0`).
   * The given function `f` must be callable with as many long values as the number of dimensions in the shape
   * array, e.g. for a 3-dimensional shape the function must be callable as `f(long, long, long)`.
   *
   * @tparam StaticExtents Encoded static extents.
   * @tparam StrideOrder Encoded stride order.
   * @tparam F Callable type.
   * @tparam R Rank of the array.
   * @tparam Int Integer type.
   *
   * @param shape Shape to loop over (index bounds).
   * @param f Callable object.
   */
  template <uint64_t StaticExtents, uint64_t StrideOrder, typename F, auto R, std::integral Int = long>
  FORCEINLINE void for_each_static(std::array<Int, R> const &shape, F &&f) { // NOLINT (we do not want to forward here)
    auto idxs = nda::stdutil::make_initialized_array<R>(0l);
    details::for_each_static_impl<0, StaticExtents, StrideOrder>(shape, idxs, f);
  }

  /**
   * @brief Loop over all possible index values of a given shape and apply a function to them.
   *
   * @details It traverses all possible indices in C-order, i.e. the last index varies the fastest.
   * The shape is specified at runtime and the given function `f` must be callable with
   * as many long values as the number of dimensions in the shape array, e.g. for a 3-dimensional
   * shape the function must be callable as `f(long, long, long)`.
   *
   * @tparam F Callable type.
   * @tparam R Number of dimensions.
   * @tparam Int Integer type used in the shape array.
   *
   * @param shape Shape to loop over (index bounds).
   * @param f Callable object.
   */
  template <typename F, auto R, std::integral Int = long>
  FORCEINLINE void for_each(std::array<Int, R> const &shape, F &&f) { // NOLINT (we do not want to forward here)
    auto idxs = nda::stdutil::make_initialized_array<R>(0l);
    details::for_each_static_impl<0, 0, 0>(shape, idxs, f);
  }

} // namespace nda
