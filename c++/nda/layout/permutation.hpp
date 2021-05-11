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
#include "../stdutil/array.hpp"
#include "../stdutil/concepts.hpp"
#include "../macros.hpp"

namespace nda {

// FIXME in C++20 , will be std::array
// For template argument with a list of integer
#define ARRAY_INT uint64_t

  // Compact representation of std::array<int,N> for N< 16
  // indeed, we can not yet (C++20 ?) template array, array_view on a std::array ...
  // when the elements are also <16/
  // As a bit pattern.
  // since N <= 16, we use 4 bits per N, for a maximum of 16* 4 = 64 bits
  // the remaining bits are 0.
  // the two following constexpr functions encode and decode it from the binary representation
  // to std::array<int, Rank>
  template <size_t Rank>
  constexpr std::array<int, Rank> decode(uint64_t binary_representation) {
    auto result = stdutil::make_initialized_array<Rank>(0);
    for (int i = 0; i < int(Rank); ++i) result[i] = (binary_representation >> (4 * i)) & 0b1111ull;
    return result;
  }

  template <size_t Rank>
  constexpr uint64_t encode(std::array<int, Rank> const &a) {
    uint64_t result = 0;
    for (int i = 0; i < int(Rank); ++i) result += (a[i] << (4 * i));
    return result;
  }

} // namespace nda

namespace nda::permutations {

  template <CONCEPT(std::integral) Int, size_t Rank>
  REQUIRES17(std::is_integral_v<Int>)
  constexpr bool is_valid(std::array<Int, Rank> const &permutation) {
    auto idx_counts = stdutil::make_initialized_array<Rank>(0);
    for (auto idx : permutation) {
      if (idx_counts[idx] > 0) return false;
      idx_counts[idx] = 1;
    }
    return true;
  }

  template <CONCEPT(std::integral) Int, size_t Rank>
  REQUIRES17(std::is_integral_v<Int>)
  constexpr std::array<Int, Rank> inverse(std::array<Int, Rank> const &permutation) {
    EXPECTS(is_valid(permutation));
    auto result = stdutil::make_initialized_array<Rank>(0);
    for (int u = 0; u < Rank; ++u) { result[permutation[u]] = u; }
    return result;
  }

  template <typename T, CONCEPT(std::integral) Int, size_t Rank>
  REQUIRES17(std::is_integral_v<Int>)
  constexpr std::array<T, Rank> apply_inverse(std::array<Int, Rank> const &permutation, std::array<T, Rank> const &a) {
    EXPECTS(is_valid(permutation));
    auto result = stdutil::make_initialized_array<Rank, T>(0);
    for (int u = 0; u < Rank; ++u) { result[permutation[u]] = a[u]; }
    return result;
  }

  template <typename T, CONCEPT(std::integral) Int, size_t Rank>
  REQUIRES17(std::is_integral_v<Int>)
  constexpr std::array<T, Rank> apply(std::array<Int, Rank> const &permutation, std::array<T, Rank> const &a) {
    EXPECTS(is_valid(permutation));
    auto result = stdutil::make_initialized_array<Rank, T>(0);
    for (int u = 0; u < Rank; ++u) { result[u] = a[permutation[u]]; }
    return result;
  }

  template <size_t Rank>
  constexpr std::array<int, Rank> identity() {
    auto result = stdutil::make_initialized_array<Rank>(0);
    for (int i = 0; i < Rank; ++i) result[i] = i;
    return result;
  }

  template <size_t Rank>
  constexpr std::array<int, Rank> reverse_identity() {
    auto result = stdutil::make_initialized_array<Rank>(0);
    for (int i = 0; i < Rank; ++i) result[i] = Rank - 1 - i;
    return result;
  }

  template <size_t Rank>
  constexpr std::array<int, Rank> transposition(int i, int j) {
    auto r = identity<Rank>();
    r[i]   = j;
    r[j]   = i;
    return r;
  }

  // cyclic permutation, p times. Forward
  // n = 1 :  0 1 2 3 ---->   3 0 1 2
  // P[n] = (Rank + n - p) % Rank
  // 4 + 0 -1 = 3, 4 + 1 -1 = 0, 4 +2 -1 = 1, etc...
  // if pos < Rank, the cycle is partial
  // cycle<5> (1, 3) --> 2 0 1 3 4
  // pos ==0 --> identity
  template <size_t Rank>
  constexpr std::array<int, Rank> cycle(int p, int pos = Rank) {
    auto result = stdutil::make_initialized_array<Rank>(0);
    for (int i = 0; i < Rank; ++i) result[i] = (i < pos ? (pos + i - p) % pos : i);
    return result;
  }

} // namespace nda::permutations
