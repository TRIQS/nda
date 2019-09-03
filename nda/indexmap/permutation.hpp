#pragma once
#include "../std_array_addons.hpp"

namespace nda::permutations {

  // permutation of [1, ..., N] are stored as a bit pattern.
  // since N <= 16, we use 4 bits per N, for a maximum of 16* 4 = 64 bits
  // the remaining bits are 0.

  // the two following constexpr functions encode and decode it from the binary representation
  // to std::array<int, Rank>
  // indeed, we can not yet (C++20 ?) template array, array_view on a std::array ...

  template <size_t Rank>
  constexpr std::array<int, Rank> decode(uint64_t binary_representation) {
    auto result = nda::make_initialized_array<Rank>(0);
    for (int i = 0; i < int(Rank); ++i) result[i] = (binary_representation >> (4 * i)) & 0b1111ull;
    return result;
  }

  template <size_t Rank>
  constexpr uint64_t encode(std::array<int, Rank> const &permutation) {
    uint64_t result = 0;
    for (int i = 0; i < int(Rank); ++i) result += (permutation[i] >> (4 * i)) & 0b1111ull;
    return result;
  }

  template <int Rank>
  constexpr std::array<int, Rank> identity() {
    auto result = nda::make_initialized_array<Rank>(0);
    for (int i = 0; i < Rank; ++i) result[i] = i;
    return result;
  }
  template <int Rank>
  constexpr std::array<int, Rank> reverse_identity() {
    auto result = nda::make_initialized_array<Rank>(0);
    for (int i = 0; i < Rank; ++i) result[i] = Rank - 1 - i;
    return result;
  }

} // namespace nda::permutations
