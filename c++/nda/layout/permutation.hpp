#pragma once
#include "../std_addons/array.hpp"

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
    auto result = nda::make_initialized_array<Rank>(0);
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

  template <typename T, auto R>
  constexpr std::array<T, R> apply_inverse(std::array<int, R> const &permutation, std::array<T, R> const &a) {
    auto result = nda::make_initialized_array<R, T>(0);
    for (int u = 0; u < R; ++u) { result[permutation[u]] = a[u]; }
    return result;
  }

  template <typename T, auto R>
  constexpr std::array<T, R> apply(std::array<int, R> const &permutation, std::array<T, R> const &a) {
    auto result = nda::make_initialized_array<R, T>(0);
    for (int u = 0; u < R; ++u) { result[u] = a[permutation[u]]; }
    return result;
  }

  template <int Rank>
  constexpr std::array<int, Rank> transposition(int i, int j) {
    auto r = identity<Rank>();
    r[i]   = j;
    r[j]   = i;
    return r;
  }

} // namespace nda::permutations
