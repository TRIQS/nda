// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides utilities to work with permutations and to compactly encode/decode
 * std::array objects.
 */

#pragma once

#include "../macros.hpp"
#include "../stdutil/array.hpp"
#include "../stdutil/concepts.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace nda {

  /**
   * @addtogroup utils_perms
   * @{
   */

  /**
   * @brief Decode a `uint64_t` into a `std::array<int, N>`.
   *
   * @details The 64-bit code is split into 4-bit chunks, and each chunk is then decoded into a value in the range
   * [0, 15]. The 4 least significant bits are decoded into the first element, the next 4-bits into second element, and
   * so on.
   *
   * @tparam N Size of the array.
   * @param binary_representation 64-bit code.
   * @return `std::array<int, N>` containing values in the range [0, 15].
   */
  template <size_t N>
  constexpr std::array<int, N> decode(uint64_t binary_representation) {
    auto result = stdutil::make_initialized_array<N>(0);
    for (int i = 0; i < N; ++i) result[i] = (binary_representation >> (4 * i)) & 0b1111ull;
    return result;
  }

  /**
   * @brief Encode a `std::array<int, N>` in a `uint64_t`.
   *
   * @details The values in the array are assumed to be in the range [0, 15]. Then each value is encoded in 4 bits, i.e.
   * the first element is encoded in the 4 least significant bits, the second element in the next 4 bits, and so on.
   *
   * @tparam N Size of the array.
   * @param a `std::array<int, N>` to encode.
   * @return 64-bit code.
   */
  template <size_t N>
  constexpr uint64_t encode(std::array<int, N> const &a) {
    uint64_t result = 0;
    for (int i = 0; i < N; ++i) {
      EXPECTS(0 <= a[i] and a[i] <= 15);
      result += (static_cast<uint64_t>(a[i]) << (4 * i));
    }
    return result;
  }

  /** @} */

} // namespace nda

namespace nda::permutations {

  /**
   * @addtogroup utils_perms
   * @{
   */

  /**
   * @brief Check if a given array is a valid permutation.
   *
   * @details A valid permutation is an array of integers with values in the range [0, N - 1] where each value appears
   * exactly once.
   *
   * @tparam Int Integral type.
   * @tparam N Degree of the permutation.
   * @param p std:::array to check.
   * @return True if the array is a valid permutation, false otherwise.
   */
  template <std::integral Int, size_t N>
  constexpr bool is_valid(std::array<Int, N> const &p) {
    auto idx_counts = stdutil::make_initialized_array<N>(0);
    for (auto idx : p) {
      if (idx < 0 or idx > N - 1 or idx_counts[idx] > 0) return false;
      idx_counts[idx] = 1;
    }
    return true;
  }

  /**
   * @brief Composition of two permutations.
   *
   * @details The second argument is applied first. Composition is not commutative in general, i.e. `compose(p1, p2) !=
   * compose(p2, p1)`.
   *
   * @tparam Int Integral type.
   * @tparam N Degree of the permutations.
   * @param p1 Permutation to apply second.
   * @param p2 Permutation to apply first.
   * @return Composition of the two permutations.
   */
  template <std::integral Int, size_t N>
  constexpr std::array<Int, N> compose(std::array<Int, N> const &p1, std::array<Int, N> const &p2) {
    EXPECTS(is_valid(p1));
    EXPECTS(is_valid(p2));
    auto result = stdutil::make_initialized_array<N>(0);
    for (int u = 0; u < N; ++u) result[u] = p1[p2[u]];
    return result;
  }

  /**
   * @brief Inverse of a permutation.
   *
   * @details The inverse, `inv`, of a permutation `p` is defined such that `compose(p, inv) == compose(inv, p) ==
   * identity`.
   *
   * @tparam Int Integral type.
   * @tparam N Degree of the permutations.
   * @param p Permutation to invert.
   * @return Inverse of the permutation.
   */
  template <std::integral Int, size_t N>
  constexpr std::array<Int, N> inverse(std::array<Int, N> const &p) {
    EXPECTS(is_valid(p));
    auto result = stdutil::make_initialized_array<N>(0);
    for (int u = 0; u < N; ++u) result[p[u]] = u;
    return result;
  }

  /**
   * @brief Apply the inverse of a permutation to a std::array.
   *
   * @details See also nda::permutations::apply and nda::permutations::inverse.
   *
   * @tparam T Value type of the array.
   * @tparam Int Integral type.
   * @tparam N Size/Degree of the array/permutation.
   * @param p Permutation to invert and apply.
   * @param a std::array to apply the inverse permutation to.
   * @return Result of applying the inverse permutation to the array.
   */
  template <typename T, std::integral Int, size_t N>
  constexpr std::array<T, N> apply_inverse(std::array<Int, N> const &p, std::array<T, N> const &a) {
    EXPECTS(is_valid(p));
    auto result = stdutil::make_initialized_array<N, T>(0);
    for (int u = 0; u < N; ++u) result[p[u]] = a[u];
    return result;
  }

  /**
   * @brief Apply a permutation to a std::array.
   *
   * @details The application of a permutation `p` to an array `a` results in a new array `b` such that `b[i] =
   * a[p[i]]`.
   *
   * @tparam T Value type of the array.
   * @tparam Int Integral type.
   * @tparam N Size/Degree of the array/permutation.
   * @param p Permutation to apply.
   * @param a std::array to apply the permutation to.
   * @return Result of applying the permutation to the array.
   */
  template <typename T, std::integral Int, size_t N>
  constexpr std::array<T, N> apply(std::array<Int, N> const &p, std::array<T, N> const &a) {
    EXPECTS(is_valid(p));
    auto result = stdutil::make_initialized_array<N, T>(0);
    for (int u = 0; u < N; ++u) result[u] = a[p[u]];
    return result;
  }

  /**
   * @brief Get the identity permutation.
   *
   * @tparam N Degree of the permutation.
   * @return Identity permutation of degree `N`.
   */
  template <size_t N>
  constexpr std::array<int, N> identity() {
    auto result = stdutil::make_initialized_array<N>(0);
    for (int i = 0; i < N; ++i) result[i] = i;
    return result;
  }

  /**
   * @brief Get the reverse identity permutation.
   *
   * @tparam N Degree of the permutation.
   * @return Reverse of the identity permutation of degree `N`.
   */
  template <size_t N>
  constexpr std::array<int, N> reverse_identity() {
    auto result = stdutil::make_initialized_array<N>(0);
    for (int i = 0; i < N; ++i) result[i] = N - 1 - i;
    return result;
  }

  /**
   * @brief Get the permutation representing a single given transposition.
   *
   * @details A transposition is a permutation in which only two elements are different from the identity permutation.
   *
   * @tparam N Degree of the permutation.
   * @param i First index to transpose.
   * @param j Second index to transpose.
   * @return Permutation corresponding to the transposition of `i` and `j`.
   */
  template <size_t N>
  constexpr std::array<int, N> transposition(int i, int j) {
    auto r = identity<N>();
    r[i]   = j;
    r[j]   = i;
    return r;
  }

  /**
   * @brief Perform a forward (partial) cyclic permutation of the identity `p` times.
   *
   * @details The permutation is partial if `pos >= 0 && pos < N`. In this case, only the first `pos` elements are
   * permuted, while the rest is left unchanged w.r.t the identity.
   *
   * @tparam N Degree of the permutation.
   * @param p Number of times to perform the cyclic permutation.
   * @param pos Number of elements to permute. If `pos < N`, the permutation is partial.
   * @return Result of the cyclic permutation.
   */
  template <size_t N>
  constexpr std::array<int, N> cycle(int p, int pos = N) {
    auto result = stdutil::make_initialized_array<N>(0);
    pos         = std::clamp(pos, 0, static_cast<int>(N));
    for (int i = 0; i < N; ++i) result[i] = (i < pos ? (pos + i - p) % pos : i);
    return result;
  }

  /** @} */

} // namespace nda::permutations
