// Copyright (c) 2018-2020 Simons Foundation
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

#ifndef STD_ADDONS_ARRAY_H
#define STD_ADDONS_ARRAY_H

// No pragma once. Easier to copy to another project.

#include <array>
#include <utility>
#include <vector>

/// =========    ADDING IN STD ===========
/// =========    NEED TO PUT IT IN STD FOR ADL ========
// How to document this ?? Manually ?
namespace std {

  template <typename T, size_t R>
  std::ostream &operator<<(std::ostream &out, std::array<T, R> const &a) {
    return out << to_string(a);
  }

  template <typename T, size_t R>
  std::string to_string(std::array<T, R> const &a) {
    std::stringstream fs;
    fs << "(";
    for (int i = 0; i < R; ++i) fs << (i == 0 ? "" : " ") << a[i];
    fs << ")";
    return fs.str();
  }

 // ------------- basic arithmetic --------------------------------------

  template <typename T, size_t R>
  constexpr std::array<T, R> operator+(std::array<T, R> const &a1, std::array<T, R> const &a2) {
    std::array<T, R> res;// = make_initialized_array<R>(T{});
    for (int i = 0; i < R; ++i) res[i] = a1[i] + a2[i];
    return res;
  }

  template <typename T, size_t R>
  constexpr std::array<T, R> operator-(std::array<T, R> const &a1, std::array<T, R> const &a2) {
    std::array<T, R> res;// FIXME MOVE THIS = make_initialized_array<R>(T{});
    for (int i = 0; i < R; ++i) res[i] = a1[i] - a2[i];
    return res;
  }


} // namespace std

#endif

#ifndef STDUTILS_ARRAY_H
#define STDUTILS_ARRAY_H

/// =========    END ADDING IN STD ===========

namespace nda::stdutil {

  namespace impl {
    template <typename T, size_t... Is>
    constexpr std::array<T, sizeof...(Is)> make_initialized_array(T v, std::index_sequence<Is...>) {
      return {(Is ? v : v)...};
    } // always v, just a trick to have the pack
  }   // namespace impl

  /**
   * @tparam R
   * @tparam T
   * make a std::array<T, R> initialized to v
   */
  template <size_t R, typename T>
  constexpr std::array<T, R> make_initialized_array(T v) {
    return impl::make_initialized_array(v, std::make_index_sequence<R>{});
  }
  
  /**
   * @tparam T  T must be constructible from U
   * @tparam U
   * @tparam R
   * make a std::array<T, R> initialized to v
   */
  template <typename T, typename U, size_t R>
  constexpr std::array<T, R> make_std_array(std::array<U, R> const &a) {
    static_assert(std::is_constructible_v<T, U>, "make_std_array : T must be constructible from U, Cf doc");
    std::array<T, R> result = make_initialized_array<R>(T{}); 
    for (int u = 0; u < R; ++u) result[u] = a[u];
    return result;
  }


  /**
  * Convert a std::array to a
  * @tparam T
  * @param a std::array to convert
  */
  template <typename T, size_t R>
  constexpr std::vector<T> to_vector(std::array<T, R> const &a) {
    std::vector<T> V(R);
    for (int i = 0; i < R; ++i) V[i] = a[i];
    return V;
  }

  /**
   * Make a new std::array by appending one element at the end
   * @tparam T
   * @tparam U Must be convertible to T
   * @param a The array
   * @param x Element to append
   * @return A new std::array with the element appended at the end
   */
  template <typename T, auto R, typename U>
  constexpr std::array<T, R + 1> append(std::array<T, R> const &a, U const &x) {
    std::array<T, R + 1> res = make_initialized_array<R + 1>(T{}); // FIXME : c++20 defect
    for (int i = 0; i < R; ++i) res[i] = a[i];
    res[R] = x;
    return res;
  }

  /**
   * Make a new std::array by appending one element at the front
   * @tparam T
   * @tparam U Must be convertible to T
   * @param a The ar/ay
   * @param x Element to append
   * @return A new std::array with the element appended at the front
   */
  template <typename T, typename U, size_t R>
  constexpr std::array<T, R + 1> front_append(std::array<T, R> const &a, U const &x) {
    std::array<T, R + 1> res = make_initialized_array<R + 1>(T{});
    res[0]                   = x;
    for (int i = 0; i < R; ++i) res[i + 1] = a[i];
    return res;
  }

  /**
   * Make a new std::array by removing one element at the end
   * @tparam T
   * @param a The array
   * @return A new std::array with the element less at the end
   */
  template <typename T, size_t R>
  constexpr std::array<T, R - 1> pop(std::array<T, R> const &a) {
    std::array<T, R - 1> res = make_initialized_array<R - 1>(T{});
    for (int i = 0; i < R - 1; ++i) res[i] = a[i];
    return res;
  }

  /**
   * Make a new std::array by removing one element at the front
   * @tparam T
   * @param a The array
   * @return A new std::array with the element less at the front
   */
  template <int N, typename T, size_t R>
  constexpr std::array<T, R - N> mpop(std::array<T, R> const &a) {
    std::array<T, R - N> res = make_initialized_array<R - N>(T{});
    ;
    for (int i = 0; i < R - N; ++i) res[i] = a[i];
    return res;
  }

  /**
   * Make a new std::array by removing one element at the front
   * @tparam T
   * @param a The array
   * @return A new std::array with the element less at the front
   */
  template <typename T, size_t R>
  constexpr std::array<T, R - 1> front_pop(std::array<T, R> const &a) {
    std::array<T, R - 1> res = make_initialized_array<R - 1>(T{});
    for (int i = 1; i < R; ++i) res[i - 1] = a[i];
    return res;
  }

  /**
   * Make a new std::array by removing one element at the front
   * @tparam T
   * @param a The array
   * @return A new std::array with the element less at the front
   */
  template <int N, typename T, size_t R>
  constexpr std::array<T, R - N> front_mpop(std::array<T, R> const &a) {
    std::array<T, R - N> res = make_initialized_array<R - N>(T{});
    for (int i = N; i < R; ++i) res[i - N] = a[i];
    return res;
  }

  /**
   * Join two arrays
   * @tparam T
   * @param a1
   * @param a2
   * @return the concatenation of [a1, a2]
   */
  template <typename T, size_t R1, size_t R2>
  constexpr std::array<T, R1 + R2> join(std::array<T, R1> const &a1, std::array<T, R2> const &a2) {
    std::array<T, R1 + R2> res = make_initialized_array<R1 + R2>(T{});
    for (int i = 0; i < R1; ++i) res[i] = a1[i];
    for (int i = 0; i < R2; ++i) res[R1 + i] = a2[i];
    return res;
  }

  /**
   * Dot product of two arrays.
   * @tparam T
   * @tparam U
   * @param a1
   * @param a2
   * @return The dot product to whatever type T*U is promoted to. If R = 0, return T{}
   */
  // ------------- dot --------------------------------------
  template <typename T, typename U, size_t R>
  constexpr auto dot_product(std::array<T, R> const &a1, std::array<U, R> const &a2) {
    if constexpr (R == 0)
      return T{};
    else {
      auto res = a1[0] * a2[0];
      for (int i = 1; i < R; ++i) res += a1[i] * a2[i];
      return res;
    }
  }

} // namespace nda::stdutil
#endif
