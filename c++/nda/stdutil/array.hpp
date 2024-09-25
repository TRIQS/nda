// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2023 Simons Foundation
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

/**
 * @file
 * @brief Provides utility functions for std::array.
 */

#ifndef STDUTILS_ARRAY_H
#define STDUTILS_ARRAY_H

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

namespace std { // has to be in the right namespace for ADL

  /**
   * @addtogroup utils_std
   * @{
   */

  /**
   * @brief Write a std::array to an output stream.
   *
   * @tparam T Value type of the array.
   * @tparam R Size of the array.
   * @param out std::ostream to stream into.
   * @param a std::array to be written.
   * @return Reference to the std::ostream.
   */
  template <typename T, size_t R>
  std::ostream &operator<<(std::ostream &out, std::array<T, R> const &a) {
    return out << to_string(a);
  }

  /**
   * @brief Get a string representation of a std::array.
   *
   * @tparam T Value type of the array.
   * @tparam R Size of the array.
   * @param a Input std::array.
   * @return std::string representation of the array.
   */
  template <typename T, size_t R>
  std::string to_string(std::array<T, R> const &a) {
    std::stringstream fs;
    fs << "(";
    for (int i = 0; i < R; ++i) fs << (i == 0 ? "" : " ") << a[i];
    fs << ")";
    return fs.str();
  }

  /**
   * @brief Add two std::array objects element-wise.
   *
   * @tparam T Value type of the arrays.
   * @tparam R Size of the arrays.
   * @param lhs Left hand side std::array operand.
   * @param rhs Right hand side std::array operand.
   * @return std::array containing the element-wise sum.
   */
  template <typename T, size_t R>
  constexpr std::array<T, R> operator+(std::array<T, R> const &lhs, std::array<T, R> const &rhs) {
    std::array<T, R> res;
    for (int i = 0; i < R; ++i) res[i] = lhs[i] + rhs[i];
    return res;
  }

  /**
   * @brief Subtract two std::array objects element-wise.
   *
   * @tparam T Value type of the arrays.
   * @tparam R Size of the arrays.
   * @param lhs Left hand side std::array operand.
   * @param rhs Right hand side std::array operand.
   * @return std::array containing the element-wise difference.
   */
  template <typename T, size_t R>
  constexpr std::array<T, R> operator-(std::array<T, R> const &lhs, std::array<T, R> const &rhs) {
    std::array<T, R> res;
    for (int i = 0; i < R; ++i) res[i] = lhs[i] - rhs[i];
    return res;
  }

  /**
   * @brief Multiply two std::array objects element-wise.
   *
   * @tparam T Value type of the first array.
   * @tparam U Value type of the second array.
   * @tparam R Size of the arrays.
   * @param lhs Left hand side std::array operand.
   * @param rhs Right hand side std::array operand.
   * @return std::array containing the element-wise product.
   */
  template <typename T, typename U, size_t R>
  constexpr auto operator*(std::array<T, R> const &lhs, std::array<U, R> const &rhs) {
    using TU = decltype(std::declval<T>() * std::declval<U>());

    std::array<TU, R> res;
    for (int i = 0; i < R; ++i) res[i] = lhs[i] * rhs[i];
    return res;
  }

  /**
   * @brief Negate a std::array element-wise (unary minus).
   *
   * @tparam T Value type of the array.
   * @tparam R Size of the array.
   * @param a std::array operand.
   * @return std::array containing the element-wise negation.
   */
  template <typename T, size_t R>
  constexpr std::array<T, R> operator-(std::array<T, R> const &a) {
    std::array<T, R> res;
    for (int i = 0; i < R; ++i) res[i] = -a[i];
    return res;
  }

  /**
   * @brief Multiply a scalar and a std::array element-wise.
   *
   * @tparam T Value type of the array/scalar.
   * @tparam R Size of the array.
   * @param s Scalar value.
   * @param a std::array operand.
   * @return std::array containing the product of each element with the scalar.
   */
  template <typename T, size_t R>
  constexpr std::array<T, R> operator*(T s, std::array<T, R> const &a) {
    std::array<T, R> res;
    for (int i = 0; i < R; ++i) res[i] = s * a[i];
    return res;
  }

} // namespace std

namespace nda::stdutil {

  /**
   * @brief Create a new std::array object initialized with a specific value.
   *
   * @tparam R Size of the array.
   * @tparam T Value type of the array.
   * @param v Value to initialize the std::array with.
   * @return std::array initialized with the constant value.
   */
  template <size_t R, typename T>
  constexpr std::array<T, R> make_initialized_array(T v) {
    return [&v]<size_t... Is>(std::index_sequence<Is...>) {
      return std::array<T, R>{((void)Is, v)...}; // NOLINT (always v, just a trick to have the pack)
    }(std::make_index_sequence<R>{});
  }

  /**
   * @brief Convert a std::array with value type `U` to a std::array with value type `T`.
   *
   * @tparam T Value type of the target array.
   * @tparam U Value type of the input array.
   * @tparam R Size of the input array.
   * @param a Input std::array.
   * @return std::array with the same values as the input array and value type `T`.
   */
  template <typename T, typename U, size_t R>
  constexpr std::array<T, R> make_std_array(std::array<U, R> const &a) {
    static_assert(std::is_constructible_v<T, U>, "Error in nda::stdutil::make_std_array: T must be constructible from U");
    std::array<T, R> result = make_initialized_array<R>(T{});
    for (int u = 0; u < R; ++u) result[u] = a[u];
    return result;
  }

  /**
   * @brief Convert a std::array to a std::vector.
   *
   * @tparam T Value type of the input array.
   * @tparam R Size of the input array.
   * @param a Input std::array.
   * @return std::vector of the same size and with the same values as the input array.
   */
  template <typename T, size_t R>
  constexpr std::vector<T> to_vector(std::array<T, R> const &a) {
    std::vector<T> V(R);
    for (int i = 0; i < R; ++i) V[i] = a[i];
    return V;
  }

  /**
   * @brief Make a new std::array by appending one element at the end to an existing std::array.
   *
   * @tparam T Value type of the input array.
   * @tparam R Size of the input array.
   * @tparam U Type of the element to append (must be convertible to `T`).
   * @param a Input std::array.
   * @param x Element to append.
   * @return A copy of the input array with the additional element appended at the end.
   */
  template <typename T, auto R, typename U>
  constexpr std::array<T, R + 1> append(std::array<T, R> const &a, U const &x) {
    std::array<T, R + 1> res;
    for (int i = 0; i < R; ++i) res[i] = a[i];
    res[R] = x;
    return res;
  }

  /**
   * @brief Make a new std::array by prepending one element at the front to an existing std::array.
   *
   * @tparam T Value type of the input array.
   * @tparam U Type of the element to prepend (must be convertible to `T`).
   * @tparam R Size of the input array.
   * @param a Input std::array.
   * @param x Element to prepend.
   * @return A copy of the input array with the additional element prepended at the front.
   */
  template <typename T, typename U, size_t R>
  constexpr std::array<T, R + 1> front_append(std::array<T, R> const &a, U const &x) {
    std::array<T, R + 1> res;
    res[0] = x;
    for (int i = 0; i < R; ++i) res[i + 1] = a[i];
    return res;
  }

  /**
   * @brief Make a new std::array by popping the last `N` elements of an existing std::array.
   *
   * @tparam N Number of elements to pop.
   * @tparam T Value type of the input array.
   * @tparam R Size of the input array.
   * @param a Input std::array.
   * @return A copy of the input array with the last `N` elements removed.
   */
  template <int N, typename T, size_t R>
  constexpr std::array<T, R - N> mpop(std::array<T, R> const &a) {
    std::array<T, R - N> res;
    for (int i = 0; i < R - N; ++i) res[i] = a[i];
    return res;
  }

  /**
   * @brief Make a new std::array by popping the last element of an existing std::array.
   *
   * @tparam T Value type of the input array.
   * @tparam R Size of the input array.
   * @param a Input std::array.
   * @return A copy of the input array with the last element removed.
   */
  template <typename T, size_t R>
  constexpr std::array<T, R - 1> pop(std::array<T, R> const &a) {
    return mpop<1>(a);
  }

  /**
   * @brief Make a new std::array by popping the first `N` elements of an existing std::array.
   *
   * @tparam N Number of elements to pop.
   * @tparam T Value type of the input array.
   * @tparam R Size of the input array.
   * @param a Input array.
   * @return A copy of the input array with the first `N` elements removed.
   */
  template <int N, typename T, size_t R>
  constexpr std::array<T, R - N> front_mpop(std::array<T, R> const &a) {
    std::array<T, R - N> res;
    for (int i = N; i < R; ++i) res[i - N] = a[i];
    return res;
  }

  /**
   * @brief Make a new std::array by popping the first element of an existing std::array.
   *
   * @tparam T Value type of the input array.
   * @tparam R Size of the input array.
   * @param a Input std::array.
   * @return A copy of the input array with the first element removed.
   */
  template <typename T, size_t R>
  constexpr std::array<T, R - 1> front_pop(std::array<T, R> const &a) {
    return front_mpop<1>(a);
  }

  /**
   * @brief Make a new std::array by joining two existing std::array objects.
   *
   * @tparam T Value type of the arrays.
   * @tparam R1 Size of the input array #1.
   * @tparam R2 Size of the input array #2.
   * @param a1 Input std::array #1.
   * @param a2 Input std::array #2.
   * @return Array of size `R1 + R2` containing the elements of the first array followed by
   * the elements of the second array.
   */
  template <typename T, size_t R1, size_t R2>
  constexpr std::array<T, R1 + R2> join(std::array<T, R1> const &a1, std::array<T, R2> const &a2) {
    std::array<T, R1 + R2> res;
    for (int i = 0; i < R1; ++i) res[i] = a1[i];
    for (int i = 0; i < R2; ++i) res[R1 + i] = a2[i];
    return res;
  }

  /**
   * @brief Calculate the sum of all elements in a std::array.
   *
   * @tparam T Value type of the array.
   * @tparam R Size of the array.
   * @param a Input std::array.
   * @return Sum of all the elements of the input array. If its size is zero, return a default
   * constructed object of type `T`.
   */
  template <typename T, size_t R>
  constexpr auto sum(std::array<T, R> const &a) {
    if constexpr (R == 0)
      return T{};
    else {
      auto res = a[0];
      for (int i = 1; i < R; ++i) res += a[i];
      return res;
    }
  }

  /**
   * @brief Calculate the product of all elements in a std::array.
   *
   * @tparam T Value type of the array.
   * @tparam R Size of the array.
   * @param a Input std::array.
   * @return Product of all elements in the input array.
   */
  template <typename T, size_t R>
  constexpr auto product(std::array<T, R> const &a) {
    static_assert(R > 0, "Error in nda::stdutil::product: Only defined for R > 0");
    auto res = a[0];
    for (int i = 1; i < R; ++i) res *= a[i];
    return res;
  }

  /**
   * @brief Calculate the dot product of two std::array objects.
   *
   * @warning This function simply calculates the sum of the element-wise products of the two arrays. For arrays with
   * complex numbers, this might not be what you expect from a dot product.
   *
   * @tparam T Value type of input array #1.
   * @tparam U Value type of input array #2.
   * @tparam R Size of the input arrays.
   * @param a1 Input array #1.
   * @param a2 Input array #2.
   * @return Sum of the element-wise products of the two arrays.
   */
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

  /** @}  */

} // namespace nda::stdutil

#endif // STDUTILS_ARRAY_H
