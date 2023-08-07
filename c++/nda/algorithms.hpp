// Copyright (c) 2019-2022 Simons Foundation
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

namespace nda {

  // FIXME : CHECK ORDER or the LOOP !
  // --------------- fold  ------------------------
  /**
   * @tparam A
   * @tparam F is a function f(x, r)
   * @tparam R
   * @param f
   * @param a
   * @param r
   *
   * fold computes f(f(r, a(0,0)), a(0,1), ...)  etc
   */
  template <Array A, typename F, typename R>
  auto fold(F f, A const &a, R r) {
    decltype(f(r, get_value_t<A>{})) r2 = r;
    // to take into account that f may be double,double -> double, while one passes 0 (an int...)
    // R = int, R2= double in such case, and the result will be a double, or narrowing will occur
    nda::for_each(a.shape(), [&a, &r2, &f](auto &&...args) { r2 = f(r2, a(args...)); });
    return r2;
  }

  /**
   * @tparam A
   * @tparam F is a function f(x, r)
   * @param f
   * @param a
   *
   * fold computes f(f(r, a(0,0)), a(0,1), ...)  etc
   */
  template <Array A, typename F>
  auto fold(F f, A const &a) {
    return fold(std::move(f), a, get_value_t<A>{});
  }

  // --------------- applications of fold -----------------------

  /// Returns true iif at least one element of the array is true
  /// \ingroup Algorithms
  template <Array A>
  bool any(A const &a) {
    static_assert(std::is_same_v<get_value_t<A>, bool>, "OOPS");
    return fold([](bool r, auto const &x) -> bool { return r or bool(x); }, a, false);
  }

  /// Returns true iif all elements of the array are true
  /// \ingroup Algorithms
  template <Array A>
  bool all(A const &a) {
    static_assert(std::is_same_v<get_value_t<A>, bool>, "OOPS");
    return fold([](bool r, auto const &x) -> bool { return r and bool(x); }, a, true);
  }

  /**
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The maximum element of A
   * \ingroup Algorithms
   */
  template <Array A>
  auto max_element(A const &a) {
    return fold(
       [](auto const &x, auto const &y) {
         using std::max;
         return max(x, y);
       },
       a, get_first_element(a));
  }

  /**
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The minimum element of A
   * \ingroup Algorithms
   */
  template <Array A>
  auto min_element(A const &a) {
    return fold(
       [](auto const &x, auto const &y) {
         using std::min;
         return min(x, y);
       },
       a, get_first_element(a));
  }

  // FIXME in matrix functions ?

  // --------------- Computation of the matrix norm ------------------------

  /**
   * @tparam A Anything modeling the ArrayOfRank<2> concept
   * @param m The object of type A
   */
  template <ArrayOfRank<2> A>
  double frobenius_norm(A const &a) {
    return std::sqrt(fold(
       [](double r, auto const &x) -> double {
         auto ab = std::abs(x);
         return r + ab * ab;
       },
       a, double(0)));
  }

  /**
   * Return the sum of all array elements added to value_t{0}
   *
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The sum of all elements of a 
   * \ingroup Algorithms
   */
  template <Array A>
  auto sum(A const &a)
    requires(nda::is_scalar_v<get_value_t<A>>)
  {
    return fold(std::plus<>{}, a);
  }

  /**
   * Return the product of all array elements multiplied with value_t{1}
   *
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The product of all elements of a
   * \ingroup Algorithms
   */
  template <Array A>
  auto product(A const &a)
    requires(nda::is_scalar_v<get_value_t<A>>)
  {
    return fold(std::multiplies<>{}, a, get_value_t<A>{1});
  }

} // namespace nda
