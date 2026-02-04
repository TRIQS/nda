// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides various algorithms to be used with nda::Array objects.
 */

#pragma once

#include "./basic_functions.hpp"
#include "./concepts.hpp"
#include "./layout/for_each.hpp"
#include "./layout/range.hpp"
#include "./macros.hpp"
#include "./map.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace nda {

  /**
   * @addtogroup av_algs
   * @{
   */

  // FIXME : CHECK ORDER of the LOOP !
  /**
   * @brief Perform a fold operation on the given nda::Array object.
   *
   * @details It calculates the following (where r is an initial value);
   *
   * @code{.cpp}
   * auto res = f(...f(f(f(r, a(0,...,0)), a(0,...,1)), a(0,...,2)), ...);
   * @endcode
   *
   * @note The array is always traversed in C-order.
   *
   * @tparam A nda::Array type.
   * @tparam F Callable type.
   * @tparam R Type of the initial value.
   * @param f Callable object taking two arguments compatible with the initial value and the array value type.
   * @param a nda::Array object.
   * @param r Initial value.
   * @return Result of the fold operation.
   */
  template <Array A, typename F, typename R>
  auto fold(F f, A const &a, R r) {
    // cast the initial value to the return type of f to avoid narrowing
    using res_t = std::decay_t<decltype(make_regular(f(r, get_value_t<A>{})))>;
    auto res    = res_t{r};
    nda::for_each(a.shape(), [&a, &res, &f](auto &&...args) { res = f(res, a(args...)); });
    return res;
  }

  /// The same as nda::fold, except that the initial value is a default constructed value type of the array.
  template <Array A, typename F>
  auto fold(F f, A const &a) {
    return fold(std::move(f), a, get_value_t<A>{});
  }

  /**
   * @brief Does any of the elements of the array evaluate to true?
   *
   * @details The given nda::Array object can also be some lazy expression that evaluates to a boolean. For example:
   *
   * @code{.cpp}
   * auto A = nda::array<double, 2>::rand(2, 3);
   * auto greater05 = nda::map([](auto x) { return x > 0.5; })(A);
   * auto res = nda::any(greater05);
   * @endcode
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @return True if at least one element of the array evaluates to true, false otherwise.
   */
  template <Array A>
  bool any(A const &a) {
    static_assert(std::is_same_v<get_value_t<A>, bool>, "Error in nda::any: Value type of the array must be bool");
    return fold([](bool r, auto const &x) -> bool { return r or bool(x); }, a, false);
  }

  /**
   * @brief Do all elements of the array evaluate to true?
   *
   * @details The given nda::Array object can also be some lazy expression that evaluates to a boolean. For example:
   *
   * @code{.cpp}
   * auto A = nda::array<double, 2>::rand(2, 3);
   * auto greater0 = nda::map([](auto x) { return x > 0.0; })(A);
   * auto res = nda::all(greater0);
   * @endcode
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @return True if all elements of the array evaluate to true, false otherwise.
   */
  template <Array A>
  bool all(A const &a) {
    static_assert(std::is_same_v<get_value_t<A>, bool>, "Error in nda::all: Value type of the array must be bool");
    return fold([](bool r, auto const &x) -> bool { return r and bool(x); }, a, true);
  }

  /**
   * @brief Find the maximum element of an array.
   *
   * @details It uses nda::fold and `std::max`.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @return Maximum element of the array.
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
   * @brief Find the minimum element of an array.
   *
   * @details It uses nda::fold and `std::min`.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @return Minimum element of the array.
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

  /**
   * @ingroup av_math
   * @brief Calculate the Frobenius norm of a 2-dimensional array.
   *
   * @tparam A nda::ArrayOfRank<2> type.
   * @param a Array object.
   * @return Frobenius norm of the array/matrix.
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
   * @brief Sum all the elements of an nda::Array object.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @return Sum of all elements.
   */
  template <Array A, typename Value = get_value_t<A>>
  auto sum(A const &a)
    requires(nda::Scalar<Value> or nda::Array<Value>)
  {
    if constexpr (nda::Scalar<Value>) {
      return fold(std::plus<>{}, a);
    } else { // Array<Value>
      return fold(std::plus<>{}, a, Value::zeros(get_first_element(a).shape()));
    }
  }

  /**
   * @brief Sum elements of an nda::Array along specified axes.
   *
   * @details This function behaves similar to `numpy.sum`. The result is an array with rank reduced by the number of 
   * axes summed over, i.e. if the original array has rank \f$ R \f$ and we sum over \f$ N \f$ axes, the resulting 
   * array has rank \f$ R - N \f$.
   * 
   * If all axes are summed over, \f$ R = N \f$, the call is dispatched to nda::sum.
   *
   * If no axes are specified, \f$ N = 0 \f$, a copy of the input array is returned.
   *
   * The given axes are expected to be unique and in the range \f$ [0, R-1] \f$.
   *
   * @tparam A nda::Array type.
   * @tparam N Number of axes to sum over.
   * @param a nda::Array object.
   * @param axes Array of axis indices to sum over.
   * @return An nda::basic_array with rank reduced by \f$ N \f$, or a scalar if all axes are summed.
   */
  template <Array A, std::integral I, size_t N>
    requires(get_rank<A> >= N and nda::Scalar<get_value_t<A>>)
  auto sum(A const &a, std::array<I, N> axes) {
    // if no axes are specified, return a copy of the input
    if constexpr (N == 0) {
      return make_regular(a);
    } else {
      // sort axes and check validity
      std::ranges::sort(axes);
      EXPECTS(std::ranges::adjacent_find(axes) == axes.end());
      EXPECTS(axes.front() >= 0 and axes.back() < get_rank<A>);

      constexpr int res_rank = get_rank<A> - static_cast<int>(N);

      if constexpr (res_rank == 0) {
        return sum(a);
      } else {
        // get the result shape and the axes that we keep (are not summed over)
        std::array<long, res_rank> keep_axes, res_shape;
        for (int i = 0; auto ax : nda::range(get_rank<A>)) {
          if (!std::ranges::binary_search(axes, ax)) {
            keep_axes[i]   = ax;
            res_shape[i++] = a.shape()[ax];
          }
        }

        // create the result array initialized to zero
        auto res = array<get_value_t<A>, res_rank>::zeros(res_shape);

        // loop over all indices of the input array and sum over the specified axes
        nda::for_each(a.shape(), [&](auto... idxs) {
          auto idx_arr = std::array{idxs...};
          std::apply([&](auto... keep) { res(idx_arr[keep]...) += a(idxs...); }, keep_axes);
        });

        return res;
      }
    }
  }

  /**
   * @brief Sum elements of an nda::Array along a specified axis.
   *
   * @details It simply calls nda::sum(A const &, std::array<I, N>).
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @param axis The axis along which to sum.
   * @return An nda::array with rank reduced by 1 or a scalar if the original array has rank 1.
   */
  template <Array A>
    requires(get_rank<A> >= 1 and nda::Scalar<get_value_t<A>>)
  auto sum(A const &a, int axis) {
    return sum(a, std::array{axis});
  }

  /**
   * @brief Multiply all the elements of an nda::Array object.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @return Product of all elements.
   */
  template <Array A, typename Value = get_value_t<A>>
  auto product(A const &a)
    requires(nda::Scalar<Value> or nda::Array<Value>)
  {
    if constexpr (nda::Scalar<Value>) {
      return fold(std::multiplies<>{}, a, get_value_t<A>{1});
    } else { // Array<Value>
      return fold(std::multiplies<>{}, a, Value::ones(get_first_element(a).shape()));
    }
  }

  /**
   * @brief Hadamard product of two nda::Array objects.
   *
   * @tparam A nda::Array type.
   * @tparam B nda::Array type.
   * @param a nda::Array object.
   * @param b nda::Array object.
   * @return A lazy nda::expr_call object representing the elementwise product of the two input objects.
   */
  template <Array A, Array B>
    requires(nda::get_rank<A> == nda::get_rank<B>)
  [[nodiscard]] constexpr auto hadamard(A &&a, B &&b) {
    return nda::map([](auto const &x, auto const &y) { return x * y; })(std::forward<A>(a), std::forward<B>(b));
  }

  /**
   * @brief Hadamard product of two std::array objects.
   *
   * @tparam T Value type of the first array.
   * @tparam U Value type of the second array.
   * @tparam R Size of the arrays.
   * @param a std::array object.
   * @param b std::array object.
   * @return std::array containing the elementwise product of the two input arrays.
   */
  template <typename T, typename U, size_t R>
  [[nodiscard]] constexpr auto hadamard(std::array<T, R> const &a, std::array<U, R> const &b) {
    return a * b;
  }

  /**
   * @brief Hadamard product of two std::vector objects.
   *
   * @tparam T Value type of the first input vector.
   * @tparam U Value type of the second input vector.
   * @param a std::vector object.
   * @param b std::vector object.
   * @return std::vector containing the elementwise product of the two input vectors.
   */
  template <typename T, typename U>
  [[nodiscard]] constexpr auto hadamard(std::vector<T> const &a, std::vector<U> const &b) {
    using TU = decltype(std::declval<T>() * std::declval<U>());
    EXPECTS(a.size() == b.size());

    std::vector<TU> c(a.size());
    for (auto i : range(c.size())) c[i] = a[i] * b[i];
    return c;
  }

  /**
   * @brief Hadamard product of two arithmetic types.
   *
   * @tparam T nda::Scalar type of the first input.
   * @tparam U nda::Scalar type of the second input.
   * @param a First input.
   * @param b Second input.
   * @return Product of the two inputs.
   */
  constexpr auto hadamard(nda::Scalar auto a, nda::Scalar auto b) { return a * b; }

  /** @} */

} // namespace nda
