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

  //TODO: Maybe add another fold function that can interact with SIMD types.
//  template <Array A, typename F_SIMD, typename F_SCALAR, Vectorizable R>
//    requires(std::is_same_v<simd::dispatch_policy_t<A, R>, simd::vectorize_t> or std::is_same_v<simd::dispatch_policy_t<A, R>, simd::emulate_t>)
//  auto fold(F_SIMD f_simd, F_SCALAR f_scalar, A const &a, native_simd<R> r_simd, R r_scalar) {
//    nda::for_each_static<0, get_layout_info<A>.stride_order, native_simd<R>::size>(
//       a.shape(),
//       [&a, &r_simd, &f_simd](auto &&...args) { r_simd = f_simd(r_simd, native_simd<R>(a.load(simd::dispatch_policy_t<A, R>{}, args...))); },
//       [&a, &r_scalar, &f_scalar](auto &&...args) { r_scalar = f_scalar(r_scalar, a(args...)); });
//    alignas(native_simd<R>::arch_type::alignment()) std::array<R, r_simd.size()> res;
//    r_simd.store(res.data());
//    for (int i = 0; i < r_simd.size(); i++) { r_scalar = f_scalar(r_scalar, res[i]); }
//    return r_scalar;
//  }

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
    using dispatch_t = simd::dispatch_policy_t<A>;
    if constexpr (std::is_same_v<dispatch_t, simd::scalar_t>) {
      return fold(
         [](auto const &x, auto const &y) {
           using std::max;
           return max(x, y);
         },
         a, get_first_element(a));
    } else {
      using value_t = get_value_t<A>;
      using simd_t  = native_simd<value_t>;
      simd_t max_simd(get_first_element(a));
      auto f_simd        = [&a, &max_simd](auto &&...args) { max_simd = xsimd::max(max_simd, a.load(dispatch_t{}, args...)); };
      value_t max_scalar = get_first_element(a);
      auto f_scalar      = [&a, &max_scalar](auto &&...args) { max_scalar = std::max(max_scalar, a(args...)); };
      nda::for_each_static<0, get_layout_info<A>.stride_order, simd_t::size>(a.shape(), std::move(f_simd), std::move(f_scalar));
      return std::max(max_scalar, xsimd::reduce_max(max_simd));
    }
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
    using dispatch_t = simd::dispatch_policy_t<A>;
    if constexpr (std::is_same_v<dispatch_t, simd::scalar_t>) {
      return fold(
         [](auto const &x, auto const &y) {
           using std::min;
           return min(x, y);
         },
         a, get_first_element(a));
    } else {
      using value_t = get_value_t<A>;
      using simd_t  = native_simd<value_t>;
      simd_t min_simd(get_first_element(a));
      auto f_simd        = [&a, &min_simd](auto &&...args) { min_simd = xsimd::min(min_simd, a.load(dispatch_t{}, args...)); };
      value_t min_scalar = get_first_element(a);
      auto f_scalar      = [&a, &min_scalar](auto &&...args) { min_scalar = std::min(min_scalar, a(args...)); };
      nda::for_each_static<0, get_layout_info<A>.stride_order, simd_t::size>(a.shape(), std::move(f_simd), std::move(f_scalar));
      return std::min(min_scalar, xsimd::reduce_min(min_simd));
    }
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
    using dispatch_t = simd::dispatch_policy_t<A>;
    if constexpr (std::is_same_v<dispatch_t, simd::scalar_t> or is_complex_v<get_value_t<A>>) {
      return std::sqrt(fold(
         [](double r, auto const &x) -> double {
           auto abs = std::abs(x);
           return xsimd::fma(abs, abs, r);
         },
         a, double(0)));
    } else {
      using value_t = get_value_t<A>;
      using simd_t  = native_simd<value_t>;
      simd_t r_simd(value_t(0));
      auto f_simd = [&a, &r_simd](auto &&...args) {
        simd_t x   = a.load(dispatch_t{}, args...);
        simd_t abs = xsimd::abs(x);
        r_simd     = xsimd::fma(abs, abs, r_simd);
      };
      double r      = 0;
      auto f_scalar = [&a, &r](auto &&...args) {
        auto abs = std::abs(a(args...));
        r        = xsimd::fma(abs, abs, r);
      };
      nda::for_each_static<0, get_layout_info<A>.stride_order, simd_t::size>(a.shape(), std::move(f_simd), std::move(f_scalar));
      return std::sqrt((static_cast<double>(xsimd::reduce_add(r_simd)) + r));
    }
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
      using dispatch_t = simd::dispatch_policy_t<A>;
      if constexpr (std::is_same_v<dispatch_t, simd::scalar_t>) {
        return fold(std::plus<>{}, a);
      } else {
        using value_t = get_value_t<A>;
        using simd_t  = native_simd<value_t>;
        simd_t sum_simd(value_t{0});
        auto f_simd = [&a, &sum_simd](auto &&...args) { sum_simd += a.load(dispatch_t{}, args...); };
        value_t sum_scalar{0};
        auto f_scalar = [&a, &sum_scalar](auto &&...args) { sum_scalar += a(args...); };
        nda::for_each_static<0, get_layout_info<A>.stride_order, simd_t::size>(a.shape(), std::move(f_simd), std::move(f_scalar));
        return sum_scalar + xsimd::reduce_add(sum_simd);
      }
    } else {
      // Array<Value>
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
   * @details It simply calls nda::sum(A const &, `std::array<I, N>`).
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
      using dispatch_t = simd::dispatch_policy_t<A>;
      if constexpr (std::is_same_v<dispatch_t, simd::scalar_t>) {
        return fold(std::multiplies<>{}, a);
      } else {
        using value_t = get_value_t<A>;
        using simd_t  = native_simd<value_t>;
        simd_t product_simd(value_t{1});
        auto f_simd = [&a, &product_simd](auto &&...args) { product_simd *= a.load(dispatch_t{}, args...); };
        value_t product_scalar{1};
        auto f_scalar = [&a, &product_scalar](auto &&...args) { product_scalar *= a(args...); };
        nda::for_each_static<0, get_layout_info<A>.stride_order, simd_t::size>(a.shape(), std::move(f_simd), std::move(f_scalar));
        return product_scalar * xsimd::reduce_mul(product_simd);
      }
    } else {
      // Array<Value>
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
    if constexpr (is_simd_enabled_v<A> and is_simd_enabled_v<B> and std::is_same_v<get_value_t<A>, get_value_t<B>>) {
      using value_t = get_value_t<A>;
      using simd_t  = native_simd<value_t>;
      struct mul {
        value_t operator()(const value_t &x, const value_t &y) const { return x * y; }
        simd_t load(const simd_t &x, const simd_t &y) const { return x * y; };
      };
      return nda::map(mul{})(std::forward<A>(a), std::forward<B>(b));
    } else {
      return nda::map([](auto const &x, auto const &y) { return x * y; })(std::forward<A>(a), std::forward<B>(b));
    }
  }

  /**
   * @brief Hadamard product of two `std::array` objects.
   *
   * @tparam T Value type of the first array.
   * @tparam U Value type of the second array.
   * @tparam R Size of the arrays.
   * @param a `std::array` object.
   * @param b `std::array` object.
   * @return `std::array` containing the elementwise product of the two input arrays.
   */
  template <typename T, typename U, size_t R>
  [[nodiscard]] constexpr auto hadamard(std::array<T, R> const &a, std::array<U, R> const &b) {
    return a * b;
  }

  /**
   * @brief Hadamard product of two `std::vector` objects.
   *
   * @tparam T Value type of the first input vector.
   * @tparam U Value type of the second input vector.
   * @param a `std::vector` object.
   * @param b `std::vector` object.
   * @return `std::vector` containing the elementwise product of the two input vectors.
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
