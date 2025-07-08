// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides basic functions to create and manipulate arrays and views.
 */

#pragma once

#include "./clef/clef.hpp"
#include "./declarations.hpp"
#include "./exceptions.hpp"
#include "./layout/for_each.hpp"
#include "./mem/address_space.hpp"
#include "./traits.hpp"

#include <itertools/itertools.hpp>

#include <array>
#include <concepts>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @addtogroup av_factories
   * @{
   */

  /**
   * @brief Make an array of the given shape on the given address space and zero-initialize it.
   *
   * @details For a more specific array type consider using nda::basic_array::zeros.
   *
   * @tparam T Value type of the array.
   * @tparam AdrSp Address space of the array.
   * @tparam Int Integer type.
   * @tparam Rank Rank of the array.
   * @param shape Shape of the array.
   * @return Zero-initialized nda::array or nda::cuarray or scalar if `Rank == 0`.
   */
  template <typename T, mem::AddressSpace AdrSp = mem::Host, std::integral Int, auto Rank>
  auto zeros(std::array<Int, Rank> const &shape) {
    static_assert(AdrSp != mem::None);
    if constexpr (Rank == 0)
      return T{};
    else if constexpr (AdrSp == mem::Host)
      return array<T, Rank>::zeros(shape);
    else
      return cuarray<T, Rank>::zeros(shape);
  }

  /**
   * @brief Make an array of the given shape on the given address space and zero-initialize it.
   *
   * @details For a more specific array type consider using nda::basic_array::zeros.
   *
   * @tparam T Value type of the array.
   * @tparam AdrSp Address space of the array.
   * @tparam Ints Integer types.
   * @param is Extent (number of elements) along each dimension.
   * @return Zero-initialized nda::array or nda::cuarray or scalar if no arguments are given.
   */
  template <typename T, mem::AddressSpace AdrSp = mem::Host, std::integral... Ints>
  auto zeros(Ints... is) {
    return zeros<T, AdrSp>(std::array<long, sizeof...(Ints)>{static_cast<long>(is)...});
  }

  /**
   * @brief Make an array of the given shape and one-initialize it.
   *
   * @details For a more specific array type consider using nda::basic_array::ones.
   *
   * @tparam T Value type of the array.
   * @tparam Int Integer type.
   * @tparam Rank Rank of the array.
   * @param shape Shape of the array.
   * @return One-initialized nda::array or scalar if `Rank == 0`.
   */
  template <typename T, std::integral Int, auto Rank>
  auto ones(std::array<Int, Rank> const &shape)
    requires(nda::is_scalar_v<T>)
  {
    if constexpr (Rank == 0)
      return T{1};
    else { return array<T, Rank>::ones(shape); }
  }

  /**
   * @brief Make an array with the given dimensions and one-initialize it.
   *
   * @details For a more specific array type consider using nda::basic_array::ones.
   *
   * @tparam T Value type of the array.
   * @tparam Ints Integer types.
   * @param is Extent (number of elements) along each dimension.
   * @return One-initialized nda::array or scalar if no arguments are given.
   */
  template <typename T, std::integral... Ints>
  auto ones(Ints... is) {
    return ones<T>(std::array<long, sizeof...(Ints)>{is...});
  }

  /**
   * @brief Make a 1-dimensional integer array and initialize it with values of a given `nda::range`.
   *
   * @tparam Int Integer type/Value type of the created array.
   * @param first First value of the range.
   * @param last Last value of the range (excluded).
   * @param step Step size of the range.
   * @return 1-dimensional integer nda::array.
   */
  template <std::integral Int = long>
  auto arange(long first, long last, long step = 1) {
    auto r = range(first, last, step);
    auto a = array<Int, 1>(r.size());
    for (auto [x, v] : itertools::zip(a, r)) x = v;
    return a;
  }

  /**
   * @brief Make a 1-dimensional integer array and initialize it with values of a given `nda::range` with a step size
   * of 1 and a starting value of 0.
   *
   * @tparam Int Integer type/Value type of the created array.
   * @param last Last value of the range (excluded).
   * @return 1-dimensional integer nda::array.
   */
  template <std::integral Int = long>
  auto arange(long last) {
    return arange<Int>(0, last);
  }

  /**
   * @brief Make an array of the given shape and initialize it with random values from the uniform distribution over
   * [0, 1).
   *
   * @details For a more specific array type consider using nda::basic_array::rand.
   *
   * @tparam ValueType Value type of the array. Either a floating point or complex type.
   * @tparam Int Integer type.
   * @tparam Rank Rank of the array.
   * @param shape Shape of the array.
   * @return Random-initialized nda::array or scalar if `Rank == 0`.
   */
  template <typename ValueType = double, std::integral Int, auto Rank>
  auto rand(std::array<Int, Rank> const &shape)
    requires(std::is_floating_point_v<ValueType> or nda::is_complex_v<ValueType>)
  {
    if constexpr (Rank == 0) {
      return stack_array<ValueType, 1>::rand(1)[0];
    } else {
      return array<ValueType, Rank>::rand(shape);
    }
  }

  /**
   * @brief Make an array of the given dimensions and initialize it with random values from the uniform distribution
   * over [0, 1).
   *
   * @details For a more specific array type consider using nda::basic_array::rand.
   *
   * @tparam RealType Value type of the array.
   * @tparam Ints Integer types.
   * @param is Extent (number of elements) along each dimension.
   * @return Random-initialized nda::array or scalar if no arguments are given.
   */
  template <typename RealType = double, std::integral... Ints>
  auto rand(Ints... is) {
    return rand<RealType>(std::array<long, sizeof...(Ints)>{is...});
  }

  /**
   * @ingroup av_utils
   * @brief Get the extent of the first dimension of the array.
   *
   * @details Equivalent to `a.extent(0)` and `a.shape()[0]`.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @return Extent of the first dimension.
   */
  template <Array A>
  long first_dim(A const &a) {
    return a.extent(0);
  }

  /**
   * @ingroup av_utils
   * @brief Get the extent of the second dimension of the array.
   *
   * @details Equivalent to `a.extent(1)` and `a.shape()[1]`.
   *
   * @tparam A nda::Array type.
   * @param a nda::Array object.
   * @return Extent of the second dimension.
   */
  template <Array A>
  long second_dim(A const &a) {
    return a.extent(1);
  }

  /**
   * @brief Make a given object regular.
   *
   * @details The return type of this function depends on the input type `A`:
   * - If `A` is an nda::Array and not regular, then the return type is nda::basic_array.
   * - If `A` has a nested type `regular_t` which is not the same as `A`, then the return type is `A::regular_t`.
   * - Otherwise, the input type is simply forwarded.
   *
   * @note Rvalue references will be moved, while lvalue references will be copied.
   *
   * @tparam A Input type.
   * @param a Input object to make regular.
   * @return Regular object.
   */
  template <typename A, typename A_t = std::decay_t<A>>
  decltype(auto) make_regular(A &&a) {
    if constexpr (Array<A> and not is_regular_v<A>) {
      return basic_array{std::forward<A>(a)};
    } else if constexpr (requires { typename A_t::regular_t; }) {
      if constexpr (not std::is_same_v<A_t, typename A_t::regular_t>)
        return typename A_t::regular_t{std::forward<A>(a)};
      else
        return std::forward<A>(a);
    } else {
      return std::forward<A>(a);
    }
  }

  /**
   * @brief Convert an nda::MemoryArray to its regular type on host memory.
   *
   * @details If the input object is already on host memory, simply forward the argument (independent of its type).
   *
   * @tparam A nda::MemoryArray type.
   * @param a nda::MemoryArray object.
   * @return (Regular) object on host memory.
   */
  template <MemoryArray A>
  decltype(auto) to_host(A &&a) {
    if constexpr (not mem::on_host<A>) {
      return get_regular_host_t<A>{std::forward<A>(a)};
    } else {
      return std::forward<A>(a);
    }
  }

  /**
   * @brief Convert an nda::MemoryArray to its regular type on device memory.
   *
   * @details If the input object is already on device memory, simply forward the argument (independent of its type).
   *
   * @tparam A nda::MemoryArray type.
   * @param a nda::MemoryArray object.
   * @return (Regular) object on device memory.
   */
  template <MemoryArray A>
  decltype(auto) to_device(A &&a) {
    if constexpr (not mem::on_device<A>) {
      return get_regular_device_t<A>{std::forward<A>(a)};
    } else {
      return std::forward<A>(a);
    }
  }

  /**
   * @brief Convert an nda::MemoryArray to its regular type on unified memory.
   *
   * @details If the input object is already on unified memory, simply forward the argument (independent of its type).
   *
   * @tparam A nda::MemoryArray type.
   * @param a nda::MemoryArray object.
   * @return (Regular) object on unified memory.
   */
  template <MemoryArray A>
  decltype(auto) to_unified(A &&a) {
    if constexpr (not mem::on_unified<A>) {
      return get_regular_unified_t<A>{std::forward<A>(a)};
    } else {
      return std::forward<A>(a);
    }
  }

  /**
   * @brief Resize a given regular array to the given shape or check if a given view as the correct shape.
   *
   * @details Regular types are resized (if necessary) by calling nda::basic_array::resize while views are checked for
   * the correct shape.
   *
   * Throws an exception if the shape of the view does not match the given shape.
   *
   * @tparam A Type of the object.
   * @param a Object to resize or check.
   * @param sha New shape.
   */
  template <typename A>
  void resize_or_check_if_view(A &a, std::array<long, A::rank> const &sha)
    requires(is_regular_or_view_v<A>)
  {
    if (a.shape() == sha) return;
    if constexpr (is_regular_v<A>) {
      a.resize(sha);
    } else {
      NDA_RUNTIME_ERROR << "Error in nda::resize_or_check_if_view: Size mismatch: " << a.shape() << " != " << sha;
    }
  }

  /**
   * @brief Make an nda::basic_array_view with a const value type from a given nda::basic_array.
   *
   * @tparam T Value type of the array.
   * @tparam R Rank of the array.
   * @tparam LP Layout policy of the array.
   * @tparam A Algebra of the array.
   * @tparam CP Container policy of the array.
   *
   * @param a nda::basic_array object.
   * @return nda::basic_array_view object with a const value type.
   */
  template <typename T, int R, typename LP, char A, typename CP>
  auto make_const_view(basic_array<T, R, LP, A, CP> const &a) {
    return basic_array_view<T const, R, LP, A>{a};
  }

  /**
   * @brief Make an nda::basic_array_view with a const value type from a given nda::basic_array_view.
   *
   * @tparam T Value type of the view.
   * @tparam R Rank of the view.
   * @tparam LP Layout policy of the view.
   * @tparam Algebra Algebra of the view.
   * @tparam AP Accessor policy of the view.
   * @tparam OP Owning policy of the view.
   *
   * @param a nda::basic_array_view object.
   * @return nda::basic_array_view object with a const value type.
   */
  template <typename T, int R, typename LP, char A, typename AP, typename OP>
  auto make_const_view(basic_array_view<T, R, LP, A, AP, OP> const &a) {
    return basic_array_view<T const, R, LP, A, AP, OP>{a};
  }

  /**
   * @brief Make an nda::array_view of a given nda::basic_array.
   *
   * @tparam T Value type of the array.
   * @tparam R Rank of the array.
   * @tparam LP Layout policy of the array.
   * @tparam A Algebra of the array.
   * @tparam CP Container policy of the array.
   *
   * @param a nda::basic_array object.
   * @return nda::array_view object.
   */
  template <typename T, int R, typename LP, char A, typename CP>
  auto make_array_view(basic_array<T, R, LP, A, CP> const &a) {
    return array_view<T, R>{a};
  }

  /**
   * @brief Make an nda::array_view of a given nda::basic_array_view.
   *
   * @tparam T Value type of the view.
   * @tparam R Rank of the view.
   * @tparam LP Layout policy of the view.
   * @tparam A Algebra of the view.
   * @tparam AP Accessor policy of the view.
   * @tparam OP Owning policy of the view.
   *
   * @param a nda::basic_array_view object.
   * @return nda::array_view object.
   */
  template <typename T, int R, typename LP, char A, typename AP, typename OP>
  auto make_array_view(basic_array_view<T, R, LP, A, AP, OP> const &a) {
    return array_view<T, R>{a};
  }

  /**
   * @brief Make an nda::array_const_view of a given nda::basic_array.
   *
   * @tparam T Value type of the array.
   * @tparam R Rank of the array.
   * @tparam LP Layout policy of the array.
   * @tparam A Algebra of the array.
   * @tparam CP Container policy of the array.
   *
   * @param a nda::basic_array object.
   * @return nda::array_const_view object.
   */
  template <typename T, int R, typename LP, char A, typename CP>
  auto make_array_const_view(basic_array<T, R, LP, A, CP> const &a) {
    return array_const_view<T, R>{a};
  }

  /**
   * @brief Make an nda::array_const_view of a given nda::basic_array_view.
   *
   * @tparam T Value type of the view.
   * @tparam R Rank of the view.
   * @tparam LP Layout policy of the view.
   * @tparam A Algebra of the view.
   * @tparam AP Accessor policy of the view.
   * @tparam OP Owning policy of the view.
   *
   * @param a nda::basic_array_view object.
   * @return nda::array_const_view object.
   */
  template <typename T, int R, typename LP, char A, typename AP, typename OP>
  auto make_array_const_view(basic_array_view<T, R, LP, A, AP, OP> const &a) {
    return array_const_view<T, R>{a};
  }

  /**
   * @brief Make an nda::matrix_view of a given nda::basic_array.
   *
   * @tparam T Value type of the array.
   * @tparam R Rank of the array.
   * @tparam LP Layout policy of the array.
   * @tparam A Algebra of the array.
   * @tparam CP Container policy of the array.
   *
   * @param a nda::basic_array object.
   * @return nda::matrix_view object.
   */
  template <typename T, int R, typename LP, char A, typename CP>
  auto make_matrix_view(basic_array<T, R, LP, A, CP> const &a) {
    return matrix_view<T, LP>{a};
  }

  /**
   * @brief Make an nda::matrix_view of a given nda::basic_array_view.
   *
   * @tparam T Value type of the view.
   * @tparam R Rank of the view.
   * @tparam LP Layout policy of the view.
   * @tparam A Algebra of the view.
   * @tparam AP Accessor policy of the view.
   * @tparam OP Owning policy of the view.
   *P
   * @param a nda::basic_array_view object.
   * @return nda::matrix_view object.
   */
  template <typename T, int R, typename LP, char A, typename AP, typename OP>
  auto make_matrix_view(basic_array_view<T, R, LP, A, AP, OP> const &a) {
    return matrix_view<T, LP>{a};
  }

  /**
   * @ingroup av_utils
   * @brief Equal-to comparison operator for two nda::Array objects.
   *
   * @tparam LHS nda::Array type of left hand side.
   * @tparam RHS nda::Array type of right hand side.
   * @param lhs Left hand side array operand.
   * @param rhs Right hand side array operand.
   * @return True if all elements are equal, false otherwise.
   */
  template <Array LHS, Array RHS>
  bool operator==(LHS const &lhs, RHS const &rhs) {
    // FIXME not implemented in clang
#ifndef __clang__
    static_assert(std::equality_comparable_with<get_value_t<LHS>, get_value_t<RHS>>,
                  "Error in nda::operator==: Only defined when elements are comparable");
#endif
    if (lhs.shape() != rhs.shape()) return false;
    bool r = true;
    nda::for_each(lhs.shape(), [&](auto &&...x) { r &= (lhs(x...) == rhs(x...)); });
    return r;
  }

  /**
   * @ingroup av_utils
   * @brief Equal-to comparison operator for a 1-dimensional nda::Array and a `std::ranges::contiguous_range`.
   *
   * @tparam A nda::Array type of left hand side.
   * @tparam R `std::ranges::contiguous_range` type of right hand side.
   * @param a Left hand side array operand.
   * @param rg Right hand side range operand.
   * @return True if all elements are equal, false otherwise.
   */
  template <ArrayOfRank<1> A, std::ranges::contiguous_range R>
  bool operator==(A const &a, R const &rg) {
    return a == basic_array_view{rg};
  }

  /**
   * @ingroup av_utils
   * @brief Equal-to comparison operator for a `std::ranges::contiguous_range` and a 1-dimensional nda::Array.
   *
   * @tparam R `std::ranges::contiguous_range` type of left hand side.
   * @tparam A nda::Array type of right hand side.
   * @param rg Left hand side range operand.
   * @param a Right hand side array operand.
   * @return True if all elements are equal, false otherwise.
   */
  template <std::ranges::contiguous_range R, ArrayOfRank<1> A>
  bool operator==(R const &rg, A const &a) {
    return a == rg;
  }

  /**
   * @ingroup clef_autoassign
   * @brief Overload of nda::clef::clef_auto_assign function for nda::Array objects.
   *
   * @tparam A nda::Array type.
   * @tparam F Callable type.
   * @param a nda::Array object.
   * @param f Callable object.
   */
  template <Array A, typename F>
  void clef_auto_assign(A &&a, F &&f) { // NOLINT (Should we forward the references?)
    nda::for_each(a.shape(), [&a, &f](auto &&...x) {
      if constexpr (clef::is_function<std::decay_t<decltype(f(x...))>>) {
        clef_auto_assign(a(x...), f(x...));
      } else {
        a(x...) = f(x...);
      }
    });
  }

  /**
   * @ingroup layout_utils
   * @brief Check if a given nda::MemoryArray has a block-strided layout.
   *
   * @details If the array is block-strided, return the number of blocks, their size and the stride.
   *
   * An array is considered to be block-strided if its data in memory is laid out as contiguous blocks of the same size
   * (> 1) repeated with a single stride in memory.
   *
   * A block-strided array has at most 1 non-contiguous index `m`, i.e. `strides[order[m]] != strides[order[m+1]] *
   * shape[order[m+1]]`.
   *
   * @tparam A nda::MemoryArray type.
   * @param a Array object.
   * @return An optional tuple (if block-strided) containing the number of blocks, their size and the stride between
   * blocks.
   */
  template <MemoryArray A>
  auto get_block_layout(A const &a) {
    EXPECTS(!a.empty());
    using opt_t = std::optional<std::tuple<int, int, int>>;

    auto const &shape   = a.indexmap().lengths();
    auto const &strides = a.indexmap().strides();
    auto const &order   = a.indexmap().stride_order;

    int data_size  = shape[order[0]] * strides[order[0]];
    int block_size = data_size;
    int block_str  = data_size;
    int n_blocks   = 1;

    for (auto n : range(A::rank)) {
      auto inner_size = (n == A::rank - 1) ? 1 : strides[order[n + 1]] * shape[order[n + 1]];
      if (strides[order[n]] != inner_size) {
        if (block_size < data_size) // second strided dimension
          return opt_t{};
        // found a strided dimension with (assumed) contiguous inner blocks
        n_blocks   = a.size() / inner_size;
        block_size = inner_size;
        block_str  = strides[order[n]];
      }
    }
    ASSERT(n_blocks * block_size == a.size());
    ASSERT(n_blocks * block_str == shape[order[0]] * strides[order[0]]);
    return opt_t{std::make_tuple(n_blocks, block_size, block_str)};
  }

  /**
   * @brief Join a sequence of nda::Array types along an existing axis.
   *
   * @details The arrays must have the same value type and also shape, except in the dimension corresponding to the
   * given axis (the first, by default).
   *
   * @tparam Axis The axis (dimension) along which to concatenate (default: 0).
   * @tparam A0 nda::Array type.
   * @tparam As nda::Array types.
   * @param a0 First array object.
   * @param as Remaining array objects.
   * @return New nda::array with the concatenated data.
   */
  template <size_t Axis = 0, Array A0, Array... As>
  auto concatenate(A0 const &a0, As const &...as) {
    // sanity checks
    auto constexpr rank = A0::rank;
    static_assert(Axis < rank);
    static_assert(have_same_rank_v<A0, As...>);
    static_assert(have_same_value_type_v<A0, As...>);
    for (auto ax [[maybe_unused]] : range(rank)) { EXPECTS(ax == Axis or ((a0.extent(ax) == as.extent(ax)) and ... and true)); }

    // construct concatenated array
    auto new_shape  = a0.shape();
    new_shape[Axis] = (as.extent(Axis) + ... + new_shape[Axis]);
    auto new_array  = array<get_value_t<A0>, rank>(new_shape);

    // slicing helper function
    auto slice_Axis = [](Array auto &a, range r) {
      auto all_or_range = std::make_tuple(range::all, r);
      return [&]<auto... Is>(std::index_sequence<Is...>) { return a(std::get < Is == Axis > (all_or_range)...); }(std::make_index_sequence<rank>{});
    };

    // initialize concatenated array
    long offset = 0;
    for (auto const &a_view : {basic_array_view(a0), basic_array_view(as)...}) {
      slice_Axis(new_array, range(offset, offset + a_view.extent(Axis))) = a_view;
      offset += a_view.extent(Axis);
    }

    return new_array;
  };

  /** @} */

} // namespace nda
