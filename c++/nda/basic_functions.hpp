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
namespace nda {

  // --------------------------- zeros ------------------------

  /// Make a array of zeros with the given dimensions.
  /// Return a scalar for the case of rank zero.
  /// If we want more general array, use the static factory zeros [See also]
  template <typename T, std::integral Int, auto Rank>
  auto zeros(std::array<Int, Rank> const &shape) {
    // For Rank == 0 we should return the underlying scalar_t
    if constexpr (Rank == 0)
      return T{0};
    else
      return array<T, Rank>::zeros(shape);
  }

  ///
  template <typename T, std::integral... Int>
  auto zeros(Int... i) {
    return zeros<T>(std::array<long, sizeof...(Int)>{i...});
  }

  // --------------------------- make_regular ------------------------

  /**
   * Return a basic_array if A fullfills the Array concept,
   * else forward the object without midifications
   *
   * @tparam A
   * @param x
   */
  template <typename A>
  auto make_regular(A &&x) {
    using A_t = std::decay_t<A>;
    if constexpr (is_ndarray_v<A_t>)
      return basic_array<get_value_t<A_t>, get_rank<A_t>, C_layout, get_algebra<A_t>, heap>{std::forward<A>(x)};
    else
      return x;
  }

  template <typename A>
  using get_regular_t = decltype(make_regular(std::declval<A>()));

  // --------------------------- resize_or_check_if_view------------------------

  /** 
   * Resize if A is a container, or assert that the view has the right dimension if A is view
   *
   * @tparam A
   * @param a A container or a view
   */
  template <typename A>
  void resize_or_check_if_view(A &a, std::array<long, A::rank> const &sha) REQUIRES(is_regular_or_view_v<A>) {
    if (a.shape() == sha) return;
    if constexpr (is_regular_v<A>) {
      a.resize(sha);
    } else {
      NDA_RUNTIME_ERROR << "Size mismatch : view class shape = " << a.shape() << " expected " << sha;
    }
  }

  // --------------- make_const_view------------------------

  /// Make a view const
  template <typename T, int R, typename L, char Algebra, typename CP>
  basic_array_view<T const, R, L, Algebra> make_const_view(basic_array<T, R, L, Algebra, CP> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AP, typename OP>
  basic_array_view<T const, R, L, Algebra, AP, OP> make_const_view(basic_array_view<T, R, L, Algebra, AP, OP> const &a) {
    return {a};
  }

  // --------------- make_array_view------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  array_view<T, R> make_array_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  array_view<T, R> make_array_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  // --------------- make_array_const_view------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  array_const_view<T, R> make_array_const_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  array_const_view<T, R> make_array_const_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  // --------------- make_matrix_view------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  matrix_view<T, L> make_matrix_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  matrix_view<T, L> make_matrix_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  /*  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>*/
  //matrix_view<T const, L> make_matrix_const_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
  //return {a};
  //}

  //template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  //matrix_view<T const, L> make_matrix_const_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
  //return {a};
  //}

  // --------------- operator == ---------------------

  /// True iif all elements are equal.
  template <typename A, typename B>
  bool operator==(A const &a, B const &b) REQUIRES(is_ndarray_v<A> and is_ndarray_v<B>) {
 // FIXME not implemented in clang .. readd when done for better error message
#ifndef __clang__
    static_assert(std::equality_comparable_with<get_value_t<A>, get_value_t<B>>, "A == B is only defined when their element can be compared");
#endif
    if (a.shape() != b.shape()) return false;
    bool r = true;
    nda::for_each(a.shape(), [&](auto &&...x) { r &= (a(x...) == b(x...)); });
    return r;
  }

  // ------------------------------- auto_assign --------------------------------------------

  template <typename A, typename F>
  void clef_auto_assign(A &&a, F &&f) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    nda::for_each(a.shape(), [&a, &f](auto &&...x) {
      if constexpr (clef::is_function<std::decay_t<decltype(f(x...))>>) {
        clef_auto_assign(a(x...), f(x...));
      } else {
        a(x...) = f(x...);
      }
    });
  }

} // namespace nda
