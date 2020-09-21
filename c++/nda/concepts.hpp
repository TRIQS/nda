// Copyright (c) 2020 Simons Foundation
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
#include "declarations.hpp"
#include "stdutil/concepts.hpp"

#if __cplusplus > 201703L

namespace nda {

  namespace concept_impl { // implementation details of concepts. Main concepts in nda::

    // -----------------------

    // It is crucial to use here a version of get_first_element with an explicitly deduced return type
    // to have SFINAE, otherwise the concept checking below will fail
    // in the case of A has a shape but NO operator(), like an initializer e.g. mpi_lazy_xxx
    template <size_t... Is, typename A>
    auto call_on_R_zeros_impl(std::index_sequence<Is...>, A const &a) -> decltype(a((0 * Is)...)) {
      return a((0 * Is)...); // repeat 0 sizeof...(Is) times
    }

    template <int R, typename A>
    auto call_on_R_zeros(A const &a) -> decltype(call_on_R_zeros_impl(std::make_index_sequence<R>{}, a)) {
      return call_on_R_zeros_impl(std::make_index_sequence<R>{}, a);
    }

    template <typename T>
    constexpr bool is_std__array_of_long_v = false;
    template <auto R>
    constexpr bool is_std__array_of_long_v<std::array<long, R>> = true;

    template <class T>
    concept IsStdArrayOfLong = is_std__array_of_long_v<std::decay_t<T>>;

    // used in array_adapter
    // clang-format off
    template <int R, typename A> concept CallableWithRLongs= requires(A const &a) {
      {call_on_R_zeros<R>(a)};
    };
    // clang-format on

  } // namespace concept_impl

  // clang-format off
  
   // FIXME : C++20 replace by  std::equality_comparable_with with <concepts> in libc++
   /// A simplified version of std::equality_comparable_with (temporary fix waiting to implementation of <concepts> in libc++
    template<class T, class U>
    concept StdEqualityComparableWith= // exposition only
    requires(const std::remove_reference_t<T>& t,
             const std::remove_reference_t<U>& u) {
      { t == u } -> std::convertible_to<bool>;
      { t != u } ->std::convertible_to<bool> ;
      { u == t } -> std::convertible_to<bool>;
      { u != t } -> std::convertible_to<bool>;
    };
  
    template <typename A> concept Array= requires(A const &a) {

  // A has a shape() which returns an array<long, R> ...
  { a.shape() } -> concept_impl::IsStdArrayOfLong;

  // and R is an int, and is the rank.
  { get_rank<A> } ->std::convertible_to<const int>;

  // a(0,0,0,0... R times) returns something, which is value_type by definition
  {concept_impl::call_on_R_zeros<get_rank<A>>(a)};
  };

  //-------------------

  template <typename A, int R>
  concept ArrayOfRank = Array<A> and (get_rank<A> == R);


  template <typename A, typename U>
  concept HasValueTypeConvertibleTo = Array<A> and (std::is_convertible_v<get_value_t<A>, U>);

  //-------------------

  template <typename A> concept ArrayInitializer = requires(A const &a) {

  // A has a shape() which returns an array<long, R> ...
  { a.shape() } -> concept_impl::IsStdArrayOfLong;

  typename A::value_type; 

  // not perfect, it should accept any layout
  {a.invoke(array_contiguous_view<typename A::value_type, get_rank<A>>{}) };

  };
  // clang-format on

} // namespace nda

// end if C++20
#endif

// C++17 backward compat workaround
// FIXME : remove in C++20

namespace nda {

  /// A trait to mark classes modeling the Ndarray concept
  template <typename T>
  inline constexpr bool is_ndarray_v = false;

  // ---------------------- Mark containers --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr bool is_ndarray_v<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = true;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr bool is_ndarray_v<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = true;

#if not(__cplusplus > 201703L)

  // --------------------------- ArrayInitializer
  template <typename T>
  inline constexpr bool is_array_initializer_v = false;

  // --------------------------- Check if a type is an array of given rank
  template <typename T, auto Rank>
  inline constexpr bool has_rank = []() {
    if constexpr (is_ndarray_v<T>) {
      return Rank == get_rank<T>;
    } else
      return false;
  }();

  // --------------------------- Workaround concept
  template <typename A, typename U>
  inline constexpr bool has_value_type_convertible_to = []() {
    if constexpr (is_ndarray_v<A>) {
      return std::is_convertible_v<get_value_t<A>, U>;
    } else
      return false;
  }();

#endif

} // namespace nda
