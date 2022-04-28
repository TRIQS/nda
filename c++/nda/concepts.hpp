// Copyright (c) 2020-2022 Simons Foundation
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
#include "traits.hpp"
#include "stdutil/concepts.hpp"

#include <type_traits>

namespace nda {

  // -------   CallableWithLongs<A, R>   ----------
  // A can be called with R longs

#ifndef __clang__
  // clang does not have an implementation yet of "Lambda in unevaluated context"
  // we make a workaround below
  // clang-format off
  template <typename A, int R>
  concept CallableWithLongs = requires(A const &a) {
   {
     []<auto... Is>(std::index_sequence<Is...>, auto const &aa) -> decltype(aa(long(Is)...)) {return aa(long(Is)...);} // a lambda to call aa(Is...)
     (std::make_index_sequence<R>{}, a) // apply it
   }; 
   // the expression a(long(Is)...) should compile, with a pack of R longs
   // the concept says nothing about the return type of this function
   // NB decltype is crucial for the concept to work, to use SFINAE. 
   // if not present, the concept will fail to compile for an A for which the a(Is...) it not well formed.
  };
// clang-format on
#else
  namespace details {
    template <auto... Is, typename A>
    auto call_on_R_longs(std::index_sequence<Is...>, A const &a) -> decltype(a((long{Is})...)); // no impl needed
  }

  template <typename A, int R>
  concept CallableWithLongs = requires(A const &a) {
    {details::call_on_R_longs(std::make_index_sequence<R>{}, a)};
  };

#endif

// -------   StdArrayOfLong   ----------
// true iif T is a std::array<long, Rank>

namespace details {
  template <typename T>
  constexpr bool is_std_array_of_long_v = false;
  template <auto R>
  constexpr bool is_std_array_of_long_v<std::array<long, R>> = true;
} // namespace details

/// Check if T is an 'std::array' of long
template <class T>
concept StdArrayOfLong = details::is_std_array_of_long_v<std::decay_t<T>>;

// -------   Scalar   ----------

/// Check if S is an arthmetic or complex type
template <typename S>
concept Scalar = nda::is_scalar_v<S>;

/// Check if S is either double of complex
template <typename S>
concept DoubleOrComplex = nda::is_double_or_complex_v<S>;

// -------   InstantiationOf   ----------

/// Check if T is an instantiation of the `template <typename...> class TMPLT`
template <typename T, template <typename...> class TMPLT>
concept InstantiationOf = nda::is_instantiation_of_v<TMPLT, T>;

namespace mem {

  struct blk_t;
  enum class AddressSpace;
  /// Concept of an Allocator
  template <typename A>
  concept Allocator = requires(A & a) {
    { a.allocate(size_t{}) }
    noexcept->std::same_as<blk_t>;
    { a.allocate_zero(size_t{}) }
    noexcept->std::same_as<blk_t>;
    { a.deallocate(std::declval<blk_t>()) }
    noexcept;
    { A::address_space } -> std::same_as<AddressSpace const &>;
  };

  /// Concept of a handle on a block of memory
  template <typename H, typename T = typename std::remove_cvref_t<H>::value_type>
  concept Handle = requires(H const &h) {
    requires std::is_same_v<typename std::remove_cvref_t<H>::value_type, T>;
    { h.is_null() }
    noexcept->std::same_as<bool>;
    { h.data() }
    noexcept->std::same_as<T *>;
    { H::address_space } -> std::same_as<AddressSpace const &>;
  };

  /// Concept of a handle that owns a block of memory
  template <typename H, typename T = typename std::remove_cvref_t<H>::value_type>
  concept OwningHandle = Handle<H, T> and requires(H const &h) {
    requires not std::is_const_v<typename std::remove_cvref_t<H>::value_type>;
    { h.size() }
    noexcept->std::same_as<long>;
  };
} // namespace mem

// -------   Array   ----------

/// Check if A has a shape, size and rank R and can be called with R integers.
template <typename A>
concept Array = requires(A const &a) {

  // A has a shape() which returns an array<long, R>
  // its length is the rank, as deduced by get_rank
  { a.shape() } -> StdArrayOfLong;

  // Contract: Expects the size to be equal to the product of the elements in shape()
  { a.size() } -> std::same_as<long>;

  // a(0,0,0,0... R times) returns something, which is of type value_type by definition
  requires CallableWithLongs<A, get_rank<A>>;
};

/// Check if A is an Array and exposes its `data()` and its `indexmap().strides()`
template <typename A, typename A_t = std::remove_cvref_t<A>>
concept MemoryArray = Array<A> && requires(A &a) {

  // Has a storage_t that is a memory handle
  typename A_t::storage_t;
  mem::Handle<typename A_t::storage_t>;

  // There is a member-type value_type that maybe const
  typename A_t::value_type;

  // We can acquire the pointer to the underlying data
  {
    a.data()
    } -> std::same_as<std::conditional_t<std::is_const_v<A> || std::is_const_v<typename A_t::value_type>, const get_value_t<A>, get_value_t<A>> *>;

  // Exposes the memory stride for each dimension
  { a.indexmap().strides() } -> StdArrayOfLong;
};

// -------   Additional Array Concepts   ----------

/// Check if A is a Array with of rank R
template <typename A, int R>
concept ArrayOfRank = Array<A> and(get_rank<A> == R);

/// Check if A is a MemoryArray with a specific rank R
template <typename A, int R>
concept MemoryArrayOfRank = MemoryArray<A> and(get_rank<A> == R);

/// Check if A is a Array or Scalar
template <typename AS>
concept ArrayOrScalar = Array<AS> or Scalar<AS>;

/// Check if M is a Matrix, i.e. ArrayOfRank<2>
template <typename M>
concept Matrix = ArrayOfRank<M, 2>;

/// Check if M is a Vector, i.e. ArrayOfRank<1>
template <typename V>
concept Vector = ArrayOfRank<V, 1>;

/// Check if M is a MemoryMatrix, i.e. MemoryArrayOfRank<2>
template <typename M>
concept MemoryMatrix = MemoryArrayOfRank<M, 2>;

/// Check if V is a MemoryVector, i.e. MemoryArrayOfRank<1>
template <typename V>
concept MemoryVector = MemoryArrayOfRank<V, 1>;

//---------ArrayInitializer  ----------
// The concept of what can be used to init an array
// it must have
// - a shape (to init the array
// - a value_type
// - be invokable on a view to init the array

///
template <typename A, typename B>
concept ArrayInitializer = requires(A const &a) {

  { a.shape() } -> StdArrayOfLong;

  typename std::remove_cvref_t<A>::value_type;

  // FIXME not perfect, it should accept any layout ??
  requires MemoryArray<B> && requires(B & b) { a.invoke(b); };
};

//---------HasValueTypeConstructibleFrom  ----------
// FIXME : We should not need this ... Only used once...

template <typename A, typename U>
concept HasValueTypeConstructibleFrom = Array<A> and(std::is_constructible_v<U, get_value_t<A>>);

} // namespace nda
