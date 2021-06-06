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

// -------   IsStdArrayOfLong   ----------
// true iif T is a std::array<long, Rank>

namespace details {
  template <typename T>
  constexpr bool is_std_array_of_long_v = false;
  template <auto R>
  constexpr bool is_std_array_of_long_v<std::array<long, R>> = true;
} // namespace details

template <class T>
concept IsStdArrayOfLong = details::is_std_array_of_long_v<std::decay_t<T>>;

// -------   Array   ----------
// main concept of the library

template <typename A>
concept Array = requires(A const &a) {

  // A has a shape() which returns an array<long, R>
  // its length is the rank, as deduced by get_rank
  { a.shape() } -> IsStdArrayOfLong;

  // a(0,0,0,0... R times) returns something, which is of type value_type by definition
  requires CallableWithLongs<A, get_rank<A>>;
};

// -------   ArrayOfRank   ----------
// An array of rank R

template <typename A, int R>
concept ArrayOfRank = Array<A> and(get_rank<A> == R);

//---------ArrayInitializer  ----------
// The concept of what can be used to init an array
// it must have
// - a shape (to init the array
// - a value_type
// - be invokable on a view to init the array

template <typename A>
concept ArrayInitializer = requires(A const &a) {

  { a.shape() } -> IsStdArrayOfLong;

  typename A::value_type;

  // FIXME not perfect, it should accept any layout ??
  {a.invoke(array_contiguous_view<typename A::value_type, get_rank<A>>{})};
};

//---------HasValueTypeConstructibleFrom  ----------
// FIXME : We should not need this ... Only used once...

template <typename A, typename U>
concept HasValueTypeConstructibleFrom = Array<A> and(std::is_constructible_v<U, get_value_t<A>>);

} // namespace nda
