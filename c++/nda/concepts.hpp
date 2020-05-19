#pragma once
#include "traits.hpp"

namespace nda {

  // --------------------------- Ndarray concept------------------------

  /// A trait to mark classes modeling the Ndarray concept
  template <typename T>
  inline constexpr bool is_ndarray_v = false;

  // --------------------------- concept : is_assign_rhs------------------------
  // Mark classes which are NOT nd_array but have :
  // .shape()
  // can be put at the RHS of assignment or used in construction of array
  /// A trait to mark classes modeling the Ndarray concept
  template <typename T>
  inline constexpr bool is_assign_rhs = false;

#if __cplusplus > 201703L

  template <typename T>
  constexpr bool is_std__array_of_long_v = false;
  template <auto R>
  constexpr bool is_std__array_of_long_v<std::array<long, R>> = true;

  template <class T>
  concept IsStdArrayOfLong = is_std__array_of_long_v<T>;

  template <class From, class To>
  concept convertible_to = std::is_convertible_v<From, To> &&requires(std::add_rvalue_reference_t<From> (&f)()) {
    static_cast<To>(f());
  };

  // clang-format off
  template <typename A> concept Array= requires(A const &a) {

  // A has a shape() which returns an array<long, R> ...
  { a.shape() } -> IsStdArrayOfLong;
  // { a.shape() } -> same_as<std::array<long, get_rank<A>>>; //IsStdArrayOfLong;

  // and R is an int, and is the rank.
  { get_rank<A> } ->convertible_to<const int>;

  // a(0,0,0,0... R times) returns something, which is value_type by definition
  {get_first_element(a)};
  };
  // clang-format on

  template <typename A, int R>
  concept ArrayOfRank = Array<A> and (get_rank<A> == R);

  // clang-format on

#endif

} // namespace nda
