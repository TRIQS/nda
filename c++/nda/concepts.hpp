#pragma once
#include "declarations.hpp"

namespace nda {

#if __cplusplus > 201703L

  template <typename T>
  constexpr bool is_std__array_of_long_v = false;
  template <auto R>
  constexpr bool is_std__array_of_long_v<std::array<long, R>> = true;

  template <class T>
  concept IsStdArrayOfLong = is_std__array_of_long_v<T>;

  template <class From, class To>
  concept convertible_to = std::is_convertible_v<From, To> and requires(std::add_rvalue_reference_t<From> (&f)()) {
    static_cast<To>(f());
  };

  // clang-format off
  template <typename A> concept Array= requires(A const &a) {

  // A has a shape() which returns an array<long, R> ...
  { a.shape() } -> IsStdArrayOfLong;

  // and R is an int, and is the rank.
  { get_rank<A> } ->convertible_to<const int>;

  // a(0,0,0,0... R times) returns something, which is value_type by definition
  {get_first_element(a)};
  };

  //-------------------

  template <typename A, int R>
  concept ArrayOfRank = Array<A> and (get_rank<A> == R);


  //-------------------

  template <typename A> concept ArrayInitializer = requires(A const &a) {

  // A has a shape() which returns an array<long, R> ...
  { a.shape() } -> IsStdArrayOfLong;

  typename A::value_type; 

  // not perfect, it should accept any layout
  {a.invoke(array_view<typename A::value_type, get_rank<A>>{}) };

  };
    // clang-format on

#endif

  // C++17 backward compat workaround

  // --------------------------- Array

  /// A trait to mark classes modeling the Ndarray concept
  template <typename T>
  inline constexpr bool is_ndarray_v = false;

  // ---------------------- Mark containers --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr bool is_ndarray_v<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = true;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr bool is_ndarray_v<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = true;

#if not __cplusplus > 201703L

  // --------------------------- ArrayInitializer
  template <typename T>
  inline constexpr bool is_array_initializer_v = false;

#endif

} // namespace nda
