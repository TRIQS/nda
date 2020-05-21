#pragma once
#include "declarations.hpp"

namespace nda {

#if __cplusplus > 201703L

  namespace concepts { // implementation details of concepts. Main concepts in nda::

    // FIXME : replace get_first_element with call_on_R_zeros ? 
    // It is crucial to use here a version of get_first_element with an explicitly deduced return type
    // to have SFINAE, otherwise the concept checking below will fail
    // in the case of A has a shape but NO operator(), like an initializer e.g. mpi_lazy_xxx
    template <size_t... Is, typename A>
    auto call_on_R_zeros_impl(std::index_sequence<Is...>, A const &a) -> decltype(a((0 * Is)...)) {
      return a((0 * Is)...); // repeat 0 sizeof...(Is) times
    }

    template <typename A>
    auto call_on_R_zeros(A const &a) -> decltype(call_on_R_zeros_impl(std::make_index_sequence<get_rank<A>>{}, a)) {
      return call_on_R_zeros_impl(std::make_index_sequence<get_rank<A>>{}, a);
    }

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

  } // namespace concepts

  // clang-format off
  template <typename A> concept Array= requires(A const &a) {

  // A has a shape() which returns an array<long, R> ...
  { a.shape() } -> concepts::IsStdArrayOfLong;

  // and R is an int, and is the rank.
  { get_rank<A> } ->concepts::convertible_to<const int>;

  // a(0,0,0,0... R times) returns something, which is value_type by definition
  {concepts::call_on_R_zeros(a)};
  };

  //-------------------

  template <typename A, int R>
  concept ArrayOfRank = Array<A> and (get_rank<A> == R);


  //-------------------

  template <typename A> concept ArrayInitializer = requires(A const &a) {

  // A has a shape() which returns an array<long, R> ...
  { a.shape() } -> concepts::IsStdArrayOfLong;

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
