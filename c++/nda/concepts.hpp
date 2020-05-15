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
  concept is_std__array_of_long_c = is_std__array_of_long_v<T>;

  template <class From, class To>
  concept convertible_to = std::is_convertible_v<From, To> &&requires(std::add_rvalue_reference_t<From> (&f)()) {
    static_cast<To>(f());
  };

  // clang-format off
  template <typename A> concept ArrayAnyRank =  requires(A const &a) {

    // A has a shape() which returns an array<long, R> ...
    { a.shape() } ->is_std__array_of_long_c;
 
    // and R is an int, and is the rank. Not strictly necessary, but will be used in Array
    { get_rank<A> } ->convertible_to<const int>;

    // a(0,0,0,0... R times) returns something, which is value_type by definition
    {get_first_element(a)};
  };

  template <typename A, int R=-1> concept Array = ArrayAnyRank<A> and ((R==-1) or (get_rank<A> ==R));

  // clang-format on

#define NDA_REQUIRES17(...)
#define NDA_REQUIRES20(...) requires(__VA_ARGS__)
#define NDA_REQUIRES(...) requires(__VA_ARGS__)

#else

#define NDA_REQUIRES20(...)

#ifdef __clang__
#define NDA_REQUIRES17(...) __attribute__((enable_if(__VA_ARGS__, AS_STRING(__VA_ARGS__))))
#define NDA_REQUIRES(...) __attribute__((enable_if(__VA_ARGS__, AS_STRING(__VA_ARGS__))))
#elif __GNUC__
#define NDA_REQUIRES17(...) requires(__VA_ARGS__)
#define NDA_REQUIRES(...) requires(__VA_ARGS__)
#endif

#endif

} // namespace nda
