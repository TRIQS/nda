#pragma once

// FIXME : to be remove. use is_instance_of
#include "./concepts.hpp"

namespace nda {

  template <typename T> struct _is_complex : std::false_type {};
  template <typename T> struct _is_complex<std::complex<T>> : std::true_type {};

  template <typename T> inline constexpr bool is_complex_v = _is_complex<std::decay_t<T>>::value;

  // FIXME : decide for decay.

  template <typename S>
  inline constexpr bool is_scalar_v =
     std::is_arithmetic_v<std::decay_t<S>> or nda::is_complex_v<std::decay_t<S>>; // painful without the decay in later code

  template <typename S>
  inline constexpr bool is_scalar_or_convertible_v = is_scalar_v<S> or std::is_constructible_v<std::complex<double>, std::decay_t<S>>;

  template <typename S, typename A>
  inline constexpr bool is_scalar_for_v = (is_scalar_v<typename A::value_t> ? is_scalar_or_convertible_v<S> : std::is_same_v<S, typename A::value_t>);

  // get the rank of an object with ndarray concept
  template <typename A> constexpr int get_rank = std::tuple_size_v<std::decay_t<decltype(std::declval<A const>().shape())>>;

  // FIXME C++20 lambda
  template <size_t... Is, typename A> auto _get_value_t_impl(std::index_sequence<Is...>, A a) {
    return a((0*Is)...); // repeat 0 sizeof...(Is) times
  }

  template <typename A> using get_value_t = decltype(_get_value_t_impl(std::make_index_sequence<get_rank<A>>(), std::declval<A const>()));

  // The Ndarray concept
  template <typename T> inline constexpr bool is_ndarray_v = false;

  // general make_regular
  template <typename A> typename A::regular_t make_regular(A &&x) REQUIRES(is_ndarray_v<A>) { return std::forward<A>(x); }
  //template <typename A> regular_t<A> make_regular(A &&x) REQUIRES(is_ndarray_v<A>) { return std::forward<A>(x); }


  
  // Algebra
  // A trait to mark a class for its algebra : 'N' = None, 'A' = array, 'M' = matrix, 'V' = vector
  template <typename A> inline constexpr char get_algebra = 'N';



} // namespace nda
