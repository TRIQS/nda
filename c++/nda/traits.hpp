#pragma once

namespace nda {

  /* // --------------------------- is_instantiation_of ------------------------*/

  /**
   * is_instantiation_of_v
   * Checks that X is a T<....>
   */
  //template <typename T, template <typename..., auto ...> class TMPLT>
  //inline constexpr bool is_instantiation_of_v = false;

  //template <template <typename..., auto ...> class TMPLT, typename... U, auto ... X>
  //inline constexpr bool is_instantiation_of_v<TMPLT<U..., X...>, TMPLT> = true;

  // --------------------------- is_complex ------------------------

  template <typename T>
  struct _is_complex : std::false_type {};

  template <typename T>
  struct _is_complex<std::complex<T>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_complex_v = _is_complex<std::decay_t<T>>::value;

  // --------------------------- is_scalar ------------------------

  template <typename S>
  inline constexpr bool is_scalar_v = std::is_arithmetic_v<S> or nda::is_complex_v<S>; // painful without the decay in later code

  template <typename S>
  inline constexpr bool is_scalar_or_convertible_v = is_scalar_v<S> or std::is_constructible_v<std::complex<double>, S>;

  template <typename S, typename A>
  inline constexpr bool is_scalar_for_v = (is_scalar_v<typename A::value_t> ? is_scalar_or_convertible_v<S> : std::is_same_v<S, typename A::value_t>);

  // ---------------------------  is_regular------------------------

  // Impl. trait to match the containers in requires. Match all regular containers (array, matrix)
  template <typename A>
  inline constexpr bool is_regular_or_view_v = false;

  // ---------------------------  is_regular_or_view_v------------------------

  // Impl. trait to match the containers in requires. Match all containers (array, matrix, view)
  template <typename A>
  inline constexpr bool is_regular_v = false;

  // --------------------------- Ndarray concept------------------------

  /// A trait to mark classes modeling the Ndarray concept
  template <typename T>
  inline constexpr bool is_ndarray_v = false;

  //template <typename T>
  //inline constexpr bool is_2d_ndarray_v = is_ndarray_v<T> and ((get_rank<T>) == 2);

  // --------------------------- get_rank ------------------------

  /// A trait to get the rank of an object with ndarray concept
  template <typename A>
  constexpr int get_rank = std::tuple_size_v<std::decay_t<decltype(std::declval<A const>().shape())>>;

  // --------------------------- get_first_element and get_value_t ------------------------

  // FIXME C++20 lambda
  template <size_t... Is, typename A>
  auto _get_first_element_impl(std::index_sequence<Is...>, A const &a) {
    return a((0 * Is)...); // repeat 0 sizeof...(Is) times
  }

  /// Get the first element of the array as a(0,0,0....) (i.e. also work for non containers, just with the concept !).
  template <typename A>
  auto get_first_element(A const &a) {
    return _get_first_element_impl(std::make_index_sequence<get_rank<A>>{}, a);
  }

  /// A trait to get the return_t of the (long, ... long) for an object with ndarray concept
  template <typename A>
  using get_value_t = decltype(get_first_element(std::declval<A const>()));

  // --------------------------- Algebra ------------------------

  /// A trait to mark a class for its algebra : 'N' = None, 'A' = array, 'M' = matrix, 'V' = vector
  template <typename A>
  inline constexpr char get_algebra = 'N';

  // ---------------------- Guarantees at compile time for some optimization  --------------------------------

  enum class layout_info_e : uint64_t {
    none                   = 0x0,
    strided_1d             = 0x1,
    smallest_stride_is_one = 0x2,
    contiguous             = strided_1d | smallest_stride_is_one
  };

  template <typename A>
  inline constexpr layout_info_e get_layout_info = layout_info_e::none;

  // & operator for the layout_info_e
  constexpr layout_info_e operator|(layout_info_e a, layout_info_e b) { return layout_info_e(uint64_t(a) | uint64_t(b)); }
  constexpr bool operator&(layout_info_e a, layout_info_e b) { return uint64_t(a) & uint64_t(b); }
  //bool operator|=(layout_info_e & a, layout_info_e b) { return a = layout_info_e(uint64_t(a) | uint64_t(b));}

  template <typename A>
  constexpr bool has_layout_contiguous = (get_layout_info<A> & layout_info_e::contiguous);
  template <typename A>
  constexpr bool has_layout_strided_1d = (get_layout_info<A> & layout_info_e::strided_1d);
  template <typename A>
  constexpr bool has_layout_smallest_stride_is_one = (get_layout_info<A> & layout_info_e::smallest_stride_is_one);

  // ---------------------- linear index  --------------------------------

  // A small vehicule for the linear index for optimized case
  struct _linear_index_t {
    long value;
  };

} // namespace nda
