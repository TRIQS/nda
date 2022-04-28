// Copyright (c) 2019-2022 Simons Foundation
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
#include <complex>
#include <type_traits>
#include <utility>

// A few addons to the std::...
#include "stdutil/complex.hpp"
#include "stdutil/array.hpp"

namespace nda {

  // Using 2i and co
  using namespace std::literals::complex_literals;

  // --------------------------- is_instantiation_of ------------------------

  template <template <typename...> class TMPLT, typename T>
  struct is_instantiation_of : std::false_type {};
  template <template <typename...> class TMPLT, typename... U>
  struct is_instantiation_of<TMPLT, TMPLT<U...>> : std::true_type {};
  template <template <typename...> class TMPLT, typename T>
  inline constexpr bool is_instantiation_of_v = is_instantiation_of<TMPLT, std::remove_cvref_t<T>>::value;

  // --------------------------- For error messages ------------------------

  // to prevent the static_assert to trigger only when instantiated
  // in a constexpr branch of a template function, use fail<T>
  // static_assert(0) would never compile.
  // fail also prints the type T in question
  template <typename... T>
  static constexpr bool always_true = true;

  template <typename... T>
  static constexpr bool always_false = false;

  template <typename... T>
  static constexpr bool with_Args = false;

  template <typename T>
  static constexpr bool with_Array = false;

  template <int R>
  static constexpr bool with_Rank = false;

  // --------------------------- is_complex ------------------------

  template <typename T>
  inline constexpr bool is_complex_v = is_instantiation_of_v<std::complex, T>;

  // --------------------------- is_scalar ------------------------

  template <typename S>
  inline constexpr bool is_scalar_v = std::is_arithmetic_v<std::remove_cvref_t<S>> or nda::is_complex_v<S>;

  template <typename S>
  inline constexpr bool is_scalar_or_convertible_v = is_scalar_v<S> or std::is_constructible_v<std::complex<double>, S>;

  template <typename S, typename A>
  inline constexpr bool is_scalar_for_v = (is_scalar_v<typename A::value_type> ? is_scalar_or_convertible_v<S> :
                                                                                 std::is_same_v<S, typename A::value_type>);

  template <typename T>
  inline constexpr bool is_double_or_complex_v = is_complex_v<T> or std::is_same_v<double, std::remove_cvref_t<T>>;

  template <typename T>
  inline constexpr bool is_blas_lapack_v = is_double_or_complex_v<T>;

  // --------------------------- Algebra ------------------------

  /// A trait to mark a class for its algebra : 'N' = None, 'A' = array, 'M' = matrix, 'V' = vector
  template <typename A>
  inline constexpr char get_algebra = 'N';

  // --------------------------- get_rank ------------------------

  /// A trait to get the rank of an object with ndarray concept
  template <typename A>
  constexpr int get_rank = std::tuple_size_v<std::remove_cvref_t<decltype(std::declval<A const>().shape())>>;

  // ---------------------------  is_regular------------------------

  // Impl. trait to match the containers in requires. Match all regular containers (array, matrix)
  template <typename A>
  inline constexpr bool is_regular_v = false;

  // ---------------------------  is_view_v------------------------

  // Impl. trait to match the containers in requires. Match all containers (array, matrix, view)
  template <typename A>
  inline constexpr bool is_view_v = false;

  // ---------------------------  is_regular_or_view_v------------------------

  // Impl. trait to match the containers in requires. Match all containers (array, matrix, view)
  template <typename A>
  inline constexpr bool is_regular_or_view_v = is_regular_v<A> or is_view_v<A>;

  // ---------------------------  is_matrix_or_view_v------------------------

  // Impl. trait to match the containers in requires. Match all containers (array, matrix, view)
  template <typename A>
  inline constexpr bool is_matrix_or_view_v = is_regular_or_view_v<A> and (get_algebra<A> == 'M') and (get_rank<A> == 2);

  // --------------------------- get_first_element and get_value_t ------------------------

  /// Get the first element of the array as a(0,0,0....) (i.e. also work for non
  /// containers, just with the concept !).
  template <typename A>
  decltype(auto) get_first_element(A const &a) {
    return [&a]<auto... Is>(std::index_sequence<Is...>)->decltype(auto) {
      return a((0 * Is)...); // repeat 0 sizeof...(Is) times
    }
    (std::make_index_sequence<get_rank<A>>{});
  }
  // decltype(auto) and not auto to work in case that a(....) is NOT copy constructible

  /// A trait to get the return_t of the (long, ... long) for an object with ndarray concept
  template <typename A>
  using get_value_t = std::decay_t<decltype(get_first_element(std::declval<A const>()))>;

  // Check all A have the same element_type
  template <typename A0, typename... A>
  inline constexpr bool have_same_value_type_v = (std::is_same_v<get_value_t<A0>, get_value_t<A>> and ... and true);

  // ---------------------- Guarantees at compile time for some optimization  --------------------------------

  enum class layout_prop_e : uint64_t {
    none                   = 0x0,
    strided_1d             = 0x1,
    smallest_stride_is_one = 0x2,
    contiguous             = strided_1d | smallest_stride_is_one
  };

  // Whether we can degrade the property. It is a partial order.
  // contiguous -> strided_1d  -> none
  // contiguous -> smallest_stride_is_one  -> none
  /// \private
  inline constexpr bool layout_property_compatible(layout_prop_e from, layout_prop_e to) {
    if (from == layout_prop_e::contiguous) return true;
    if (from == layout_prop_e::none) return (to == layout_prop_e::none);
    return ((to == layout_prop_e::none) or (to == from));
  }

  //  operator for the layout_prop_e
  constexpr layout_prop_e operator|(layout_prop_e a, layout_prop_e b) { return layout_prop_e(uint64_t(a) | uint64_t(b)); }

  constexpr layout_prop_e operator&(layout_prop_e a, layout_prop_e b) { return layout_prop_e{uint64_t(a) & uint64_t(b)}; }
  //bool operator|=(layout_prop_e & a, layout_prop_e b) { return a = layout_prop_e(uint64_t(a) | uint64_t(b));}

  inline constexpr bool has_strided_1d(layout_prop_e x) { return uint64_t(x) & uint64_t(layout_prop_e::strided_1d); }
  inline constexpr bool has_smallest_stride_is_one(layout_prop_e x) { return uint64_t(x) & uint64_t(layout_prop_e::smallest_stride_is_one); }
  inline constexpr bool has_contiguous(layout_prop_e x) { return has_strided_1d(x) and has_smallest_stride_is_one(x); }

  // FIXME : I need a NONE for stride_order. For the scalars ...
  struct layout_info_t {
    uint64_t stride_order = 0;
    layout_prop_e prop    = layout_prop_e::none;
  };

  // Combining layout_info
  constexpr layout_info_t operator&(layout_info_t a, layout_info_t b) {
    if (a.stride_order == b.stride_order)
      return layout_info_t{a.stride_order, layout_prop_e(uint64_t(a.prop) & uint64_t(b.prop))};
    else
      return layout_info_t{uint64_t(-1), layout_prop_e::none}; // -1 is undefined stride_order, it corresponds to no permutation
  }

  template <typename A>
  inline constexpr layout_info_t get_layout_info = layout_info_t{};

  template <typename A>
  constexpr bool has_contiguous_layout = (has_contiguous(get_layout_info<std::decay_t<A>>.prop));
  template <typename A>
  constexpr bool has_layout_strided_1d = (has_strided_1d(get_layout_info<std::decay_t<A>>.prop));
  template <typename A>
  constexpr bool has_layout_smallest_stride_is_one = (has_smallest_stride_is_one(get_layout_info<std::decay_t<A>>.prop));

  // ---------------------- linear index  --------------------------------

  // A small vehicule for the linear index for optimized case
  struct _linear_index_t {
    long value;
  };

  // debug tool
  template <typename T>
  [[deprecated]] void show_the_type(T &&) {}

} // namespace nda
