// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides concepts for the nda library.
 */

#pragma once

#include "./stdutil/concepts.hpp"
#include "./traits.hpp"

#include <array>
#include <concepts>
#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @addtogroup utils_concepts
   * @{
   */

  // clang has full support for "lambdas in unevaluated context" only for versions >= 17.
#ifndef __clang__

  // clang-format off
  /**
   * @brief Check if a given type can be called with a certain number of long arguments.
   *
   * @details An example of a type satisfying this concept is e.g. nda::basic_array or nda::basic_array_view. More
   * generally, every type modelling the nda::Array concept has to be nda::CallableWithLongs as well.
   *
   * @tparam A Type to check.
   * @tparam R Number of long arguments.
   */
  template <typename A, int R>
  concept CallableWithLongs = requires(A const &a) {
    // if decltype is not present, the concept will fail to compile for an A for which the a(Is...) is not well formed
    []<auto... Is>(std::index_sequence<Is...>, auto const &aa) -> decltype(aa(long(Is)...)) {return aa(long(Is)...);}
    (std::make_index_sequence<R>{}, a);
  };
  // clang-format on

#else

  namespace detail {

    // Helper function declaration to check if A can be called with R long arguments.
    template <auto... Is, typename A>
    auto call_on_R_longs(std::index_sequence<Is...>, A const &a) -> decltype(a((long{Is})...)); // no impl needed

  } // namespace detail

  /**
   * @brief Check if a given type can be called with a certain number of long arguments.
   *
   * @details An example of a type satisfying this concept is e.g. nda::basic_array or nda::basic_array_view. More
   * generally, every type modelling the nda::Array concept has to be nda::CallableWithLongs as well.
   *
   * @tparam A Type to check.
   * @tparam R Number of long arguments.
   */
  template <typename A, int R>
  concept CallableWithLongs = requires(A const &a) {
    { detail::call_on_R_longs(std::make_index_sequence<R>{}, a) };
  };

#endif // __clang__

  namespace detail {

    // Check if T is of type std::array<long, R> for arbitrary array sizes R.
    template <typename T>
    constexpr bool is_std_array_of_long_v = false;

    // Specialization of nda::detail::is_std_array_of_long_v for cvref types.
    template <typename T>
      requires(!std::is_same_v<T, std::remove_cvref_t<T>>)
    constexpr bool is_std_array_of_long_v<T> = is_std_array_of_long_v<std::remove_cvref_t<T>>;

    // Specialization of nda::detail::is_std_array_of_long_v for std::array<long, R>.
    template <auto R>
    constexpr bool is_std_array_of_long_v<std::array<long, R>> = true;

  } // namespace detail

  /**
   * @brief Check if a given type is of type `std::array<long, R>` for some arbitrary `R`.
   *
   * @details The shape and the strides of every nda::MemoryArray type is represented as a `std::array<long, R>`, where
   * `R` is the rank of the array.
   *
   * @tparam T Type to check.
   */
  template <class T>
  concept StdArrayOfLong = detail::is_std_array_of_long_v<T>;

  /**
   * @brief Check if a given type is either an arithmetic or complex type.
   * @tparam S Type to check.
   */
  template <typename S>
  concept Scalar = nda::is_scalar_v<S>;

  /**
   * @brief Check if a given type is either a double or complex type.
   * @tparam S Type to check.
   */
  template <typename S>
  concept DoubleOrComplex = nda::is_double_or_complex_v<S>;

  /**
   * @brief Check if a given type is an instantiation of some other template type.
   *
   * @details See nda::is_instantiation_of for more information.
   *
   * @tparam T Type to check.
   * @tparam TMPLT Template template type to check against.
   */
  template <typename T, template <typename...> class TMPLT>
  concept InstantiationOf = nda::is_instantiation_of_v<TMPLT, T>;

  /**
   * @brief True iif T is same_as any of the Us
   *
   * @details See nda::is_any_of for implementation.
   *
   * @tparam T Type to check
   * @tparam Us Types to check against
   */
  template <typename T, typename... Us>
  concept AnyOf = is_any_of<T, Us...>;

  /** @} */

  namespace mem {

    /**
     * @addtogroup mem_utils
     * @{
     */

    /// @cond
    // Forward declarations.
    struct blk_t;
    enum class AddressSpace;
    /// @endcond

    /**
     * @brief Check if a given type satisfies the allocator concept.
     *
     * @details Allocators are used to reserve and free memory. Depending on their address space, the memory can either
     * be allocated on the Host (CPU), on the Device (GPU) or on unified memory (see nda::mem::AddressSpace).
     *
     * @note The named <a href="https://en.cppreference.com/w/cpp/named_req/Allocator">C++ Allocator
     * requirements</a> of the standard library are different.
     *
     * @tparam A Type to check.
     */
    template <typename A>
    concept Allocator = requires(A &a) {
      { a.allocate(size_t{}) } noexcept -> std::same_as<blk_t>;
      { a.allocate_zero(size_t{}) } noexcept -> std::same_as<blk_t>;
      { a.deallocate(std::declval<blk_t>()) } noexcept;
      { A::address_space } -> std::same_as<AddressSpace const &>;
    };

    /**
     * @brief Check if a given type satisfies the memory handle concept.
     *
     * @details Memory handles are used to manage memory. They are responsible for providing access to the data which
     * can be located on the stack (CPU), the heap (CPU), the device (GPU) or on unified memory (see
     * nda::mem::AddressSpace).
     *
     * @tparam H Type to check.
     * @tparam T Value type of the handle.
     */
    template <typename H, typename T = typename std::remove_cvref_t<H>::value_type>
    concept Handle = requires(H const &h) {
      requires std::is_same_v<typename std::remove_cvref_t<H>::value_type, T>;
      { h.is_null() } noexcept -> std::same_as<bool>;
      { h.data() } noexcept -> std::same_as<T *>;
      { H::address_space } -> std::same_as<AddressSpace const &>;
    };

    /**
     * @brief Check if a given type satisfies the owning memory handle concept.
     *
     * @details In addition to the requirements of the nda::mem::Handle concept, owning memory handles are aware of the
     * size of the memory they manage and can be used to release the memory.
     *
     * @tparam H Type to check.
     * @tparam T Value type of the handle.
     */
    template <typename H, typename T = typename std::remove_cvref_t<H>::value_type>
    concept OwningHandle = Handle<H, T> and requires(H const &h) {
      requires not std::is_const_v<typename std::remove_cvref_t<H>::value_type>;
      { h.size() } noexcept -> std::same_as<long>;
    };

    /** @} */

  } // namespace mem

  /**
   * @addtogroup av_utils
   * @{
   */

  /**
   * @brief Check if a given type satisfies the array concept.
   *
   * @details An array is a multi-dimensional container of elements. It is characterized by a certain size (number of
   * elements), rank (number of dimensions) and shape (extent of each dimension). Furthermore, it provides access to its
   * elements by overloading the function call operator.
   *
   * Examples of types satisfying this concept are e.g. nda::basic_array or nda::basic_array_view.
   *
   * @note std::array does not satisfy this concept, as it does not have a shape or provides access via the function
   * call operator.
   *
   * @tparam A Type to check.
   */
  template <typename A>
  concept Array = requires(A const &a) {
    { a.shape() } -> StdArrayOfLong;
    { a.size() } -> std::same_as<long>;
    requires CallableWithLongs<A, get_rank<A>>;
  };

  /**
   * @brief Check if a given type satisfies the memory array concept.
   *
   * @details In addition to the requirements of the nda::Array concept, memory arrays
   * provide access to the underlying memory storage and use an nda::idx_map to specify the
   * layout of the data in memory.
   *
   * An example of a type satisfying the nda::Array but not the nda::MemoryArray concept is nda::expr.
   *
   * @tparam A Type to check.
   */
  template <typename A, typename A_t = std::remove_cvref_t<A>>
  concept MemoryArray = Array<A> && requires(A &a) {
    typename A_t::storage_t;
    requires mem::Handle<typename A_t::storage_t>;
    typename A_t::value_type;
    {
      a.data()
    } -> std::same_as<std::conditional_t<std::is_const_v<std::remove_reference_t<A>> || std::is_const_v<typename A_t::value_type>,
                                         const get_value_t<A>, get_value_t<A>> *>;
    { a.indexmap().strides() } -> StdArrayOfLong;
  };

  /**
   * @brief Check if a given type is an nda::Array of a certain rank.
   *
   * @tparam A Type to check.
   * @tparam R Rank of the nda::Array type.
   */
  template <typename A, int R>
  concept ArrayOfRank = Array<A> and (get_rank<A> == R);

  /**
   * @brief Check if a given type is an nda::MemoryArray of a certain rank.
   *
   * @tparam A Type to check.
   * @tparam R Rank of the nda::MemoryArray type.
   */
  template <typename A, int R>
  concept MemoryArrayOfRank = MemoryArray<A> and (get_rank<A> == R);

  /**
   * @brief Check if if a given type is either an nda::Array or an nda::Scalar.
   * @tparam AS Type to check.
   */
  template <typename AS>
  concept ArrayOrScalar = Array<AS> or Scalar<AS>;

  /**
   * @brief Check if a given type is a matrix, i.e. an nda::ArrayOfRank<2>.
   * @note The algebra of the type is not checked here (see nda::get_algebra).
   * @tparam M Type to check.
   */
  template <typename M>
  concept Matrix = ArrayOfRank<M, 2>;

  /**
   * @brief Check if a given type is a vector, i.e. an nda::ArrayOfRank<1>.
   * @note The algebra of the type is not checked here (see nda::get_algebra).
   * @tparam V Type to check.
   */
  template <typename V>
  concept Vector = ArrayOfRank<V, 1>;

  /**
   * @brief Check if a given type is a memory matrix, i.e. an nda::MemoryArrayOfRank<2>.
   * @note The algebra of the type is not checked here (see nda::get_algebra).
   * @tparam M Type to check.
   */
  template <typename M>
  concept MemoryMatrix = MemoryArrayOfRank<M, 2>;

  /**
   * @brief Check if a given type is a memory vector, i.e. an nda::MemoryArrayOfRank<1>.
   * @note The algebra of the type is not checked here (see nda::get_algebra).
   * @tparam V Type to check.
   */
  template <typename V>
  concept MemoryVector = MemoryArrayOfRank<V, 1>;

  /**
   * @brief Check if a given type satisfies the array initializer concept for a given nda::MemoryArray type.
   *
   * @details They are mostly used in lazy mpi calls (see e.g. nda::mpi_reduce).
   *
   * @tparam A Type to check.
   * @tparam B nda::MemoryArray type.
   */
  template <typename A, typename B>
  concept ArrayInitializer = requires(A const &a) {
    { a.shape() } -> StdArrayOfLong;
    typename std::remove_cvref_t<A>::value_type;
    requires MemoryArray<B> && requires(B &b) { a.invoke(b); }; // FIXME not perfect: it should accept any layout ??
  };

  // FIXME : We should not need this ... Only used once...
  /**
   * @brief Check if a given type is constructible from the value type of a given nda::Array type.
   *
   * @tparam A nda::Array type.
   * @tparam U Type to check.
   */
  template <typename A, typename U>
  concept HasValueTypeConstructibleFrom = Array<A> and (std::is_constructible_v<U, get_value_t<A>>);

  /** @} */

} // namespace nda
