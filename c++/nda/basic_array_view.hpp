// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides the generic class for views.
 */

#pragma once

#include "./basic_functions.hpp"
#include "./clef.hpp"
#include "./concepts.hpp"
#include "./declarations.hpp"
#include "./iterators.hpp"
#include "./layout/for_each.hpp"
#include "./layout/idx_map.hpp"
#include "./layout/permutation.hpp"
#include "./layout/range.hpp"
#include "./layout/slice_static.hpp"
#include "./macros.hpp"
#include "./matrix_functions.hpp"
#include "./mem/address_space.hpp"
#include "./mem/fill.hpp"
#include "./mem/memcpy.hpp"
#include "./mem/memset.hpp"
#include "./mem/policies.hpp"
#include "./traits.hpp"

#include <itertools/itertools.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>

namespace std {

  /**
   * @brief std::swap is deleted for nda::basic_array_view.
   * @warning The std::swap is WRONG for an nda::basic_array_view because of its copy/move semantics. Use nda::swap
   * instead (the correct one, found by ADL).
   */
  template <typename V1, int R1, typename LP1, char A1, typename AP1, typename OP1, typename V2, int R2, typename LP2, char A2, typename AP2,
            typename OP2>
  void swap(nda::basic_array_view<V1, R1, LP1, A1, AP1, OP1> &a, nda::basic_array_view<V2, R2, LP2, A2, AP2, OP2> &b) = delete;

} // namespace std

namespace nda {

  /**
   * @ingroup arrays_views
   * @brief A generic view of a multi-dimensional array.
   *
   * @details Together with nda::basic_array, this class forms the backbone of the nda library. It is templated with
   * the following parameters:
   *
   * - `ValueType`: This is the type of the elements in the view. Depending on how the view was constructed and on the
   * underlying array, this can be a const or non-const type.
   * - `Rank`: Integer specifying the number of dimensions of the view. This is a compile-time constant.
   * - `LayoutPolicy`: The layout policy specifies how the view accesses its elements. It provides a mapping from
   * multi-dimensional to linear indices and vice versa (see @ref layout_pols).
   * - `Algebra`: The algebra specifies how the view behaves when it is used in an expression. Possible values are 'A'
   * (array), 'M' (matrix) and 'V' (vector) (see nda::get_algebra).
   * - `AccessorPolicy`: The accessor policy specifies how the view accesses the data pointer. This can be useful for
   * optimizations.
   * - `OwningPolicy`: The owning policy specifies the ownership of the data pointer and who is responsible for
   * handling the memory resources (see @ref mem_handles).
   *
   * In contrast to regular arrays (see nda::basic_array), views do not own the data they point to. They are a fast and
   * efficient way to access and manipulate already existing data:
   *
   * @code{.cpp}
   * // create a regular 3x3 array of ones
   * auto arr = nda::ones<int>(3, 3);
   * std::cout << arr << std::endl;
   *
   * // zero out the first column
   * arr(nda::range::all, 0) = 0;
   * std::cout << arr << std::endl;
   * @endcode
   *
   * Output:
   *
   * @code{bash}
   *
   * [[1,1,1]
   *  [1,1,1]
   *  [1,1,1]]
   *
   * [[0,1,1]
   *  [0,1,1]
   *  [0,1,1]]
   * @endcode
   *
   * Views are usually created by taking a slice of a regular nda::basic_array or another view. In the example above,
   * `arr(nda::range::all, 0)` creates a view of the first column of the regular array `arr`, which is then set to zero.
   * A view of the full array can be created with `arr()`.
   *
   * @tparam ValueType Type stored in the array.
   * @tparam Rank Number of dimensions of the view.
   * @tparam LayoutPolicy Policy determining the memory layout.
   * @tparam Algebra Algebra of the view.
   * @tparam AccessorPolicy Policy determining how the data pointer is accessed.
   * @tparam OwningPolicy Policy determining the ownership of the data.
   */
  template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  class basic_array_view {
    // Compile-time checks.
    static_assert((Algebra != 'N'), "Internal error in nda::basic_array_view: Algebra 'N' not supported");
    static_assert((Algebra != 'M') or (Rank == 2), "Internal error in nda::basic_array_view: Algebra 'M' requires a rank 2 view");
    static_assert((Algebra != 'V') or (Rank == 1), "Internal error in nda::basic_array_view: Algebra 'V' requires a rank 1 view");

    public:
    /// Type of the values in the view (might be const).
    using value_type = ValueType;

    /// Type of the memory layout policy (see @ref layout_pols).
    using layout_policy_t = LayoutPolicy;

    /// Type of the memory layout (an nda::idx_map).
    using layout_t = typename LayoutPolicy::template mapping<Rank>;

    /// Type of the accessor policy (see e.g. nda::default_accessor).
    using accessor_policy_t = AccessorPolicy;

    /// Type of the owning policy (see @ref mem_pols).
    using owning_policy_t = OwningPolicy;

    /// Type of the memory handle (see @ref mem_handles).
    using storage_t = typename OwningPolicy::template handle<ValueType>;

    /// The associated regular (nda::basic_array) type.
    using regular_type = basic_array<std::remove_const_t<ValueType>, Rank, C_layout, Algebra, heap<mem::get_addr_space<storage_t>>>;

    /// Number of dimensions of the view.
    static constexpr int rank = Rank;

    private:
    // Type of the view itself.
    using self_t = basic_array_view;

    // Constexpr variable that is true if the view is a view (always for basic_array_view).
    static constexpr bool is_view = true;

    // Constexpr variable that is true if the value_type is const.
    static constexpr bool is_const = std::is_const_v<ValueType>;

    // Memory layout of the view, i.e. the nda::idx_map.
    layout_t lay;

    // Memory handle of the view.
    storage_t sto;

    // Declare any nda::basic_array as a friend.
    template <typename T, int R, typename L, char A, typename CP>
    friend class basic_array;

    // Declare any other nda::basic_array_view as a friend.
    template <typename T, int R, typename L, char A, typename AP, typename OP>
    friend class basic_array_view;

    // Check the layout compatibility of this view with another layout.
    template <typename L>
    static constexpr bool requires_runtime_check = not layout_property_compatible(L::template mapping<Rank>::layout_prop, layout_t::layout_prop);

    public:
    // FIXME : TRIQS PORTING
    // private constructor for the previous friend
    /**
     * @brief Construct a view from a given layout and memory handle.
     *
     * @warning This should not be used directly. Use one of the other constructors instead.
     *
     * @param idxm Layout of the view.
     * @param st Memory handle of the view.
     */
    basic_array_view(layout_t const &idxm, storage_t st) : lay(idxm), sto(std::move(st)) {}

    public:
    // backward : FIXME : temporary to be removed
    /// @deprecated Convert the current view to a view with an 'A' (array) algebra.
    [[deprecated]] auto as_array_view() { return basic_array_view<ValueType, Rank, LayoutPolicy, 'A', AccessorPolicy, OwningPolicy>{*this}; };

    /// @deprecated Convert the current view to a view with an 'A' (array) algebra.
    [[deprecated]] auto as_array_view() const {
      return basic_array_view<const ValueType, Rank, LayoutPolicy, 'A', AccessorPolicy, OwningPolicy>{*this};
    };

    /// Default constructor constructs an empty view with a default constructed memory handle and layout.
    basic_array_view() = default;

    /// Default move constructor moves the memory handle and layout.
    basic_array_view(basic_array_view &&) = default;

    /// Default copy constructor copies the memory handle and layout.
    basic_array_view(basic_array_view const &) = default;

    /**
     * @brief Generic constructor from any nda::MemoryArray type.
     *
     * @details It simply copies the memory layout and initializes the memory handle with the handle of the given
     * nda::MemoryArray object.
     *
     * @tparam A nda::MemoryArray type of the same rank as the view.
     * @param a nda::MemoryArray object.
     */
    template <MemoryArrayOfRank<Rank> A>
      requires((get_layout_info<A>.stride_order == layout_t::stride_order_encoded)
               and (std::is_same_v<std::remove_const_t<ValueType>, get_value_t<A>>)
               and (std::is_const_v<ValueType> or !std::is_const_v<typename std::decay_t<A>::value_type>))
    explicit(requires_runtime_check<typename std::decay_t<A>::layout_policy_t>)
       basic_array_view(A &&a) noexcept // NOLINT (should we forward the reference?)
       : lay(a.indexmap()), sto(a.storage()) {}

    /**
     * @brief Construct a view from a bare pointer to some contiguous data and a shape.
     *
     * @note We do not have any control over the specified dimensions. The caller has to ensure their correctness and
     * their compatibility with the given data pointer.
     *
     * @param shape Shape of the view.
     * @param p Pointer to the data.
     */
    basic_array_view(std::array<long, Rank> const &shape, ValueType *p) noexcept : basic_array_view(layout_t{shape}, p) {}

    /**
     * @brief Construct a view from a bare pointer to some contiguous data and a memory layout.
     *
     * @note We do not have any control over the given layout. The caller has to ensure its correctness and its
     * compatibility with the given data pointer.
     *
     * @param idxm Layout of the view.
     * @param p Pointer to the data.
     */
    basic_array_view(layout_t const &idxm, ValueType *p) noexcept : lay(idxm), sto{p} {}

    /**
     * @brief Construct a 1-dimensional view of a std::array.
     *
     * @tparam N Size of the std::array.
     * @param a Reference to a std::array object.
     */
    template <size_t N>
      requires(Rank == 1)
    explicit basic_array_view(std::array<ValueType, N> &a) noexcept : basic_array_view{{long(N)}, a.data()} {}

    /**
     * @brief Construct a 1-dimensional view of a std::array.
     *
     * @tparam N Size of the std::array.
     * @param a Const reference to a std::array object.
     */
    template <size_t N>
      requires(Rank == 1 and std::is_const_v<ValueType>)
    explicit basic_array_view(std::array<std::remove_const_t<ValueType>, N> const &a) noexcept : basic_array_view{{long(N)}, a.data()} {}

    /**
     * @brief Construct a 1-dimensional view of a general contiguous range.
     *
     * @tparam R Type of the range.
     * @param rg Range object.
     */
    template <std::ranges::contiguous_range R>
      requires(Rank == 1 and not MemoryArray<R>
               and (std::is_same_v<std::ranges::range_value_t<R>, ValueType> or std::is_same_v<const std::ranges::range_value_t<R>, ValueType>))
    explicit basic_array_view(R &rg) noexcept : basic_array_view{{long(std::ranges::size(rg))}, std::to_address(std::begin(rg))} {}

    /**
     * @brief Copy assignment operator makes a deep copy of the contents of the view.
     *
     * @details The dimension of the right hand side must be large enough or the behaviour is undefined. If
     * `NDA_ENFORCE_BOUNDCHECK` is defined, bounds checking is enabled.
     *
     * @param rhs Right hand side of the assignment operation.
     */
    basic_array_view &operator=(basic_array_view const &rhs) noexcept {
      assign_from_ndarray(rhs);
      return *this;
    }

    /**
     * @brief Assignment operator makes a deep copy of the contents of an nda::ArrayOfRank object.
     *
     * @details The dimension of the right hand side must be large enough or the behaviour is undefined. If
     * `NDA_ENFORCE_BOUNDCHECK` is defined, bounds checking is enabled.
     *
     * @tparam RHS nda::ArrayOfRank type with the same rank as the view.
     * @param rhs Right hand side of the assignment operation.
     */
    template <ArrayOfRank<Rank> RHS>
    basic_array_view &operator=(RHS const &rhs) noexcept {
      static_assert(!is_const, "Cannot assign to an nda::basic_array_view with const value_type");
      assign_from_ndarray(rhs);
      return *this;
    }

    /**
     * @brief Assignment operator assigns a scalar to the view.
     *
     * @details The behavior depends on the algebra of the view:
     * - 'A' (array) and 'V' (vector): The scalar is assigned to all elements of the view.
     * - 'M' (matrix): The scalar is assigned to the diagonal elements of the shorter dimension.
     *
     * @tparam RHS Type of the scalar.
     * @param rhs Right hand side of the assignment operation.
     */
    template <typename RHS>
    basic_array_view &operator=(RHS const &rhs) noexcept
      requires(is_scalar_for_v<RHS, basic_array_view>)
    {
      static_assert(!is_const, "Cannot assign to an nda::basic_array_view with const value_type");
      assign_from_scalar(rhs);
      return *this;
    }

    /**
     * @brief Assignment operator uses an nda::ArrayInitializer to assign to the view.
     *
     * @details The shape of the view is expected to be the same as the shape of the initializer.
     *
     * @tparam Initializer nda::ArrayInitializer type.
     * @param initializer Initializer object.
     */
    template <ArrayInitializer<basic_array_view> Initializer>
    basic_array_view &operator=(Initializer const &initializer) noexcept {
      EXPECTS(shape() == initializer.shape());
      initializer.invoke(*this);
      return *this;
    }

    /**
     * @brief Rebind the current view to another view.
     *
     * @details It simply copies the layout and memory handle of the other view.
     *
     * @tparam T Value type of the other view.
     * @tparam R Rank of the other view.
     * @tparam LP Layout policy of the other view.
     * @tparam A Algebra of the other view.
     * @tparam AP Accessor policy of the other view.
     * @tparam OP Owning policy of the other view.
     *
     * @param v Other view.
     */
    template <typename T, int R, typename LP, char A, typename AP, typename OP>
    void rebind(basic_array_view<T, R, LP, A, AP, OP> v) noexcept {
      static_assert(R == Rank, "Cannot rebind a view to another view of a different rank");
      static_assert(std::is_same_v<std::remove_const_t<T>, std::remove_const_t<ValueType>>,
                    "Cannot rebind a view to another view with a different value_type (except for const)");
      static constexpr bool same_type = std::is_same_v<T, ValueType>;
      static_assert(same_type or is_const, "Cannot rebind a view with non-const value_type to another view with const value_type");
      if constexpr (same_type) {
        // FIXME Error message in layout error !
        lay = v.lay;
        sto = v.sto;
      } else if constexpr (is_const) {
        // the last if is always trivially true but in case of an error in the static_assert above,
        // it improves the error message by not compiling the = afterwards
        lay = layout_t{v.indexmap()};
        sto = storage_t{v.storage()};
      }
    }

    /**
     * @brief Swap two views by swapping their memory handles and layouts.
     *
     * @details This does not modify the data the views point to.
     *
     * @param a First view.
     * @param b Second view.
     */
    friend void swap(basic_array_view &a, basic_array_view &b) noexcept {
      std::swap(a.lay, b.lay);
      std::swap(a.sto, b.sto);
    }

    /**
     * @brief Swap two views by swapping their data.
     *
     * @details This modifies the data the views point to.
     *
     * @param a First view.
     * @param b Second view.
     */
    friend void deep_swap(basic_array_view a, basic_array_view b) noexcept {
      auto tmp = make_regular(a);
      a        = b;
      b        = tmp;
    }

// include common functionality of arrays and views
#include "./_impl_basic_array_view_common.hpp"
  };

  // Class template argument deduction guides.
  template <MemoryArray A>
  basic_array_view(A &&a)
     -> basic_array_view<std::conditional_t<std::is_const_v<std::remove_reference_t<A>>, const typename std::decay_t<A>::value_type,
                                            typename std::decay_t<A>::value_type>,
                         get_rank<A>, typename std::decay_t<A>::layout_policy_t, get_algebra<A>, default_accessor, borrowed<mem::get_addr_space<A>>>;

  template <typename V, size_t R>
  basic_array_view(std::array<V, R> &a)
     -> basic_array_view<V, 1, nda::basic_layout<nda::static_extents(R), nda::C_stride_order<1>, nda::layout_prop_e::contiguous>, 'V',
                         default_accessor, borrowed<>>;

  template <typename V, size_t R>
  basic_array_view(std::array<V, R> const &a)
     -> basic_array_view<const V, 1, nda::basic_layout<nda::static_extents(R), nda::C_stride_order<1>, nda::layout_prop_e::contiguous>, 'V',
                         default_accessor, borrowed<>>;

  template <std::ranges::contiguous_range R>
  basic_array_view(R &r) -> basic_array_view<std::conditional_t<std::is_const_v<R>, const typename R::value_type, typename R::value_type>, 1,
                                             C_layout, 'V', default_accessor, borrowed<>>;

} // namespace nda
