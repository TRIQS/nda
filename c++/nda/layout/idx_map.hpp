// Copyright (c) 2018--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a class that maps multi-dimensional indices to a linear index and vice versa.
 */

#pragma once

#include "./permutation.hpp"
#include "./range.hpp"
#include "./slice_static.hpp"
#include "../macros.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nda {

  /**
   * @addtogroup layout_idx
   * @{
   */

  /**
   * @brief Fortran/Column-major stride order.
   * @details The first dimension varies the fastest, the last dimension the slowest.
   * @tparam Rank Number of dimensions.
   */
  template <int Rank>
  constexpr uint64_t Fortran_stride_order = nda::encode(nda::permutations::reverse_identity<Rank>());

  /**
   * @brief C/Row-major stride order.
   * @details The last dimension varies the fastest, the first dimension the slowest.
   * @tparam Rank Number of dimensions.
   */
  template <int Rank>
  constexpr uint64_t C_stride_order = nda::encode(nda::permutations::identity<Rank>());

  /**
   * @brief Layout that specifies how to map multi-dimensional indices to a linear/flat index.
   *
   * @details It stores the shape of the array, i.e. the length of each dimension, and the strides of each dimension.
   * The stride of dimension `i` is the number of elements to skip in memory when the index of dimension `i` is
   * incremented by one. For example:
   * - To iterate over every element of a 5x5x5 array in C-order use the strides `(25, 5, 1)`.
   * - To iterate over every 2nd element of a 1D array use the stride `(2)`.
   * - To iterate over every 2nd column of a 10x10 array in Fortran-order use the strides `(1, 20)`.
   * - To iterate over every 2nd row of a 10x10 array in Fortran-order use the strides `(2, 10)`.
   *
   * The template parameters `StaticExtents` and `StrideOrder` are encoded as `uint64_t`. They can be decoded to a
   * `std::array<int, Rank>` using the nda::decode function. The encoding limits the number of dimensions, i.e. the
   * rank, to 16.
   *
   * The static extent array specifies the length of each dimension at compile-time. A zero value means that the length
   * in this dimension is dynamic and will be specified at runtime. Note that static lengths cannot exceed 16 (due to
   * the encoding). For example:
   * - `StaticExtents = nda::encode(std::array<int, Rank>{0, 10, 0, 5})` corresponds to a 4D array with dynamic extents
   * in dimension 0 and 2, and static extents of length 10 and 5 in dimension 1 and 3, respectively.
   *
   * The stride order specifies the order in which the dimensions are stored in memory. In its array form, it can be any
   * permutation of the integer values from `0` to `Rank - 1`. The slowest varying dimension is the first element of the
   * array and the fastest varying dimension is the last element of the array. For example:
   * - C-order: `StrideOrder = nda::encode(std::array<int, Rank>{0, 1, ..., Rank - 2, Rank - 1})`.
   * - Fortran-order: `StrideOrder = nda::encode(std::array<int, Rank>{Rank - 1, Rank - 2, ..., 1, 0})`.
   *
   * `LayoutProp` is an enum flag that can be used to make further compile-time guarantees about the layout of the data
   * in memory (see nda::layout_prop_e).
   *
   * @tparam Rank Number of dimensions.
   * @tparam StaticExtent Compile-time known shape (zero if fully dynamic).
   * @tparam StrideOrder Order in which the dimensions are stored in memory.
   * @tparam LayoutProp Compile-time guarantees about the layout of the data in memory.
   */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map {
    static_assert(Rank < 16, "Error in nda::idx_map: Rank must be < 16");
    static_assert((StrideOrder != 0) or (Rank == 1), "Error in nda::idx_map: StrideOrder can only be zero for 1D arrays");

    // Extents of all dimensions (the shape of the map).
    std::array<long, Rank> len{};

    // Strides of all dimensions.
    std::array<long, Rank> str{};

    public:
    /// Encoded static extents.
    static constexpr uint64_t static_extents_encoded = StaticExtents;

    /// Decoded static extents.
    static constexpr std::array<int, Rank> static_extents = decode<Rank>(StaticExtents);

    /// Decoded stride order.
    static constexpr std::array<int, Rank> stride_order = (StrideOrder == 0 ? permutations::identity<Rank>() : decode<Rank>(StrideOrder));

    /// Encoded stride order.
    static constexpr uint64_t stride_order_encoded = encode(stride_order);

    /// Compile-time memory layout properties.
    static constexpr layout_prop_e layout_prop = LayoutProp;

    /// Compile-time information about the layout (stride order and layout properties).
    static constexpr layout_info_t layout_info = layout_info_t{stride_order_encoded, layout_prop};

    /// Alias template to check if type `T` can be used to access a single element.
    template <typename T>
    static constexpr int argument_is_allowed_for_call = std::is_constructible_v<long, T>;

    /// Alias template to check if type `T` can be used to either access a single element or a slice of elements.
    template <typename T>
    static constexpr int argument_is_allowed_for_call_or_slice =
       std::is_same_v<range, T> or std::is_same_v<range::all_t, T> or std::is_same_v<ellipsis, T> or std::is_constructible_v<long, T>;

    protected:
    /// Number of dynamic dimensions/extents.
    static constexpr int n_dynamic_extents = []() {
      int r = 0;
      for (int u = 0; u < Rank; ++u) r += (static_extents[u] == 0 ? 1 : 0);
      return r;
    }();

    public:
    /**
     * @brief Get the rank of the map.
     * @return Number of dimensions.
     */
    static constexpr int rank() noexcept { return Rank; }

    /**
     * @brief Get the total number of elements.
     * @return Product of the extents of all dimensions.
     */
    [[nodiscard]] long size() const noexcept { return std::accumulate(len.cbegin(), len.cend(), 1L, std::multiplies<>{}); }

    /**
     * @brief Get the size known at compile-time.
     * @return Zero if it has at least one dynamic dimension, otherwise the product of the static extents.
     */
    static constexpr long ce_size() noexcept {
      if constexpr (n_dynamic_extents != 0) {
        return 0;
      } else {
        return std::accumulate(static_extents.cbegin(), static_extents.cend(), 1L, std::multiplies<>{});
      }
    }

    /**
     * @brief Get the extents of all dimensions.
     * @return `std::array<long, Rank>` containing the extent of each dimension.
     */
    [[nodiscard]] std::array<long, Rank> const &lengths() const noexcept { return len; }

    /**
     * @brief Get the strides of all dimensions.
     * @return `std::array<long, Rank>` containing the stride of each dimension.
     */
    [[nodiscard]] std::array<long, Rank> const &strides() const noexcept { return str; }

    /**
     * @brief Get the value of the smallest stride (positive or negative).
     * @return Stride of the fastest varying dimension.
     */
    [[nodiscard]] long min_stride() const noexcept { return str[stride_order[Rank - 1]]; }

    /**
     * @brief Is the data contiguous in memory?
     *
     * @details The data is contiguous in memory if the size of the map is equal to the number of memory locations
     * spanned by the map, i.e. the absolute value of the product of the largest stride times the length of the
     * corresponding dimension.
     *
     * @return True if the data is contiguous in memory, false otherwise.
     */
    [[nodiscard]] bool is_contiguous() const noexcept {
      auto s = size();
      if (s == 0) return true;
      return (std::abs(str[stride_order[0]] * len[stride_order[0]]) == s);
    }

    /**
     * @brief Are all strides positive?
     * @return True if all strides are positive, false otherwise.
     */
    [[nodiscard]] bool has_positive_strides() const noexcept { return (*std::min_element(str.cbegin(), str.cend()) >= 0); }

    /**
     * @brief Is the data strided in memory with a constant stride?
     *
     * @details The data is strided in memory with a constant stride if the size of the map times the absolute value of
     * the stride of the fastest dimension with an extent bigger than 1 is equal to the number of memory locations
     * spanned by the map, i.e. the absolute value of the product of the largest stride times the length of the
     * corresponding dimension.
     *
     * @return True if the data is strided in memory with a constant stride, false otherwise.
     */
    [[nodiscard]] bool is_strided_1d() const noexcept {
      auto s = size();
      if (s == 0) return true;
      int i = Rank - 1;
      while (len[stride_order[i]] == 1 and i > 0) --i;
      return (std::abs(str[stride_order[0]] * len[stride_order[0]]) == s * std::abs(str[stride_order[i]]));
    }

    /**
     * @brief Is the stride order equal to C-order?
     * @return True if the stride order is the identity permutation.
     */
    static constexpr bool is_stride_order_C() { return (stride_order == permutations::identity<Rank>()); }

    /**
     * @brief Is the stride order equal to Fortran-order?
     * @return True if the stride order is the reverse identity permutation.
     */
    static constexpr bool is_stride_order_Fortran() { return (stride_order == permutations::reverse_identity<Rank>()); }

    /**
     * @brief Check if a given shape and strides are compatible with the stride order.
     *
     * @tparam Int Integer type.
     * @param lenptr Pointer to a shape array.
     * @param strptr Pointer to a stride array.
     * @return True if the given shape and strides are compatible with the stride order.
     */
    template <std::integral Int>
    [[nodiscard]] static bool is_stride_order_valid(Int *lenptr, Int *strptr) {
      auto dims_to_check = std::vector<int>{};
      dims_to_check.reserve(Rank);
      for (auto dim : stride_order)
        if (lenptr[dim] > 1) dims_to_check.push_back(dim);
      for (int n = 1; n < dims_to_check.size(); ++n)
        if (std::abs(strptr[dims_to_check[n - 1]]) < std::abs(strptr[dims_to_check[n]])) return false;
      return true;
    }

    /**
     * @brief Check if the shape and strides of the current map are compatible with its stride order.
     * @details See idx_map::is_stride_order_valid(Int *lenptr, Int *strptr)).
     * @return True if the shape and strides are compatible with the stride order.
     */
    [[nodiscard]] bool is_stride_order_valid() const { return is_stride_order_valid(len.data(), str.data()); }

    private:
    // Compute contiguous strides from the shape.
    void compute_strides_contiguous() {
      long s = 1;
      for (int v = rank() - 1; v >= 0; --v) {
        int u  = stride_order[v];
        str[u] = s;
        s *= len[u];
      }
      ENSURES(s == size());
    }

    // Check that the static extents and the shape are compatible.
    void assert_static_extents_and_len_are_compatible() const {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if constexpr (n_dynamic_extents != Rank) {
#ifndef NDEBUG
        for (int u = 0; u < Rank; ++u)
          if (static_extents[u] != 0) EXPECTS(static_extents[u] == len[u]);
#endif
      }
#endif
    }

    // Should we check the stride order when constructing an idx_map from a shape and strides?
#ifdef NDA_DEBUG
    static constexpr bool check_stride_order = true;
#else
    static constexpr bool check_stride_order = false;
#endif

    // Combine static and dynamic extents.
    static std::array<long, Rank> merge_static_and_dynamic_extents(std::array<long, n_dynamic_extents> const &dynamic_extents) {
      std::array<long, Rank> extents;
      for (int u = 0, v = 0; u < Rank; ++u) extents[u] = (static_extents[u] == 0 ? dynamic_extents[v++] : static_extents[u]);
      return extents;
    }

    // FIXME ADD A CHECK layout_prop_e ... compare to stride and

    public:
    /**
     * @brief Default constructor.
     *
     * @details For purely static maps, the shape is set to the static extents and the strides are assumed to be
     * contiguous.
     *
     * For all other maps, the shape is set to zero and the strides are not initialized.
     */
    idx_map() {
      if constexpr (n_dynamic_extents == 0) {
        for (int u = 0; u < Rank; ++u) len[u] = static_extents[u];
        compute_strides_contiguous();
      }
    }

    /**
     * @brief Construct a new map from an existing map with different layout properties.
     *
     * @tparam LP Layout properties of the other nda::idx_map.
     * @param idxm Other nda::idx_map object.
     */
    template <layout_prop_e LP>
    idx_map(idx_map<Rank, StaticExtents, StrideOrder, LP> const &idxm) noexcept : len(idxm.lengths()), str(idxm.strides()) {
      // check strides and stride order of the constructed map
      EXPECTS(is_stride_order_valid());

      // check that the layout properties are compatible
      if constexpr (not layout_property_compatible(LP, layout_prop)) {
        if constexpr (has_contiguous(layout_prop)) {
          EXPECTS_WITH_MESSAGE(idxm.is_contiguous(), "Error in nda::idx_map: Constructing a contiguous from a non-contiguous layout");
        }
        if constexpr (has_strided_1d(layout_prop)) {
          EXPECTS_WITH_MESSAGE(idxm.is_strided_1d(), "Error in nda::idx_map: Constructing a strided_1d from a non-strided_1d layout");
        }
      }
    }

    /**
     * @brief Construct a new map from an existing map with different layout properties and different static extents.
     *
     * @tparam SE Static extents of the other nda::idx_map.
     * @tparam LP Layout properties of the other nda::idx_map.
     * @param idxm Other nda::idx_map object.
     */
    template <uint64_t SE, layout_prop_e LP>
    idx_map(idx_map<Rank, SE, StrideOrder, LP> const &idxm) noexcept(false) : len(idxm.lengths()), str(idxm.strides()) {
      // check strides and stride order
      EXPECTS(is_stride_order_valid());

      // check that the layout properties are compatible
      if constexpr (not layout_property_compatible(LP, LayoutProp)) {
        if constexpr (has_contiguous(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(idxm.is_contiguous(), "Error in nda::idx_map: Constructing a contiguous from a non-contiguous layout");
        }
        if constexpr (has_strided_1d(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(idxm.is_strided_1d(), "Error in nda::idx_map: Constructing a strided_1d from a non-strided_1d layout");
        }
      }

      // check that the static extents and the shape are compatible
      assert_static_extents_and_len_are_compatible();
    }

    /**
     * @brief Construct a new map from a given shape and strides.
     *
     * @param shape Shape of the new map.
     * @param strides Strides of the new map.
     */
    idx_map(std::array<long, Rank> const &shape, // NOLINT (only throws if check_stride_order is true)
            std::array<long, Rank> const &strides) noexcept(!check_stride_order)
       : len(shape), str(strides) {
      EXPECTS(std::all_of(shape.cbegin(), shape.cend(), [](auto const &i) { return i >= 0; }));
      if constexpr (check_stride_order) {
        if (not is_stride_order_valid()) throw std::runtime_error("Error in nda::idx_map: Incompatible strides, shape and stride order");
      }
    }

    /**
     * @brief Construct a new map from a given shape and with contiguous strides.
     *
     * @tparam Int Integer type.
     * @param shape Shape of the new map.
     */
    template <std::integral Int = long>
    idx_map(std::array<Int, Rank> const &shape) noexcept : len(stdutil::make_std_array<long>(shape)) {
      EXPECTS(std::all_of(shape.cbegin(), shape.cend(), [](auto const &i) { return i >= 0; }));
      assert_static_extents_and_len_are_compatible();
      compute_strides_contiguous();
    }

    /**
     * @brief Construct a new map from an array with its dynamic extents.
     *
     * @details The missing extents are taken from the static extents, i.e. if a static extent is zero, it is replaced
     * by the corresponding dynamic extent.
     *
     * @param shape std::array with the dynamic extents only.
     */
    idx_map(std::array<long, n_dynamic_extents> const &shape) noexcept
      requires((n_dynamic_extents != Rank) and (n_dynamic_extents != 0))
       : idx_map(merge_static_and_dynamic_extents(shape)) {}

    /**
     * @brief Construct a new map from an existing map with a different stride order.
     *
     * @warning This constructor is deleted.
     *
     * @tparam SE Static extents of the other nda::idx_map.
     * @tparam SO Stride order of the other nda::idx_map.
     * @tparam LP Layout properties of the other nda::idx_map.
     */
    template <uint64_t SE, uint64_t SO, layout_prop_e LP>
    idx_map(idx_map<Rank, SE, SO, LP> const &)
      requires(stride_order_encoded != SO)
    {
      static_assert((stride_order_encoded == SO), "Error in nda::idx_map: Incompatible stride orders");
    }

    /**
     * @brief Construct a new map with a shape of a different rank.
     * @warning This constructor is deleted.
     * @tparam R Rank of the given shape.
     */
    template <int R>
    idx_map(std::array<long, R> const &)
      requires(R != Rank)
    {
      static_assert(R == Rank, "Error in nda::idx_map: Incompatible ranks");
    }

    /// Default copy constructor.
    idx_map(idx_map const &) = default;

    /// Default move constructor.
    idx_map(idx_map &&) = default;

    /// Default copy assignment operator.
    idx_map &operator=(idx_map const &) = default;

    /// Default move assignment operator.
    idx_map &operator=(idx_map &&) = default;

    private:
    // Get the contribution to the linear index in case of an nda::ellipsis argument.
    template <bool skip_stride, auto I>
    [[nodiscard]] FORCEINLINE long myget(ellipsis) const noexcept {
      // nda::ellipsis are skipped and do not contribute to the linear index
      return 0;
    }

    // Get the contribution to the linear index in case of a long argument.
    template <bool skip_stride, auto I>
    [[nodiscard]] FORCEINLINE long myget(long arg) const noexcept {
      if constexpr (skip_stride and (I == stride_order[Rank - 1])) {
        // optimize for the case when the fastest varying dimension is contiguous in memory
        return arg;
      } else {
        // otherwise multiply the argument by the stride of the current dimension
        return arg * std::get<I>(str);
      }
    }

    // Is the smallest stride equal to one, i.e. is the fastest varying dimension contiguous in memory?
    static constexpr bool smallest_stride_is_one = has_smallest_stride_is_one(layout_prop);

    // Implementation of the function call operator that takes a multi-dimensional index and maps it to a linear index.
    template <typename... Args, size_t... Is>
    [[nodiscard]] FORCEINLINE long call_impl(std::index_sequence<Is...>, Args... args) const noexcept {
      // set e_pos to the position of the ellipsis, otherwise to -1
      static constexpr int e_pos = ((std::is_same_v<Args, ellipsis> ? int(Is) + 1 : 0) + ...) - 1;

      if constexpr (e_pos == -1) {
        // no ellipsis present
        if constexpr (smallest_stride_is_one) {
          // optimize for the case that the fastest varying dimension is contiguous in memory
          return (myget<true, Is>(static_cast<long>(args)) + ...);
        } else {
          // arbitrary layouts
          return ((args * std::get<Is>(str)) + ...);
        }
      } else {
        // empty ellipsis is present and needs to be skipped
        return (myget<smallest_stride_is_one, (Is < e_pos ? Is : Is - 1)>(args) + ...);
      }
    }

    public:
    /**
     * @brief Function call operator to map a given multi-dimensional index to a linear index.
     *
     * @details All arguments are either convertible to type `long` or are of type nda::ellipsis. The number of
     * non-ellipsis arguments must be equal to the rank of the map and there must be at most one nda::ellipsis. If an
     * nda::ellipsis is present, it is skipped and does not influence the result.
     *
     * Let \f$ (a_0, ..., a_{n-1}) \f$ be the given multi-dimensional index and let \f$ (s_0, ..., s_{n-1}) \f$ be the
     * strides of the index map. Then the equation that maps the multi-dimensional index to the corresponding linear
     * index \f$ f \f$ is as follows:
     * \f[
     *   f = \sum_{i=0}^{n-1} a_i * s_i \; .
     * \f]
     *
     * @tparam Args Types of the arguments.
     * @param args Multi-dimensional index.
     * @return Linear/Flat index.
     */
    template <typename... Args>
    FORCEINLINE long operator()(Args const &...args) const
#ifdef NDA_ENFORCE_BOUNDCHECK
       noexcept(false) {
      assert_in_bounds(rank(), len.data(), args...);
#else
       noexcept(true) {
#endif
      return call_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
    }

    /**
     * @brief Calculate the multi-dimensional index from a given linear index.
     *
     * @details Let \f$ f \f$ be the given linear index and let \f$ (s_0, ..., s_{n-1}) \f$ be the strides of the index
     * map. Then the corresponding multi-dimensional index \f$ (a_0, ..., a_{n-1}) \f$ satisfies the following equation:
     * \f[
     *   f = \sum_{i=0}^{n-1} a_i * s_i \; .
     * \f]
     *
     * Furthermore, if \f$ (p_0, ..., p_{n-1}) \f$ is the stride order of the map, let us define the residues
     * \f$ r_{p_i} \f$ as
     * \f[
     *   r_{p_i} = \sum_{j=i}^{n-1} s_{p_j} * a_{p_j} \; ,
     * \f]
     * with the property that \f$ s_{p_i} \leq r_{p_i} < s_{p_{i-1}} \f$ for \f$ i = 1, ..., n-1 \f$ and \f$ s_{p_0}
     * \leq r_{p_0} = f \f$. We can use this property to first calculate the residues and then recover the
     * multi-dimensional index.
     *
     * @param lin_idx Linear/Flat index.
     * @return Multi-dimensional index.
     */
    std::array<long, Rank> to_idx(long lin_idx) const {
      // compute residues starting from slowest index
      std::array<long, Rank> residues;
      residues[0] = lin_idx;
      for (auto i : range(1, Rank)) { residues[i] = residues[i - 1] % str[stride_order[i - 1]]; }

      // convert residues to indices, ordered from slowest to fastest
      std::array<long, Rank> idx;
      idx[Rank - 1] = residues[Rank - 1] / str[stride_order[Rank - 1]];
      for (auto i : range(Rank - 2, -1, -1)) { idx[i] = (residues[i] - residues[i + 1]) / str[stride_order[i]]; }

      // reorder indices according to stride order
      return permutations::apply_inverse(stride_order, idx);
    }

    /**
     * @brief Get a new nda::idx_map by taking a slice of the current one.
     *
     * @details See nda::slice_static::slice_idx_map for more information.
     *
     * @tparam Args Types of the arguments.
     * @param args Multi-dimensional index consisting of `long`, `nda::range`, `nda::range::all_t` or nda::ellipsis
     * objects.
     * @return A std::pair containing the offset in memory, i.e. the flat index of the first element of the slice and
     * the new nda::idx_map.
     */
    template <typename... Args>
    auto slice(Args const &...args) const {
      return slice_static::slice_idx_map(*this, args...);
    }

    /**
     * @brief Equal-to operator for two nda::idx_map objects.
     *
     * @tparam R Rank of the other nda::idx_map.
     * @tparam SE Static extents of the other nda::idx_map.
     * @tparam SO Stride order of the other nda::idx_map.
     * @tparam LP Layout properties of the other nda::idx_map.
     * @param rhs Right hand side nda::idx_map operand.
     * @return True if their ranks, shapes and strides are equal.
     */
    template <int R, uint64_t SE, uint64_t SO, layout_prop_e LP>
    bool operator==(idx_map<R, SE, SO, LP> const &rhs) const {
      return (Rank == R and len == rhs.lengths() and str == rhs.strides());
    }

    /**
     * @brief Create a new map by permuting the indices/dimensions of the current map with a given permutation.
     *
     * @details Let `A` be the current and ``A'`` the new, permuted index map. `P` is the given permutation. We define
     * the permuted nda::idx_map ``A'`` to be the one with the following properties:
     * - ``A'(i_0,...,i_{n-1}) = A(i_{P[0]},...,i_{P[n-1]})``
     * - ``A'.lengths()[k] == A.lengths()[P^{-1}[k]]``
     * - ``A'.strides()[k] == A.strides()[P^{-1}[k]]``
     * - The stride order of ``A'`` is the composition of `P` and the stride order of `A` (note that the stride order
     * itself is a permutation).
     *
     * @tparam Permutation Permutation to apply.
     * @return New nda::idx_map with permuted indices.
     */
    template <uint64_t Permutation>
    auto transpose() const {
      // Makes a new transposed idx_map with permutation P such that
      // denoting here A = this, A' = P A = returned_value
      // A'(i_k) = A(i_{P[k]})
      //
      // Note that this convention is the correct one to have a (left) action of the symmetric group on
      // a array and it may not be completely obvious.
      // Proof
      //  let's operate with P then Q, and denote A'' = Q A'. We want to show that A'' = (QP) A
      //   A'(i_k) = A(i_{P[k]})
      //   A''(j_k) = A'(j_{Q[k]})
      //   then i_k = j_{Q[k]} and  A''(j_k) =  A(i_{P[k]}) = A(j_{Q[P[k]]}) = A(j_{(QP)[k]}), q.e.d
      //
      // NB test will test this composition
      // Denoting this as A, an indexmap, calling it returns the linear index given by
      //
      // A(i_k) = sum_k i_k * S[k] (1)
      //
      // where S[k] denotes the strides.
      //
      // 1- S' : strides of A'
      //    A'(i_k) = sum_k i_{P[k]} * S[k] = sum_k i_k * S[P{^-1}[k]]
      //     so
      //         S'[k] = S[P{^-1}[k]]  (2)
      //    i.e. apply (inverse(P), S) or apply_inverse directly.
      //
      // 2- L' : lengths of A'
      //    if L[k] is the k-th length, then because of the definition of A', i.e. A'(i_k) = A(i_{P[k]})
      //    i_q in the lhs A is at position q' such that P[q'] = q  (A'(i0 i1 i2...) = A( i_P0 i_P1 i_P2....)
      //    hence L'[q] = L[q'] =  L[P^{-1}[q]]
      //    same for static length
      //
      // 3- stride_order: denoted in this paragraph as Q (and Q' for A').
      //    by definition Q is a permutation such that Q[0] is the slowest index, Q[Rank -1] the fastest
      //    hence S[Q[k]] is a strictly decreasing sequence (as checked by strides_compatible_to_stride_order)
      //    we want therefore Q' the permutation that will sort the S', i.e.
      //    S'[Q'[k]] = S[Q[k]]
      //    using (2), we have S[P{^-1}[Q'[k]]] = S[Q[k]]
      //    so the permutation Q' is such that  P{^-1}Q' = Q  or Q' = PQ (as permutation product/composition).
      //    NB : Q and P are permutations, so the operation must be a composition, not an apply (apply applies
      //    a P to any set, like L, S, not only a permutation) even though they are all std::array in the code ...
      //
      static constexpr std::array<int, Rank> permu              = decode<Rank>(Permutation);
      static constexpr std::array<int, Rank> new_stride_order   = permutations::compose(permu, stride_order);
      static constexpr std::array<int, Rank> new_static_extents = permutations::apply_inverse(permu, static_extents);

      return idx_map<Rank, encode(new_static_extents), encode(new_stride_order), LayoutProp>{permutations::apply_inverse(permu, lengths()),
                                                                                             permutations::apply_inverse(permu, strides())};
    }
  };

  /** @} */

} // namespace nda
