// Copyright (c) 2020-2023 Simons Foundation
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

/**
 * @file
 * @brief Provides an extension to nda::idx_map to support string indices.
 */

#pragma once

#include "./idx_map.hpp"
#include "./policies.hpp"
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../declarations.hpp"
#include "../exceptions.hpp"
#include "../traits.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @addtogroup layout_idx
   * @{
   */

  /// @cond
  // Forward declaration.
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class rect_str;
  /// @endcond

  namespace detail {

    // Get the corresponding nda::rect_str type from a given nda::idx_map type.
    template <typename T>
    struct rect_str_from_base;

    // Specialization of rect_str_from_base for nda::idx_map.
    template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
    struct rect_str_from_base<idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>> {
      // nda::rect_str type.
      using type = rect_str<Rank, StaticExtents, StrideOrder, LayoutProp>;
    };

  } // namespace detail

  /**
   * @brief Layout that specifies how to map multi-dimensional indices including possible string indices to a
   * linear/flat index.
   *
   * @details It extends the functionality of nda::idx_map by supporting string indices.
   *
   * @tparam Rank Number of dimensions.
   * @tparam StaticExtent Compile-time known shape (zero if fully dynamic).
   * @tparam StrideOrder Order in which the dimensions are stored in memory.
   * @tparam LayoutProp Compile-time guarantees about the layout of the data in memory.
   */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class rect_str : public idx_map<Rank, StaticExtents, StrideOrder, LayoutProp> {
    // Type for storing the string indices.
    using ind_t = nda::array<nda::array<std::string, 1>, 1>;

    // Type of the nda::idx_map base class.
    using base_t = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>;

    // String indices for each dimension.
    mutable std::shared_ptr<ind_t const> s_indices;

    // Number of dynamic dimensions/extents.
    using base_t::n_dynamic_extents;

    // Get the shape of the nda::rect_str from given string indices.
    static std::array<long, Rank> make_shape_from_string_indices(ind_t const &str_indices) {
      if (str_indices.size() != Rank)
        NDA_RUNTIME_ERROR << "Error in rect_str::make_shape_from_string_indices: String indices do not have the correct rank";
      std::array<long, Rank> sha;
      for (int i = 0; i < Rank; ++i) sha[i] = str_indices[i].size();
      return sha;
    }

    public:
    /**
     * @brief Get the string indices.
     *
     * @details If the string indices are not yet initialized, they are initialized with the default values, i.e. "0",
     * "1", "2", ..., and so on.
     *
     * @return String indices.
     */
    auto const &get_string_indices() const {
      if (not s_indices) {
        // string indices are not initialized
        auto ind = ind_t(Rank);
        for (int i = 0; i < Rank; ++i) {
          auto a = nda::array<std::string, 1>(this->lengths()[i]);
          for (int j = 0; j < a.size(); ++j) a(j) = std::to_string(j);
          ind(i) = std::move(a);
        }
        s_indices = std::make_shared<ind_t>(std::move(ind));
      }
      return *s_indices;
    }

    /// Alias template to check if type `T` can be used to access a specific element.
    template <typename T>
    static constexpr int argument_is_allowed_for_call = base_t::template argument_is_allowed_for_call<T> or std::is_constructible_v<std::string, T>;

    /// Alias template to check if type `T` can be used to either access a specific element or a slice of elements.
    template <typename T>
    static constexpr int argument_is_allowed_for_call_or_slice =
       base_t::template argument_is_allowed_for_call_or_slice<T> or std::is_constructible_v<std::string, T>;

    /**
     * @brief Default constructor.
     * @details The string indices are not initialized and the underlying nda::idx_map is default constructed.
     */
    rect_str() = default;

    /**
     * @brief Construct an nda::rect_str from a given nda::idx_map.
     * @param idxm nda::idx_map object.
     */
    rect_str(base_t const &idxm) noexcept : base_t{idxm} {}

    /**
     * @brief Construct an nda::rect_str from a given nda::idx_map and string indices.
     *
     * @warning The shape of the string indices is not checked to be consistent with the shape of the nda::idx_map.
     *
     * @param idxm nda::idx_map object.
     * @param str_indices String indices.
     */
    rect_str(base_t const &idxm, ind_t const &str_indices) noexcept : base_t{idxm}, s_indices{std::make_shared<ind_t>(std::move(str_indices))} {}

    /**
     * @brief Construct an nda::rect_str from another nda::rect_str with different layout properties.
     *
     * @tparam LP Layout properties of the other nda::rect_str.
     * @param rstr Other nda::rect_str object.
     */
    template <layout_prop_e LP>
    rect_str(rect_str<Rank, StaticExtents, StrideOrder, LP> const &rstr) noexcept
       : base_t{rstr}, s_indices{std::make_shared<ind_t>(rstr.get_string_indices())} {}

    /**
     * @brief Construct an nda::rect_str from another nda::rect_str with different layout properties and static extents.
     *
     * @tparam SE Static extents of the other nda::rect_str.
     * @tparam LP Layout properties of the other nda::rect_str.
     * @param rstr Other nda::rect_str object.
     */
    template <uint64_t SE, layout_prop_e LP>
    rect_str(rect_str<Rank, SE, StrideOrder, LP> const &rstr) noexcept(false)
       : base_t{rstr}, s_indices{std::make_shared<ind_t>(rstr.get_string_indices())} {}

    /**
     * @brief Construct an nda::rect_str from a given shape and strides.
     *
     * @param shape Shape of the map.
     * @param strides Strides of the map.
     */
    rect_str(std::array<long, Rank> const &shape, std::array<long, Rank> const &strides) noexcept : base_t{shape, strides} {}

    /**
     * @brief Construct an nda::rect_str from a given shape and with contiguous strides.
     * @param shape Shape of the map.
     */
    rect_str(std::array<long, Rank> const &shape) noexcept : base_t{shape} {}

    /**
     * @brief Construct an nda::rect_str from given string indices and with contiguous strides.
     * @param str_indices String indices.
     */
    rect_str(nda::array<nda::array<std::string, 1>, 1> str_indices) noexcept(false)
       : base_t{make_shape_from_string_indices(str_indices)}, s_indices{std::make_shared<ind_t>(std::move(str_indices))} {}

    /**
     * @brief Construct an nda::rect_str from an array with its dynamic extents.
     * @details The missing extents are taken from the static extents.
     * @param shape sta::array containing the dynamic extents only.
     */
    rect_str(std::array<long, base_t::n_dynamic_extents> const &shape) noexcept
      requires((base_t::n_dynamic_extents != Rank) and (base_t::n_dynamic_extents != 0))
       : base_t{shape} {}

    /// Default copy constructor.
    rect_str(rect_str const &) = default;

    /// Default move constructor.
    rect_str(rect_str &&) = default;

    /// Default copy assignment operator.
    rect_str &operator=(rect_str const &) = default;

    /// Default move assignment operator.
    rect_str &operator=(rect_str &&) = default;

    private:
    // Convert a given string argument into a corresponding index. If the argument isn't a string, it is returned as is.
    template <typename T>
    auto peel_string(int pos, T const &arg) const {
      if constexpr (not std::is_constructible_v<std::string, T>)
        // argument is not a string, simply return it
        return arg;
      else {
        // argument is a string, find its position in the string indices of the given dimension
        auto const &sind = get_string_indices();
        auto const &idx  = sind[pos];
        auto it          = std::find(idx.begin(), idx.end(), arg);
        if (it == idx.end()) NDA_RUNTIME_ERROR << "Error in nda::rect_str: Key " << arg << " at position " << pos << " does not match an index";
        return it - idx.begin();
      }
    }

    // Actual implementation of the function call operator.
    template <typename... Args, size_t... Is>
    [[nodiscard]] FORCEINLINE long call_impl(std::index_sequence<Is...>, Args... args) const {
      // calls the underlying nda::idx_map::operator() after converting all string arguments to long indices
      return base_t::operator()(peel_string(Is, args)...);
    }

    public:
    /**
     * @brief Function call operator to map a given multi-dimensional index to a linear index.
     *
     * @details See also nda::idx_map.
     *
     * @tparam Args Types of the arguments.
     * @param args Multi-dimensional index, including possible string indices.
     * @return Linear/Flat index.
     */
    template <typename... Args>
    FORCEINLINE long operator()(Args const &...args) const {
      return call_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
    }

    private:
    // Helper function to get a new nda::rect_str by taking a slice of the current one.
    template <typename... Args, auto... Is>
    FORCEINLINE decltype(auto) slice_impl(std::index_sequence<Is...>, Args const &...args) const {
      // convert string arguments to long indices and slice the underlying nda::idx_map
      auto const [offset, idxm2] = base_t::slice(peel_string(Is, args)...);

      // type of sliced nda::rect_str
      using new_rect_str_t = typename detail::rect_str_from_base<std::decay_t<decltype(idxm2)>>::type;

      // if the string indices have not been initialized, simply return a new nda::rect_str with the sliced nda::idx_map
      if (not s_indices) return std::make_pair(offset, new_rect_str_t{idxm2});

      // otherwise slice the string indices as well (not optimized but simple)
      auto const &current_ind = get_string_indices();
      ind_t ind2((not argument_is_allowed_for_call<Args> + ...)); // will not work for ellipsis that cover more than one dimension
      auto add_string_indices = [p = 0, &current_ind, &ind2](int n, auto const &y) mutable -> void {
        using U = std::decay_t<decltype(y)>;
        if constexpr (not argument_is_allowed_for_call<U>) { ind2[p++] = current_ind[n](y); }
      };
      (add_string_indices(Is, args), ...);

      return std::make_pair(offset, new_rect_str_t{idxm2, ind2});
    }

    public:
    /**
     * @brief Get a new nda::rect_str by taking a slice of the current one.
     *
     * @warning nda::ellipsis that cover more than 1 dimension will not work properly. Use `nda::range::all_t` instead.
     *
     * @tparam Args Types of the arguments.
     * @param args Multi-dimensional index consisting of strings, `long`, `nda::range`, `nda::range::all_t` or
     * nda::ellipsis objects.
     * @return A std::pair containing the offset in memory, i.e. the flat index of the first element of the slice and
     * the new nda::rect_str.
     */
    template <typename... Args>
    auto slice(Args const &...args) const {
      return slice_impl(std::make_index_sequence<sizeof...(args)>{}, args...);
    }

    /**
     * @brief Equal-to operator for two nda::rect_str objects.
     *
     * @param rhs Right hand side nda::rect_str operand.
     * @return True if the underlying nda::idx_map and the string indices are equal, false otherwise.
     */
    bool operator==(rect_str const &rhs) const {
      return base_t::operator==(rhs) and (!s_indices or !rhs.s_indices or (*s_indices == *(rhs.s_indices)));
    }

    /**
     * @brief Not-equal-to operator for two nda::rect_str objects.
     *
     * @param rhs Right hand side nda::rect_str operand.
     * @return True if they are not equal, false otherwise.
     */
    bool operator!=(rect_str const &rhs) { return !(operator==(rhs)); }

    /**
     * @brief Create a new nda::rect_str by permuting the indices/dimensions with a given permutation.
     *
     * @details Let `A` be the current and ``A'`` the new, permuted map. `P` is the given permutation. We define the
     * permuted nda::rect_str ``A'`` to be the one with the following properties:
     * - ``A'(i_0,...,i_{n-1}) = A(i_{P[0]},...,i_{P[n-1]})``
     * - ``A'.lengths()[k] == A.lengths()[P^{-1}[k]]``
     * - ``A'.strides()[k] == A.strides()[P^{-1}[k]]``
     * - The stride order of ``A'`` is the composition of `P` and the stride order of `A` (note that the stride order
     * itself is a permutation).
     *
     * @tparam Permutation Permutation to apply.
     * @return New nda::rect_str with permuted indices.
     */
    template <uint64_t Permutation>
    auto transpose() const {
      // transpose the underlying nda::idx_map
      auto idxm2 = base_t::template transpose<Permutation>();

      // type of transposed nda::rect_str
      using new_rect_str_t = typename detail::rect_str_from_base<std::decay_t<decltype(idxm2)>>::type;

      // if the string indices have not been initialized, simply return the transposed nda::rect_str
      if (not s_indices) return new_rect_str_t{idxm2};

      // otherwise transpose the string indices as well
      ind_t ind2(s_indices->size());
      static constexpr std::array<int, Rank> permu = decode<Rank>(Permutation);
      for (int u = 0; u < Rank; ++u) { ind2[permu[u]] = (*s_indices)[u]; }
      return new_rect_str_t{idxm2, ind2};
    }
  };

  /** @} */

  /**
   * @addtogroup layout_pols
   * @{
   */

  /// @cond
  // Forward declarations.
  struct C_stride_layout_str;
  struct F_stride_layout_str;
  /// @endcond

  /**
   * @brief Contiguous layout policy with C-order (row-major order) and possible string indices.
   * @details The last dimension varies the fastest, the first dimension varies the slowest.
   */
  struct C_layout_str {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = rect_str<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = C_stride_layout_str;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = C_layout_str;
  };

  /**
   * @brief Contiguous layout policy with Fortran-order (column-major order) and possible string indices.
   * @details The first dimension varies the fastest, the last dimension varies the slowest.
   */
  struct F_layout_str {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = rect_str<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::contiguous>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = F_stride_layout_str;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = F_layout_str;
  };

  /**
   * @brief Strided (non-contiguous) layout policy with C-order (row-major order) and possible string indices.
   * @details The last dimension varies the fastest, the first dimension varies the slowest.
   */
  struct C_stride_layout_str {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = rect_str<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = C_stride_layout_str;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = C_layout_str;
  };

  /**
   * @brief Strided (non-contiguous) layout policy with Fortran-order (column-major order) and possible string indices.
   * @details The first dimension varies the fastest, the last dimension varies the slowest.
   */
  struct F_stride_layout_str {
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = rect_str<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::none>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = F_stride_layout_str;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = F_layout_str;
  };

  /**
   * @brief Generic layout policy with arbitrary order and possible string indices.
   *
   * @tparam StaticExtent Compile-time known shape (zero if dynamic).
   * @tparam StrideOrder Order in which the dimensions are stored in memory.
   * @tparam LayoutProp Compile-time guarantees about the layout of the data in memory.
   */
  template <uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  struct basic_layout_str {
    // FIXME C++20 : StrideOrder will be a std::array<int, Rank> WITH SAME rank
    /// Multi-dimensional to flat index mapping.
    template <int Rank>
    using mapping = rect_str<Rank, StaticExtents, StrideOrder, LayoutProp>;

    /// The same layout policy, but with no guarantee of contiguity.
    using with_lowest_guarantee_t = basic_layout_str<StaticExtents, StrideOrder, layout_prop_e::none>;

    /// The same layout policy, but with guarantee of contiguity.
    using contiguous_t = basic_layout_str<StaticExtents, StrideOrder, layout_prop_e::contiguous>;
  };

  namespace detail {

    // Get the correct layout policy given a general nda::rect_str.
    template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
    struct layout_to_policy<rect_str<Rank, StaticExtents, StrideOrder, LayoutProp>> {
      using type = basic_layout_str<StaticExtents, StrideOrder, LayoutProp>;
    };

    // Get the correct layout policy given a general nda::rect_str.
    template <int Rank>
    struct layout_to_policy<rect_str<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>> {
      using type = C_layout_str;
    };

    // Get the correct layout policy given a strided nda::rect_str with C-order.
    template <int Rank>
    struct layout_to_policy<rect_str<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>> {
      using type = C_stride_layout_str;
    };

  } // namespace detail

  /** @} */

} // namespace nda
