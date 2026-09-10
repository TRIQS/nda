// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a lazy expression for advanced (NumPy-style) array indexing.
 */

#pragma once

#include "./concepts.hpp"
#include "./layout/bound_check_worker.hpp"
#include "./layout/for_each.hpp"
#include "./layout/range.hpp"
#include "./layout/slice_static.hpp"
#include "./macros.hpp"
#include "./mem/address_space.hpp"
#include "./stdutil/array.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nda {

  namespace detail {

    // Number of arguments that are index containers.
    template <typename... Args>
    constexpr size_t n_idx_containers = (size_t{0} + ... + static_cast<size_t>(IndexContainer<Args>));

    // Is the argument an ellipsis?
    template <typename T>
    constexpr bool is_ellipsis_arg = std::same_as<T, ellipsis>;

    // Position of the argument that applies to dimension n when calling an object of rank Rank with Args.
    // An ellipsis covers Rank - sizeof...(Args) + 1 dimensions.
    template <int Rank, typename... Args>
    constexpr int arg_pos(int n) {
      static_assert((static_cast<int>(is_ellipsis_arg<Args>) + ...) <= 1, "At most one ellipsis argument is allowed");
      constexpr int e_pos = slice_static::detail::ellipsis_position<Args...>();
      constexpr int e_len = Rank - static_cast<int>(sizeof...(Args)) + 1;
      return slice_static::detail::q_of_n(n, e_pos, e_len);
    }

    // Dimension that each dimension n becomes after calling an object of rank Rank with Args, or -1 if a long argument
    // removes it.
    template <int Rank, typename... Args>
    constexpr std::array<int, Rank> new_dim_of_n() {
      constexpr std::array<bool, sizeof...(Args)> is_long = {IndexType<Args>...};
      auto result                                         = std::array<int, Rank>{};
      for (int n = 0, p = 0; n < Rank; ++n) result[n] = is_long[arg_pos<Rank, Args...>(n)] ? -1 : p++;
      return result;
    }

    // Dimensions of the view obtained by calling an object of rank Rank with Args that are indexed by an IndexContainer.
    template <int Rank, typename... Args>
    constexpr auto indexed_dims_of() {
      constexpr std::array<bool, sizeof...(Args)> is_container = {IndexContainer<Args>...};
      constexpr auto new_dim                                   = new_dim_of_n<Rank, Args...>();
      auto result                                              = std::array<int, n_idx_containers<Args...>>{};
      for (int n = 0, c = 0; n < Rank; ++n)
        if (is_container[arg_pos<Rank, Args...>(n)]) result[c++] = new_dim[n];
      return result;
    }

    // Do the arguments access an element or slice an object of rank Rank: ranges, range::all, ellipsis or longs?
    template <int Rank, typename... Args>
    constexpr bool is_call_or_slice = ((is_range_or_ellipsis<Args> or IndexType<Args>) and ...)
       and (sizeof...(Args) == Rank or (ellipsis_is_present<Args...> and sizeof...(Args) <= Rank + 1));

    // Replace index containers with range::all for view construction.
    template <typename Arg>
    auto make_view_arg_for_indexed(Arg const &arg) {
      if constexpr (IndexContainer<Arg>) {
        return range::all;
      } else {
        return arg;
      }
    }

    // Convert the index containers among the arguments to an array of index lists (std::vector<long>).
    template <typename... Args>
    auto make_idx_lists(Args const &...args) {
      auto result = std::array<std::vector<long>, n_idx_containers<Args...>>{};
      size_t i    = 0;
      auto append = [&]<typename Arg>(Arg const &arg) {
        if constexpr (IndexContainer<Arg>) result[i++] = std::vector<long>(std::ranges::begin(arg), std::ranges::end(arg));
      };
      (append(args), ...);
      return result;
    }

  } // namespace detail

  /**
   * @addtogroup av_utils
   * @{
   */

  /**
   * @brief Lazy expression for advanced array indexing with arbitrary index containers.
   *
   * @details This expression type enables NumPy-style advanced indexing where arrays can be
   * indexed with containers of arbitrary integers (e.g., `std::vector<long>`). Unlike regular
   * slicing which returns a view, advanced indexing returns a lazy expression because the
   * selected elements may not be contiguous in memory.
   *
   * The expression stores:
   * - The source array with any range-based slicing already applied. For lvalue arrays and for views this is a
   *   borrowed view. For rvalue arrays with heap storage it is a view with an nda::shared owning policy that takes over
   *   the memory of the temporary (no copy). Only for rvalue arrays with other storage (e.g. stack or SSO) it is a copy
   *   of the sliced array.
   * - One index list (`std::vector<long>`) per indexed dimension, copied from the index containers
   *
   * Example usage:
   * @code{.cpp}
   * nda::array<double, 2> arr(10, 20);
   * std::vector<long> indices = {3, 7, 1};
   * auto expr = arr(indices, nda::range::all);  // Returns expr_indexed
   * nda::array<double, 2> result = expr;        // Materializes the expression
   * @endcode
   *
   * @tparam A Type of the internal array: a `basic_array_view` with a borrowed or shared owning policy, or a
   * `basic_array` if the source was an rvalue with non-heap storage.
   * @tparam IndexedDims `std::array<int, N>` with the dimensions of the array indexed by the containers.
   */
  template <typename A, auto IndexedDims>
  struct expr_indexed {
    /// Number of index lists.
    static constexpr size_t n_idx_lists = IndexedDims.size();
    static_assert(n_idx_lists > 0, "expr_indexed requires at least one index container");
    static_assert(mem::on_host<A>, "expr_indexed only supports arrays in the Host address space");

    /// Internal array/view of the source data.
    A a;

    /// Index lists, one per indexed dimension.
    std::array<std::vector<long>, n_idx_lists> idx_lists;

    /// Dimensions of the array indexed by the index lists.
    static constexpr std::array<int, n_idx_lists> indexed_dims = IndexedDims;

    /// Rank of the expression (same as underlying array).
    static constexpr int rank = get_rank<A>;

    /// Value type of the underlying array.
    using value_type = typename A::value_type;

    /**
     * @brief Construct from an array/view and the index lists.
     * @param a Array/view of the source data.
     * @param idx_lists Index lists, one per indexed dimension.
     */
    expr_indexed(A a, std::array<std::vector<long>, n_idx_lists> idx_lists) : a(std::move(a)), idx_lists(std::move(idx_lists)) {}

    // Copy and move construction rebind the expression, assignment (see below) writes through to the elements, like
    // for nda::basic_array_view.

    /// Default copy constructor.
    expr_indexed(expr_indexed const &) = default;

    /// Default move constructor.
    expr_indexed(expr_indexed &&) = default;

    private:
    // Position of the index list for dimension d, or -1 if there is none.
    static constexpr int idx_list_of_dim(int d) {
      for (size_t c = 0; c < n_idx_lists; ++c)
        if (indexed_dims[c] == d) return static_cast<int>(c);
      return -1;
    }

    // Is dimension d indexed by an index list?
    static constexpr bool has_idx_list(int d) { return idx_list_of_dim(d) >= 0; }

    // Map an index along dimension D through its index list, if any.
    template <int D>
    [[nodiscard]] long resolve_index(long i) const {
      if constexpr (not has_idx_list(D)) {
        return i;
      } else {
        auto const &lst = idx_lists[idx_list_of_dim(D)];
#ifdef NDA_ENFORCE_BOUNDCHECK
        auto len = std::ssize(lst);
        assert_in_bounds(1, &len, i);
#endif
        return lst[i];
      }
    }

    // Restrict an index list to a range. range::all leaves it unchanged; this overload also takes an ellipsis, which
    // derives from range::all_t.
    static std::vector<long> slice_idx_list(std::vector<long> const &lst, range const &rg) {
      auto len = std::ssize(lst);
#ifdef NDA_ENFORCE_BOUNDCHECK
      assert_in_bounds(1, &len, rg);
#endif
      auto n      = slice_static::detail::get_length(rg, len);
      auto result = std::vector<long>{};
      result.reserve(n);
      for (long k = 0; k < n; ++k) result.push_back(lst[rg.first() + k * rg.step()]);
      return result;
    }
    static std::vector<long> slice_idx_list(std::vector<long> const &lst, range::all_t) { return lst; }

    // Index lists that survive a call with Args (i.e. not consumed by a long argument): their positions in idx_lists
    // and the dimensions they index in the sliced array.
    template <typename... Args>
    struct remaining_idx_lists {
      static constexpr auto new_dim = detail::new_dim_of_n<rank, Args...>();
      static constexpr size_t size  = std::ranges::count_if(indexed_dims, [](int d) { return new_dim[d] >= 0; });
      static constexpr auto pos     = [] {
        auto result = std::array<int, size>{};
        for (size_t c = 0, k = 0; c < n_idx_lists; ++c)
          if (new_dim[indexed_dims[c]] >= 0) result[k++] = static_cast<int>(c);
        return result;
      }();
      static constexpr auto dims = [] {
        auto result = std::array<int, size>{};
        for (size_t k = 0; k < size; ++k) result[k] = new_dim[indexed_dims[pos[k]]];
        return result;
      }();
    };

    // Argument passed to the array for dimension D in a block-wise assignment: a long on indexed dimensions, range::all
    // otherwise.
    template <int D>
    using block_arg_t = std::conditional_t<has_idx_list(D), long, range::all_t>;

    // Can RHS be called with the arguments of a block-wise assignment?
    template <typename RHS>
    static constexpr bool can_assign_blocks = []<size_t... Ds>(std::index_sequence<Ds...>) {
      return requires(RHS const &rhs, block_arg_t<static_cast<int>(Ds)> const &...args) { rhs(args...); };
    }(std::make_index_sequence<rank>{});

    // Call f once per combination of positions in the index lists with one argument per dimension: the position on
    // indexed dimensions and range::all otherwise. Calling the expression or a same-shaped array with these arguments
    // yields the block of the remaining dimensions.
    template <typename F>
    void for_each_block(F &&f) const {
      auto idx_shape = std::array<long, n_idx_lists>{};
      for (size_t c = 0; c < n_idx_lists; ++c) idx_shape[c] = std::ssize(idx_lists[c]);
      nda::for_each(idx_shape, [&f](auto... is) {
        auto is_tuple  = std::tie(is...);
        auto block_arg = [&is_tuple]<int D>() -> block_arg_t<D> {
          if constexpr (has_idx_list(D)) {
            return std::get<idx_list_of_dim(D)>(is_tuple);
          } else {
            return range::all;
          }
        };
        [&]<size_t... Ds>(std::index_sequence<Ds...>) {
          f(block_arg.template operator()<static_cast<int>(Ds)>()...);
        }(std::make_index_sequence<rank>{});
      });
    }

    // Element access and slicing implementation, see operator() for the semantics.
    template <typename Self, typename... Args>
    [[nodiscard]] static decltype(auto) slice_impl(Self &self, Args const &...args) {
      auto args_tuple = std::tie(args...);

      // Argument passed on to the array for dimension D: a long resolves through the index list (if any) and removes
      // the dimension, an ellipsis becomes range::all, a range on an indexed dimension is applied to the index list
      // instead (below) and everything else is forwarded.
      auto array_arg = [&]<int D, typename Arg>(Arg const &arg) {
        if constexpr (IndexType<Arg>) {
          return self.template resolve_index<D>(arg);
        } else if constexpr (has_idx_list(D) or detail::is_ellipsis_arg<Arg>) {
          return range::all;
        } else {
          return arg;
        }
      };
      decltype(auto) sliced = [&]<size_t... Ds>(std::index_sequence<Ds...>) -> decltype(auto) {
        return self.a(array_arg.template operator()<static_cast<int>(Ds)>(std::get<detail::arg_pos<rank, Args...>(Ds)>(args_tuple))...);
      }(std::make_index_sequence<rank>{});

      using remaining = remaining_idx_lists<Args...>;
      if constexpr (remaining::size == 0) {
        return sliced;
      } else {
        // slice the index list at position C with the argument of its dimension
        auto sliced_idx_list = [&]<int C>() {
          return slice_idx_list(self.idx_lists[C], std::get<detail::arg_pos<rank, Args...>(indexed_dims[C])>(args_tuple));
        };
        return [&]<size_t... Ks>(std::index_sequence<Ks...>) {
          return expr_indexed<decltype(sliced), remaining::dims>{std::move(sliced), {sliced_idx_list.template operator()<remaining::pos[Ks]>()...}};
        }(std::make_index_sequence<remaining::size>{});
      }
    }

    public:
    /**
     * @brief Get the shape of the expression.
     * @return `std::array<long, rank>` specifying the shape.
     */
    [[nodiscard]] std::array<long, rank> shape() const {
      auto result = a.shape();
      for (size_t c = 0; c < n_idx_lists; ++c) result[indexed_dims[c]] = std::ssize(idx_lists[c]);
      return result;
    }

    /**
     * @brief Get the total size of the expression.
     * @return Number of elements.
     */
    [[nodiscard]] long size() const { return stdutil::product(shape()); }

    /**
     * @brief Copy assignment operator assigns the elements of the right hand side to the indexed elements.
     * @param rhs Source expression to assign from.
     * @return Reference to this expression.
     */
    expr_indexed &operator=(expr_indexed const &rhs) { return operator= <expr_indexed>(rhs); }

    /// Move assignment operator, same as copy assignment.
    expr_indexed &operator=(expr_indexed &&rhs) { return operator= <expr_indexed>(rhs); }

    /**
     * @brief Assignment operator from an array or expression.
     *
     * @details The assignment loops over the indexed dimensions only and assigns the block of the remaining dimensions
     * as a view, so that the optimized copy of nda::basic_array_view is used. If the right hand side cannot be sliced
     * with `long` and `range::all` arguments, the assignment is done element by element.
     *
     * @tparam RHS Type satisfying ArrayOfRank<rank>.
     * @param rhs Source array/expression to assign from.
     * @return Reference to this expression.
     */
    template <ArrayOfRank<rank> RHS>
    expr_indexed &operator=(RHS const &rhs) {
      EXPECTS(shape() == rhs.shape());
      auto assign = [this, &rhs](auto const &...args) { (*this)(args...) = rhs(args...); };
      if constexpr (can_assign_blocks<RHS>) {
        for_each_block(assign);
      } else {
        nda::for_each(shape(), assign);
      }
      return *this;
    }

    /**
     * @brief Assignment operator from a scalar value.
     *
     * @details Loops over the indexed dimensions only and fills the block of the remaining dimensions.
     *
     * @tparam RHS Scalar type.
     * @param rhs Scalar value to assign to all elements.
     * @return Reference to this expression.
     */
    template <typename RHS>
      requires(is_scalar_for_v<RHS, expr_indexed>)
    expr_indexed &operator=(RHS const &rhs) {
      for_each_block([this, &rhs](auto const &...args) { (*this)(args...) = rhs; });
      return *this;
    }

    /**
     * @brief Function call operator for element access and slicing.
     *
     * @details Takes the same arguments as an nda::basic_array_view: `range`, `range::all`, `ellipsis` and `long`.
     * A `long` removes the dimension; on an indexed dimension the index is first mapped through the index list, which
     * is then dropped. Ranges on an indexed dimension select a subset of the index list, on other dimensions they are
     * forwarded to the underlying array.
     *
     * @tparam Args Argument types.
     * @param args Indices or slicing arguments.
     * @return A new expr_indexed with the remaining index lists, or the sliced view or element reference if no
     * index list remains.
     */
    template <typename... Args>
      requires(detail::is_call_or_slice<rank, Args...>)
    [[nodiscard]] decltype(auto) operator()(Args const &...args) {
      return slice_impl(*this, args...);
    }

    /// Const overload.
    template <typename... Args>
      requires(detail::is_call_or_slice<rank, Args...>)
    [[nodiscard]] decltype(auto) operator()(Args const &...args) const {
      return slice_impl(*this, args...);
    }
  };

  /**
   * @brief Get the algebra of an expr_indexed (inherits from the underlying array).
   * @tparam A Array type.
   * @tparam IndexedDims Indexed dimensions.
   */
  template <typename A, auto IndexedDims>
  inline constexpr char get_algebra<expr_indexed<A, IndexedDims>> = get_algebra<A>;

  /// Specialization of nda::is_expression for nda::expr_indexed types.
  template <typename A, auto IndexedDims>
  inline constexpr bool is_expression<expr_indexed<A, IndexedDims>> = true;

  /** @} */

  namespace mem {

    /// Specialization of nda::mem::get_addr_space for nda::expr_indexed types.
    template <typename A, auto IndexedDims>
    static constexpr AddressSpace get_addr_space<expr_indexed<A, IndexedDims>> = get_addr_space<A>;

  } // namespace mem

} // namespace nda
