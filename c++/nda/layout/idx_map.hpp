// Copyright (c) 2018-2021 Simons Foundation
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
#include <vector>
#include <tuple>
#include <algorithm>
#include <array>
#include <numeric>

#include "../macros.hpp"
#include "../traits.hpp"
#include "./range.hpp"
#include "./bound_check_worker.hpp"
#include "./for_each.hpp"

namespace nda {

  // forward
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map;

  namespace slice_static {
    template <int R, uint64_t SE, uint64_t SO, layout_prop_e LP, typename... T>
    FORCEINLINE decltype(auto) slice_stride_order(idx_map<R, SE, SO, LP> const &idxm, T const &...x);
  }
  // end forward

  template <int Rank>
  constexpr uint64_t Fortran_stride_order = nda::encode(nda::permutations::reverse_identity<Rank>());

  template <int Rank>
  constexpr uint64_t C_stride_order = nda::encode(nda::permutations::identity<Rank>());

  // -----------------------------------------------------------------------------------
  /**
   *
   * The layout that maps the indices to linear index.
   *
   * Basically lengths and strides for each dimensions
   *
   * @tparam Rank : rank of the index map
   * 
   * @tparam StaticExtents : encoded std::array{0, d1, 0, d3}
   *   where d1, d3 are static dimensions for index 1,3
   *         NB Limitation : d1, d3 < 16 (until C++20)
   *         0 mean dynamic dimension
   *   NB : if StaticExtents ==0, it means all dimensions are static
   *
   * @tparam StrideOrder : a permutation for the memory stride_order of the array
   *    
   *    StrideOrder[0] : the slowest index,
   *    StrideOrder[Rank-1] : the fastest index
   *    Example :
   *        012 : C the last index is the fastest
   *        210 : Fortran, the first index is the fastest
   *        120 : storage (i,j,k) is : index j is slowest, then k, then i
   *    
   *    NB : StrideOrder = 0 is the default and it is means 0 order
   * 
   * @tparam LayoutProp : A flags of compile time guarantees for the layout
   *
   */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map {
    static_assert(Rank < 16, "Rank must be < 16"); // C++17 constraint. Relax this in C++20
    static_assert((StrideOrder != 0) or (Rank == 1), "Oops");
    std::array<long, Rank> len, str; // lenghts and strides

    public:
    static constexpr uint64_t static_extents_encoded      = StaticExtents;
    static constexpr std::array<int, Rank> static_extents = decode<Rank>(StaticExtents);

    static constexpr std::array<int, Rank> stride_order =
       (StrideOrder == 0 ? permutations::identity<Rank>() : decode<Rank>(StrideOrder)); // 0 is C stride_order
    static constexpr uint64_t stride_order_encoded = encode(stride_order);

    static constexpr layout_prop_e layout_prop = LayoutProp;
    static constexpr layout_info_t layout_info = layout_info_t{stride_order_encoded, layout_prop};

    template <typename T>
    static constexpr int argument_is_allowed_for_call = std::is_constructible_v<long, T>;

    template <typename T>
    static constexpr int argument_is_allowed_for_call_or_slice =
       std::is_same_v<range, T> or std::is_same_v<range::all_t, T> or std::is_same_v<ellipsis, T> or std::is_constructible_v<long, T>;

    protected:
    static constexpr int n_dynamic_extents = []() {
      int r = 0;
      for (int u = 0; u < Rank; ++u) r += (static_extents[u] == 0 ? 1 : 0);
      return r;
    }();

    public:
    // ----------------  Accessors -------------------------

    /// Rank of the map (number of arguments)
    static constexpr int rank() noexcept { return Rank; }

    /// Total number of elements (products of lengths in each dimension).
    // NB recomputed at each call (FIXME Optimize this ?)
    [[nodiscard]] long size() const noexcept { return std::accumulate(len.cbegin(), len.cend(), 1L, std::multiplies<>{}); }

    /// Compile time size, 0 means "dynamical"
    static constexpr long ce_size() noexcept {
      if constexpr (n_dynamic_extents != 0) { // quick general case
        return 0;
      } else {
        long s = 1;
        for (int u = 0; u < Rank; ++u) s *= static_extents[u];
        return s;
      }
    }

    /// Lengths of each dimension.
    [[nodiscard]] std::array<long, Rank> const &lengths() const noexcept { return len; }

    /// Strides of each dimension.
    [[nodiscard]] std::array<long, Rank> const &strides() const noexcept { return str; }

    /// Value of the minimum stride (i.e the fastest one)
    [[nodiscard]] long min_stride() const noexcept { return str[stride_order[Rank - 1]]; }

    /// Is the data contiguous in memory ? [NB recomputed at each call]
    [[nodiscard]] bool is_contiguous() const noexcept {
      int slowest_index = std::distance(str.begin(), std::max_element(str.begin(), str.end())); // index with minimal stride
      return (str[slowest_index] * len[slowest_index] == size());
    }

    /// Is the data strided 1d in memory ? [NB recomputed at each call]
    [[nodiscard]] bool is_strided_1d() const noexcept {
      int slowest_index = std::distance(str.begin(), std::max_element(str.begin(), str.end())); // index with minimal stride
      return (str[slowest_index] * len[slowest_index] == size() * min_stride());
    }

    /// Is the order in memory C ?
    static constexpr bool is_stride_order_C() {
      // operator == of std:array is constexpr only since C++20
      return (stride_order == permutations::identity<Rank>());
      //return (encode(stride_order) == encode(permutations::identity<Rank>()));
    }

    /// Is the order in memory Fortran ?
    static constexpr bool is_stride_order_Fortran() {
      return (stride_order == permutations::reverse_identity<Rank>());
      //return (encode(stride_order) == encode(permutations::reverse_identity<Rank>()));
    }

    // ----------------  Constructors -------------------------

    private:
    // compute strides for a contiguous array from len
    void compute_strides_contiguous() {
      long s = 1;
      for (int v = rank() - 1; v >= 0; --v) { // rank() is constexpr, allowing compiler to transform loop...
        int u  = stride_order[v];
        str[u] = s;
        s *= len[u];
      }
      ENSURES(s == size());
    }

    // FIXME ADD A CHECK layout_prop_e ... compare to stride and

    public:
    /// Default constructor. Strides are not initiliazed.
    idx_map() {
      if constexpr (n_dynamic_extents == 0) { // full static array
        for (int u = 0; u < Rank; ++u) len[u] = static_extents[u];
        compute_strides_contiguous();
      } else {
        for (int u = 0; u < Rank; ++u)
          len[u] = 0; // FIXME. Needed ? To have the proper invariant of the array : shape = (0,0,...) and pointer is null
      }
    }

    idx_map(idx_map const &) = default;
    idx_map(idx_map &&)      = default;
    idx_map &operator=(idx_map const &) = default;
    idx_map &operator=(idx_map &&) = default;

    /**
     * Check if the strides of all non-trivial dimensions (l[d] > 1)
     * respect the stride_order
     * @param lenptr Pointer to the lengths
     * @param strptr Pointer to the strides
     */
    template <std::integral Int>
    [[nodiscard]] static bool is_stride_order_valid(Int *lenptr, Int *strptr) {
      auto dims_to_check = std::vector<int>{};
      dims_to_check.reserve(Rank);
      for (auto dim : stride_order)
        if (lenptr[dim] > 1) dims_to_check.push_back(dim);

      for (int n = 1; n < dims_to_check.size(); ++n)
        if (strptr[dims_to_check[n - 1]] < strptr[dims_to_check[n]]) return false;

      return true;
    }

    [[nodiscard]] bool is_stride_order_valid() const { return is_stride_order_valid(len.data(), str.data()); }

    /** 
     * From an idxmap with other info flags
     * @param idxm
     */
    template <layout_prop_e P>
    idx_map(idx_map<Rank, StaticExtents, StrideOrder, P> const &idxm) noexcept : len(idxm.lengths()), str(idxm.strides()) {
      EXPECTS(is_stride_order_valid());
      if constexpr (not layout_property_compatible(P, LayoutProp)) {
        if constexpr (has_contiguous(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(
             idxm.is_contiguous(),
             "Failed check of contiguity. Constructing a contiguous layout from another layout which was not guaranteed to be contiguous at compile time. The check fails so your program is incorrect");
        }
        if constexpr (has_strided_1d(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(
             idxm.is_strided_1d(),
             "Failed check of quasi-contiguity (1d-strided). Constructing a contiguous layout from another layout which was not guaranteed to be quasi-contiguous at compile time. The check fails so your program is incorrect");
        }
      }
    }

    private:
    void assert_static_extents_and_len_are_compatible() const {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if constexpr (n_dynamic_extents != Rank) { // there are some static extents
// to avoid warning
#ifndef NDEBUG
        for (int u = 0; u < Rank; ++u)
          if (static_extents[u] != 0) EXPECTS(static_extents[u] == len[u]);
#endif
      }
#endif
    }

    public:
    /// Construct from a compatible static_extents
    template <uint64_t SE, layout_prop_e P>
    idx_map(idx_map<Rank, SE, StrideOrder, P> const &idxm) noexcept(false) : len(idxm.lengths()), str(idxm.strides()) { // can throw
      EXPECTS(is_stride_order_valid());
      if constexpr (not layout_property_compatible(P, LayoutProp)) {
        if constexpr (has_contiguous(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(
             idxm.is_contiguous(),
             "Failed check of contiguity. Constructing a contiguous layout from another layout which was not guaranteed to be contiguous at compile time. The check fails so your program is incorrect");
        }
        if constexpr (has_strided_1d(LayoutProp)) {
          EXPECTS_WITH_MESSAGE(
             idxm.is_strided_1d(),
             "Failed check of quasi-contiguity (1d-strided). Constructing a contiguous layout from another layout which was not guaranteed to be quasi-contiguous at compile time. The check fails so your program is incorrect");
        }
      }
      assert_static_extents_and_len_are_compatible();
    }

    private:
#ifdef NDA_DEBUG
    static constexpr bool check_stride_order = true;
#else
    static constexpr bool check_stride_order = false;
#endif

    public:
    idx_map(std::array<long, Rank> const &shape, std::array<long, Rank> const &strides) noexcept(!check_stride_order) : len(shape), str(strides) {
      if constexpr (check_stride_order)
        if (not is_stride_order_valid())
          throw std::runtime_error("ERROR: strides of idx_map do not match stride order of the type\n");
    }

    /// Construct from the shape. If StaticExtents are present, the corresponding component of the shape must be equal to it.
    template <std::integral Int = long>
    idx_map(std::array<Int, Rank> const &shape) noexcept : len(stdutil::make_std_array<long>(shape)) {
      assert_static_extents_and_len_are_compatible();
      compute_strides_contiguous();
    }

    private:
    static std::array<long, Rank> merge_static_and_dynamic_extents(std::array<long, n_dynamic_extents> const &s) {
      std::array<long, Rank> r;
      for (int u = 0, v = 0; u < Rank; ++u) r[u] = (static_extents[u] == 0 ? s[v++] : static_extents[u]);
      return r;
    }

    public:
    /// When StaticExtents are present, constructs from the dynamic extents only
    idx_map(std::array<long, n_dynamic_extents> const &shape) noexcept requires((n_dynamic_extents != Rank) and (n_dynamic_extents != 0))
       : idx_map(merge_static_and_dynamic_extents(shape)) {}

    /// \private
    /// trap for error. If one tries to construct a view with a mismatch of stride order
    // The compiler selects this constructor instead of presenting a long list, and then goes into a dead end.
    template <uint64_t StaticExtents2, uint64_t StrideOrder2, layout_prop_e P>
    idx_map(idx_map<Rank, StaticExtents2, StrideOrder2, P> const &) requires(StrideOrder != StrideOrder2) {
      static_assert((StrideOrder == StrideOrder2), "Can not construct a layout from another one with a different stride order");
    }

    /// \private
    /// trap for error. For R = Rank, the non template has priority
    template <int R>
    idx_map(std::array<long, R> const &) requires(R != Rank) {
      static_assert(R == Rank, "Rank of the argument incorrect in idx_map construction");
    }

    // ----------------  Call operator -------------------------
    private:
    template <bool skip_stride, auto Is>
    [[nodiscard]] FORCEINLINE long myget(ellipsis) const noexcept {
      return 0;
    }

    template <bool skip_stride, auto Is>
    [[nodiscard]] FORCEINLINE long myget(long arg) const noexcept {
      if constexpr (skip_stride and (Is == stride_order[Rank - 1])) // this is the slowest stride
        return arg;
      else
        return arg * std::get<Is>(str);
    }

    static constexpr bool smallest_stride_is_one = has_smallest_stride_is_one(LayoutProp);

    // call implementation
    template <typename... Args, size_t... Is>
    [[nodiscard]] FORCEINLINE long call_impl(std::index_sequence<Is...>, Args... args) const noexcept {
      static constexpr int e_pos = ((std::is_same_v<Args, ellipsis> ? int(Is) + 1 : 0) + ...) - 1; // position ellipsis + 1 or 0

      if constexpr (e_pos == -1) { // common case, no ellipsis
        if constexpr (smallest_stride_is_one)
          return (myget<true, Is>(args) + ...);
        else
          return ((args * std::get<Is>(str)) + ...);
      } else {
        // there is an empty ellipsis to skip
        return (myget<smallest_stride_is_one, (Is < e_pos ? Is : Is - 1)>(args) + ...);
      }
    }

    public:
    /**
     * Number of variables must be exactly the rank or are optionally
     * checked at runtime
     *
     * @return : 
     *       the linear position 
     *
     */
    template <typename... Args>
    FORCEINLINE long operator()(Args const &...args) const
#ifdef NDA_ENFORCE_BOUNDCHECK
       noexcept(false) {
      details::assert_in_bounds(rank(), len.data(), args...);
#else
       noexcept(true) {
#endif
      // there may be an empty ellipsis which we will need to skip. e_pos = 128 if no ellipsis
      return call_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
    }

    // ----------------  Slice -------------------------

    template <typename... Args>
    auto slice(Args const &...args) const {
      return slice_static::slice_stride_order(*this, args...);
    }

    // ----------------  Comparison -------------------------

    bool operator==(idx_map const &x) const = default;

    // ---------------- Transposition -------------------------

    /**
     * Makes a new transposed idx_map with permutation P such that 
     * denoting here  A = this, A' = P A = returned_value
     * A'(i_k) = A(i_{P[k]})
     *
     * Note that this convention is the correct one to have a (left) action of the symmetric group on a array
     * and it may not be completely obvious.
     * Proof
     *  let's operate with P then Q, and denote A'' = Q A'. We want to show that A'' = (QP) A
     *   A'(i_k) = A(i_{P[k]})
     *   A''(j_k) = A'(j_{Q[k]})
     *   then i_k = j_{Q[k]} and  A''(j_k) =  A(i_{P[k]}) = A(j_{Q[P[k]]}) = A(j_{(QP)[k]}), q.e.d
     *
     * NB test will test this composition
     */
    template <uint64_t Permutation>
    auto transpose() const {
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
      //    NB : Q and P are permutations, so the operation must be a composition, not an apply (apply applies a P to any set, like L, S, not only a permutation)
      //    even though they are all std::array in the code ...
      //
      static constexpr std::array<int, Rank> permu              = decode<Rank>(Permutation);
      static constexpr std::array<int, Rank> new_stride_order   = permutations::compose(permu, stride_order);
      static constexpr std::array<int, Rank> new_static_extents = permutations::apply_inverse(permu, static_extents);

      // FIXME
      // Compute the new layout_prop of the new view
      // NB : strided_1d property is preserved, but smallest_stride_is_one is not
      static constexpr layout_prop_e new_layout_prop =
         layout_prop_e::none; // BUT FIX (has_strided_1d(layout_prop) ? layout_prop_e::strided_1d : layout_prop_e::none);

      return idx_map<Rank, encode(new_static_extents), encode(new_stride_order), new_layout_prop>{permutations::apply_inverse(permu, lengths()),
                                                                                                  permutations::apply_inverse(permu, strides())};
    }

  }; // idx_map class

} // namespace nda
