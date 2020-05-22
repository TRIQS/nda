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

  template <int Rank>
  constexpr uint64_t Fortran_stride_order = nda::encode(nda::permutations::reverse_identity<Rank>());

  template <int Rank>
  constexpr uint64_t C_stride_order = nda::encode(nda::permutations::identity<Rank>());

  // constexpr std::array<long,  > ce_len, ce_stri :   -1 -->  dynamic.
  // https://godbolt.org/z/qmKWpj

  // -----------------------------------------------------------------------------------
  /**
   *
   * The map of the indices to linear index.
   *
   * It is a set of lengths and strides for each dimensions, and a shift.
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
    std::array<long, Rank> len, str;               // lenghts and strides

    public:
    static constexpr uint64_t static_extents_encoded      = StaticExtents;
    static constexpr std::array<int, Rank> static_extents = decode<Rank>(StaticExtents);

    static constexpr std::array<int, Rank> stride_order =
       (StrideOrder == 0 ? permutations::identity<Rank>() : decode<Rank>(StrideOrder)); // 0 is C stride_order
    static constexpr uint64_t stride_order_encoded = encode(stride_order);

    static constexpr layout_prop_e layout_prop = LayoutProp;
    static constexpr layout_info_t layout_info = layout_info_t{stride_order_encoded, layout_prop};

    private:
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
    [[nodiscard]] long size() const noexcept { return std::accumulate(len.cbegin(), len.cend(), 1, std::multiplies<>{}); }

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

    /// Is the data contiguous in memory ? [NB recomputed at each call]
    [[nodiscard]] bool is_contiguous() const noexcept {
      int slowest_index = std::distance(str.begin(), std::max_element(str.begin(), str.end())); // index with minimal stride
      return (str[slowest_index] * len[slowest_index] == size());
    }

    /// Is the order in memory C ?
    static constexpr bool is_stride_order_C() {
      // operator == of std:array is constexpr only since C++20
#if __cplusplus > 201703L
      return (stride_order == permutations::identity<Rank>());
#else
      return (encode(stride_order) == encode(permutations::identity<Rank>()));
#endif
    }

    /// Is the order in memory Fortran ?
    static constexpr bool is_stride_order_Fortran() {
#if __cplusplus > 201703L
      return (stride_order == permutations::reverse_identity<Rank>());
#else
      return (encode(stride_order) == encode(permutations::reverse_identity<Rank>()));
#endif
    }

    /// Value of the minimum stride (i.e the fastest one)
    [[nodiscard]] long min_stride() const noexcept { return str[stride_order[Rank - 1]]; }

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
     * From an idxmap with other info flags
     * @param idxm
     */
    template <layout_prop_e P>
    idx_map(idx_map<Rank, StaticExtents, StrideOrder, P> const &idxm) noexcept : len(idxm.lengths()), str(idxm.strides()) {
      static_assert(is_degradable(P, LayoutProp), "Can not construct the view: it would violate some compile time guarantees about the layout");
    }

    private:
    void assert_static_extents_and_len_are_compatible() const {
#ifdef NDA_DEBUG
      if constexpr (n_dynamic_extents != Rank) { // there are some static extents
        for (int u = 0; u < Rank; ++u)
          if (static_extents[u] != 0) EXPECTS(static_extents[u] == len[u]);
      }
#endif
    }
    
    public:

    /// Construct from a compatible static_extents
    template <uint64_t SE, layout_prop_e P>
    idx_map(idx_map<Rank, SE, StrideOrder, P> const &idxm) noexcept(false) : len(idxm.lengths()), str(idxm.strides()) { // can throw
      static_assert(is_degradable(P, LayoutProp), "Can not construct the view: it would violate some compile time guarantees about the layout");
      assert_static_extents_and_len_are_compatible();
    }

    ///
    idx_map(std::array<long, Rank> const &shape, std::array<long, Rank> const &strides) noexcept : len(shape), str(strides) {}

    ///
    idx_map(std::array<long, Rank> const &shape) noexcept : len(shape) {
      assert_static_extents_and_len_are_compatible();
      compute_strides_contiguous();
    }

    private:
    std::array<long, Rank> _embed_array(std::array<long, n_dynamic_extents> const &s) {
      std::array<long, Rank> r;
      for (int u = 0, v = 0; u < Rank; ++u) r[u] = (static_extents[u] == 0 ? s[v++] : static_extents[u]);
      return r;
    }

    public:
    /** 
     * Construction from the lengths, the strides
     * @param lengths
     * @param strides
     */
    idx_map(std::array<long, n_dynamic_extents> const &lengths) noexcept REQUIRES((n_dynamic_extents != Rank) and (n_dynamic_extents != 0))
       : idx_map(_embed_array(lengths)) {}

    /// \private
    /// trap for error. If one tries to construct a view with a mismatch of stride order
    template <uint64_t StaticExtents2, uint64_t StrideOrder2, layout_prop_e P>
    idx_map(idx_map<Rank, StaticExtents2, StrideOrder2, P> const &) REQUIRES((StaticExtents2 != StaticExtents) or (StrideOrder != StrideOrder2)) {
      static_assert(not((StaticExtents2 != StaticExtents) or (StrideOrder != StrideOrder2)),
                    "Can not construct a layout from another one with a different stride order");
    }

    /// \private
    /// trap for incorrect calls. For R = Rank, the non template has priority
    template <int R>
    idx_map(std::array<long, R> const &) {
      static_assert(R == Rank, "Rank of the argument incorrect in idx_map construction");
    }

    // ----------------  Call operator -------------------------

    private:
    template <size_t Is>
    [[nodiscard]] FORCEINLINE long __get(long arg) const noexcept {
      if constexpr (Is == stride_order[Rank - 1])
        return arg;
      else
        return arg * std::get<Is>(str);
    }

    // call implementation
    template <typename... Args, size_t... Is>
    [[nodiscard]] FORCEINLINE long call_impl(std::index_sequence<Is...>, Args... args) const noexcept {
      if constexpr (LayoutProp & layout_prop_e::smallest_stride_is_one)
        return (__get<Is>(args) + ...);
      else
        return ((args * std::get<Is>(str)) + ...);
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
    FORCEINLINE long operator()(Args const &... args) const
#ifdef NDA_ENFORCE_BOUNDCHECK
       noexcept(false) {
      details::assert_in_bounds(rank(), len.data(), args...);
#else
       noexcept(true) {
#endif
      return call_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
    }

    // ----------------  Comparison -------------------------
    bool operator==(idx_map const &x) const { return (len == x.len) and (str == x.str); }

#if not __cplusplus > 201703L
    bool operator!=(idx_map const &x) { return !(operator==(x)); }
#endif

  }; // idx_map class

  //// DEBUG
  //template <int Rank, uint64_t SO1, layout_prop_e LP1,  uint64_t SO2, layout_prop_e LP2>
  //bool operator ==(idx_map<Rank, SO1, LP1> const & i1, idx_map<Rank, SO2, LP2> const & i2) {
  //if constexpr((LP1 != LP2) or ((SO1 != SO2)) return false;
  //else

  // ---------------- Transposition -------------------------

  // FIXME : why is Permutation template  in apply_invers ? constexpr is enough ??
  template <ARRAY_INT Permutation, int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  auto transpose(idx_map<Rank, StaticExtents, StrideOrder, LayoutProp> const &idx) {

    using idx_t = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>;

    static constexpr std::array<int, Rank> new_stride_order   = permutations::apply_inverse<Permutation>(idx_t::stride_order);
    static constexpr std::array<int, Rank> new_static_extents = permutations::apply_inverse<Permutation>(idx_t::static_extents);

    // compute the new layout_info...
    // strided_1d does not change, but min_stride_is_1 can !
    static constexpr layout_prop_e new_layout = (idx_t::layout_prop & layout_prop_e::strided_1d ? layout_prop_e::strided_1d : layout_prop_e::none);

    return idx_map<Rank, encode(new_static_extents), encode(new_stride_order), new_layout>{permutations::apply_inverse<Permutation>(idx.lengths()),
                                                                                           permutations::apply_inverse<Permutation>(idx.strides())};
  }

} // namespace nda
