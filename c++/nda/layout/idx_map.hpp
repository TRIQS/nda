#pragma once
#include <vector>
#include <tuple>
#include <algorithm>
#include <array>
#include <numeric>

#include "../macros.hpp"
#include "../traits.hpp"

// FIXME : move it in decl.hpp
namespace nda {

  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map;
}

#include "./range.hpp"
#include "./bound_check_worker.hpp"
#include "./slice_static.hpp"
#include "./for_each.hpp"
#include "./permutation.hpp"

namespace boost::serialization {
  class access;
}

namespace nda {

  /// Type of all shapes
  template <int Rank>
  using shape_t = std::array<long, Rank>;

  /// Shape factory
  template <typename... T>
  shape_t<sizeof...(T)> make_shape(T... args) noexcept {
    return {args...};
  }

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
   */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map {

    static_assert(Rank < 64, "Rank must be < 64"); // constraint of slice implementation. ok...

    std::array<long, Rank> len, str;

    public:
    // FIXME : rename DECODE : it is not a permutation
    static constexpr std::array<int, Rank> static_extents = decode<Rank>(StaticExtents);
    static constexpr uint64_t static_extents_encoded      = StaticExtents;

    static constexpr int rank_dynamic = []() {
      int r = 0;
      for (int u = 0; u < Rank; ++u) r += (static_extents[u] == 0 ? 1 : 0);
      return r;
    }();

    // main property : idx_map<Rank, stride_order_encoded, layout_prop> is THIS
    static constexpr std::array<int, Rank> stride_order =
       (StrideOrder == 0 ? permutations::identity<Rank>() : decode<Rank>(StrideOrder)); // 0 is C stride_order

    static constexpr uint64_t stride_order_encoded  = encode(stride_order);
    static constexpr uint64_t stride_order_as_given = StrideOrder;

    static constexpr layout_prop_e layout_prop = LayoutProp;
    static constexpr layout_info_t layout_info = layout_info_t{stride_order_encoded, layout_prop};

    // ----------------  Accessors -------------------------

    /// Rank of the map (number of arguments)
    static constexpr int rank() noexcept { return Rank; }

    /// Compile time size, 0 is unknown
    static constexpr long ce_size() noexcept {
      if constexpr (rank_dynamic != 0) { // quick general case
        return 0;
      } else {
        long s = 1;
        for (int u = 0; u < Rank; ++u) s *= static_extents[u];
        return s;
      }
    }

    /** 
     * Lengths of each dimension.
     */
    [[nodiscard]] std::array<long, Rank> const &lengths() const noexcept { return len; }

    /** 
     * Strides of each dimension.
     */
    [[nodiscard]] std::array<long, Rank> const &strides() const noexcept { return str; }

    /// Total number of elements (products of lengths in each dimension).
    [[nodiscard]] long size() const noexcept { return std::accumulate(len.cbegin(), len.cend(), 1, std::multiplies<long>()); }

    /// Is the data contiguous in memory ?
    [[nodiscard]] bool is_contiguous() const noexcept {
      int slowest_index = std::distance(str.begin(), std::max_element(str.begin(), str.end())); // index with minimal stride
      return (str[slowest_index] * len[slowest_index] == size());
    }

    ///
    static constexpr bool is_stride_order_C() {
      return (encode(stride_order) == encode(permutations::identity<Rank>()));
    } // (stride_order == permutations::identity<Rank>()); }

    ///
    static constexpr bool is_stride_order_Fortran() {
      return (encode(stride_order) == encode(permutations::reverse_identity<Rank>()));
    } //(stride_order == permutations::reverse_identity<Rank>()); }

    /////
    //[[nodiscard]] bool check_stride_order() const noexcept {
    //bool r = true;
    //for (int i = 1; i < rank(); ++i) r &= (str[stride_order[i - 1]] <= str[stride_order[i]]); // runtime
    //return r;
    //}

    [[nodiscard]] long min_stride() const noexcept { return str[stride_order[Rank - 1]]; }

    // ----------------  Constructors -------------------------

    private:
    void _compute_strides() {
      // compute the strides for a compact array
      long s = 1;
      for (int v = rank() - 1; v >= 0; --v) { // rank() is constexpr ...
        int u  = stride_order[v];
        str[u] = s;
        s *= len[u];
      }
      ENSURES(s == size());
    }

    public:
    /// Default constructor. Lengths and Strides are not initiliazed.
    idx_map() {
      if constexpr (rank_dynamic == 0) { // full static array
        for (int u = 0; u < Rank; ++u) len[u] = static_extents[u];
        _compute_strides();
      } else {
        for (int u = 0; u < Rank; ++u) len[u] = 0; // to have the proper invariant of the array : shape = (0,0,...) and pointer is null
      }
    }

    ///
    idx_map(idx_map const &) = default;

    ///
    idx_map(idx_map &&) = default;

    ///
    idx_map &operator=(idx_map const &) = default;

    ///
    idx_map &operator=(idx_map &&) = default;

    /** 
     * From an idxmap with other info flags
     * @param idxm
     */
    template <layout_prop_e LayoutProp2>
    idx_map(idx_map<Rank, StaticExtents, StrideOrder, LayoutProp2> const &idxm) noexcept : len(idxm.lengths()), str(idxm.strides()) {
      static_assert(is_degradable(LayoutProp2, LayoutProp),
                    "Can not construct the view: it would violate some compile time guarantees about the layout");
    }

    /** 
     * Construction from the lengths, the strides
     * @param lengths
     * @param strides
     */
    idx_map(std::array<long, Rank> const &lengths, std::array<long, Rank> const &strides) noexcept : len(lengths), str(strides) {}

    /** 
     * Construction from the lengths, the strides
     * @param lengths
     * @param strides
     */
    idx_map(std::array<long, Rank> const &lengths) noexcept : len(lengths) {

#ifdef NDA_DEBUG
      if constexpr (rank_dynamic != Rank) { // there are some static extents
        for (int u = 0; u < Rank; ++u)
          if (static_extents[u] != 0) EXPECTS(static_extents[u] == len[u]);
      }
#endif
      _compute_strides();
    }

    private:
    std::array<long, Rank> _embed_array(std::array<long, rank_dynamic> const &s) {
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
    idx_map(std::array<long, rank_dynamic> const &lengths) noexcept REQUIRES((rank_dynamic != Rank) and (rank_dynamic != 0))
       : idx_map(_embed_array(lengths)) {}

    // trap for incorrect calls. For R = Rank, the non template has priority
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
     *      if one argument is a range, or ellipsis : the sliced idx_map
     *      else : the linear position (long)
     *
     */
    template <typename... Args>
    FORCEINLINE auto operator()(Args const &... args) const
#ifdef NDA_ENFORCE_BOUNDCHECK
       noexcept(false) {
      details::assert_in_bounds(rank(), len.data(), args...);
#else
       noexcept(true) {
#endif
      return call_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
    }

    //// ----------------  Iterator -------------------------

    //using iterator = idx_map_iterator<idx_map<Rank, StrideOrder, LayoutProp>>;

    //[[nodiscard]] iterator begin() const { return {this}; }
    //[[nodiscard]] iterator cbegin() const { return {this}; }
    //[[nodiscard]] typename iterator::end_sentinel_t end() const { return {}; }
    //[[nodiscard]] typename iterator::end_sentinel_t cend() const { return {}; }

    // ----------------  Comparison -------------------------
    bool operator==(idx_map const &x) { return (len == x.len) and (str == x.str); }

    bool operator!=(idx_map const &x) { return !(operator==(x)); }

    // ----------------  Private -------------------------
    private:
    //  BOOST Serialization
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int) {
      ar &len &str;
    }

  }; // idx_map class

  //// DEBUG
  //template <int Rank, uint64_t SO1, layout_prop_e LP1,  uint64_t SO2, layout_prop_e LP2>
  //bool operator ==(idx_map<Rank, SO1, LP1> const & i1, idx_map<Rank, SO2, LP2> const & i2) {
  //if constexpr((LP1 != LP2) or ((SO1 != SO2)) return false;
  //else

  // ---------------- Transposition -------------------------

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

  //// ----------------  More complex iterators -------------------------

  //template <int Rank, bool WithIndices, typename TraversalOrder> struct _changed_iter {
  //idx_map<Rank> const *im;
  //using iterator = idx_map_iterator<Rank, WithIndices, TraversalOrder>;
  //iterator begin() const { return {im}; }
  //iterator cbegin() const { return {im}; }
  //typename iterator::end_sentinel_t end() const { return {}; }
  //typename iterator::end_sentinel_t cend() const { return {}; }
  //};

  //// for (auto [pos, idx] : enumerate(idxmap)) : iterate on position and indices
  //template <int Rank> _changed_iter<Rank, true, traversal::C_t> enumerate_indices(idx_map<Rank> const &x) { return {&x}; }

  //template <int Rank> _changed_iter<Rank, false, traversal::Dynamical_t> in_stride_order_order(idx_map<Rank> const &x) { return {&x}; }
  //template <int Rank> _changed_iter<Rank, true, traversal::Dynamical_t> enumerate_indices_in_stride_order_order(idx_map<Rank> const &x) { return {&x}; }

  // ----------------  foreach  -------------------------

  /*  template <int Rank, uint64_t StrideOrder, layout_prop_e LayoutProp, typename... Args>*/
  //FORCEINLINE void for_each(idx_map<Rank, StrideOrder, LayoutProp> const &idx, Args &&... args) {
  //for_each(idx.lengths(), std::forward<Args>(args)...);
  /*}*/

} // namespace nda
