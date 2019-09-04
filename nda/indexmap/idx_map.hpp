#pragma once
#include <vector>
#include <tuple>
#include <algorithm>
#include <array>
#include <numeric>

#include "../macros.hpp"
#include "../traits.hpp"

namespace nda {

  template <int Rank, uint64_t Layout = 0>
  class idx_map;

}

#include "./slice_static.hpp"
#include "./bound_check_worker.hpp"
#include "./idx_map_iterator.hpp"
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

  // -----------------------------------------------------------------------------------
  /**
   *
   * The map of the indices to linear index.
   *
   * It is a set of lengths and strides for each dimensions, and a shift.
   *
   * @tparam Rank : rank of the index map
   * 
   * @tparam Layout : a permutation for the memory layout of the array
   *    
   *    Layout[0] : the slowest index,
   *    Layout[Rank-1] : the fastest index
   *    Example :
   *        012 : C the last index is the fastest
   *        210 : Fortran, the first index is the fastest
   *        120 : storage (i,j,k) is : index j is slowest, then k, then i
   *    
   *    NB : Layout = 0 is the default and it is means 0 order
   *
   * */
  template <int Rank, uint64_t Layout>
  class idx_map {

    static_assert(Rank < 64, "Rank must be < 64"); // constraint of slice implementation. ok...

    std::array<long, Rank> len, str;
    long _offset = 0;

    public:
    static constexpr std::array<int, Rank> layout =
       (Layout == 0 ? permutations::identity<Rank>() : permutations::decode<Rank>(Layout)); // 0 is C layout

    // ----------------  Accessors -------------------------

    /// Rank of the map (number of arguments)
    static constexpr int rank() noexcept { return Rank; }

    /** 
     * Lengths of each dimension.
     */
    std::array<long, Rank> const &lengths() const noexcept { return len; }

    /** 
     * Strides of each dimension.
     */
    std::array<long, Rank> const &strides() const noexcept { return str; }

    /// Total number of elements (products of lengths in each dimension).
    long size() const noexcept { return std::accumulate(len.cbegin(), len.cend(), 1, std::multiplies<long>()); }

    /// Shift from origin
    long offset() const noexcept { return _offset; }

    /// Is the data contiguous in memory ?
    bool is_contiguous() const noexcept {
      int slowest_index = std::distance(str.begin(), std::min_element(str.begin(), str.end())); // index with minimal stride
      return (str[slowest_index] * len[slowest_index] == size());
    }

    ///
    static constexpr bool is_layout_C() { return (layout == permutations::identity<Rank>()); }

    ///
    static constexpr bool is_layout_Fortran() { return (layout == permutations::reverse_identity<Rank>()); }

    ///
    bool check_layout() const noexcept {
      bool r = true;
      for (int i = 1; i < rank(); ++i) r &= (str[layout[i - 1]] <= str[layout[i]]); // runtime
      return r;
    }

    // ----------------  Constructors -------------------------

    /// Default constructor. Lengths and Strides are not initiliazed.
    idx_map() = default;

    ///
    idx_map(idx_map const &) = default;

    ///
    idx_map(idx_map &&) = default;

    ///
    idx_map &operator=(idx_map const &) = default;

    ///
    idx_map &operator=(idx_map &&) = default;

    /** 
     * Construction from the lengths, the strides, offset
     * @param lengths
     * @param strides
     * @param offset
     */
    idx_map(std::array<long, Rank> const &lengths, std::array<long, Rank> const &strides, long offset) noexcept
       : len(lengths), str(strides), _offset(offset) {}

    /** 
     * Construction from the lengths, the strides, offset
     * @param lengths
     * @param strides
     * @param offset
     */
    idx_map(std::array<long, Rank> const &lengths) noexcept : len(lengths) {
      // compute the strides for a compact array
      long s = 1;

      for (int v = this->rank() - 1; v >= 0; --v) { // rank() is constexpr ...
        int u  = layout[v];
        str[u] = s;
        s *= len[u];
      }
     ENSURES(s == size());
    }

    // trap for incorrect calls. For R = Rank, the non template has priority
    template <int R>
    idx_map(std::array<long, R> const &) {
      static_assert(R == Rank, "Rank of the argument incorrect in idx_map construction");
    }

    /**
     * @param idx An index map with dynamical rank
     *
     * NB : Throws if idx has the correct rank or throws
     */
    /*  idx_map(idx_map_dyn const &m) {
      if (m.rank() != Rank) NDA_RUNTIME_ERROR << "Can not construct from a dynamical rank " << m.rank() << " while expecting " << Rank;
      for (int u = 0; u < rank(); ++u) {
        len[u] = m.lengths()[u];
        str[u] = m.strides()[u];
      }
      _offset = m.offset();
    }
    */

    // ----------------  Call operator -------------------------

    private:
    template <size_t Is>
    FORCEINLINE long __get(long arg) const noexcept {
      if constexpr (Is == layout[Rank - 1])
        return arg;
      else
        return arg * std::get<Is>(str);
    }

    // call implementation
    template <uint64_t Guarantee, typename... Args, size_t... Is>
    FORCEINLINE long call_impl(std::index_sequence<Is...>, Args... args) const noexcept {
      if constexpr (guarantee::has_smallest_stride_is_one(Guarantee))
        return (__get<Is>(args) + ...);
      else
        return ((args * std::get<Is>(str)) + ...);
    }

#ifdef NDA_ENFORCE_BOUNDCHECK
    static constexpr bool enforce_bound_check = true;
#else
    static constexpr bool enforce_bound_check = false;
#endif

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
    template <uint64_t Guarantee, typename... Args>
    FORCEINLINE auto slice_or_position(Args const &... args) const noexcept(enforce_bound_check) {

      static_assert(((((std::is_base_of_v<range_tag, Args> or std::is_constructible_v<long, Args>) ? 0 : 1) + ...) == 0),
                    "Slice arguments must be convertible to range, Ellipsis, or long");

      static constexpr int n_args_ellipsis = ((std::is_same_v<Args, ellipsis>)+...);
      static constexpr int n_args_long     = (std::is_constructible_v<long, Args> + ...);

      static_assert(n_args_ellipsis <= 1, "Only one ellipsis argument is authorized");
      static_assert((sizeof...(Args) <= Rank), "Incorrect number of arguments in array call ");
      static_assert((n_args_ellipsis == 1) or (sizeof...(Args) == Rank), "Incorrect number of arguments in array call ");

#ifdef NDA_ENFORCE_BOUNDCHECK
      details::assert_in_bounds(rank(), len.data(), args...);
#endif

      if constexpr (n_args_long == Rank) { // no range, ellipsis, we simply compute the linear position
        auto _fold = call_impl<Guarantee>(std::make_index_sequence<sizeof...(Args)>{},
                               args...);           // NB do not use index_sequence_for : one instantation only by # args.
        if (guarantee::has_zero_offset(Guarantee)) // zero offset optimization
          return _fold;
        else
          return _offset + _fold;
      } else { // otherwise we make a  new sliced idx_map
        return slice_static::slice(std::make_index_sequence<Rank - n_args_long>{}, std::make_index_sequence<Rank>{},
                                   std::make_index_sequence<sizeof...(Args)>{}, *this, args...);
      }
    }

    // FIXME kept for the test for the moment
    template <typename... Args>
    FORCEINLINE auto operator()(Args const &... args) const noexcept(enforce_bound_check) {
      return slice_or_position<0>(args...);
    }

    // ----------------  Iterator -------------------------

    using iterator = idx_map_iterator<idx_map<Rank, Layout>>;

    iterator begin() const { return {this}; }
    iterator cbegin() const { return {this}; }
    typename iterator::end_sentinel_t end() const { return {}; }
    typename iterator::end_sentinel_t cend() const { return {}; }

    // ----------------  Comparison -------------------------
    bool operator==(idx_map const &x) { return (len == x.len) and (str == x.str) and (_offset == x._offset); }

    bool operator!=(idx_map const &x) { return !(operator==(x)); }

    // ----------------  Private -------------------------
    private:
    //  BOOST Serialization
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int) {
      ar &len &str &_offset;
    }

  }; // idx_map class

  // ---------------- Transposition -------------------------

  template <int Rank>
  idx_map<Rank> transpose(idx_map<Rank> const &idx, std::array<int, Rank> const &perm) {
    std::array<long, Rank> l, s;
    for (int u = 0; u < idx.rank(); ++u) {
      l[perm[u]] = idx.lengths()[u];
      s[perm[u]] = idx.strides()[u];
    }
    return {l, s, idx.offset()};
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

  //template <int Rank> _changed_iter<Rank, false, traversal::Dynamical_t> in_layout_order(idx_map<Rank> const &x) { return {&x}; }
  //template <int Rank> _changed_iter<Rank, true, traversal::Dynamical_t> enumerate_indices_in_layout_order(idx_map<Rank> const &x) { return {&x}; }

  // ----------------  foreach  -------------------------

  template <int R, typename... Args>
  FORCEINLINE void for_each(idx_map<R> const &idx, Args &&... args) {
    for_each(idx.lengths(), std::forward<Args>(args)...);
  }

} // namespace nda
