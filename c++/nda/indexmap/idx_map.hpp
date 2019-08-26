#pragma once
#include <vector>
#include <tuple>
#include <algorithm>
#include <array>
#include <numeric>

#include "./slice_static.hpp"
#include "./bound_check_worker.hpp"
#include "./idx_map_iterator.hpp"
#include "./for_each.hpp"

namespace boost::serialization {
  class access;
}

namespace nda {

  /// Type of all shapes
  template <int Rank> using shape_t = std::array<long, Rank>;

  /// Shape factory
  template <typename... T> shape_t<sizeof...(T)> make_shape(T... args) noexcept { return {args...}; }

  namespace layout {

    inline struct C_t {
    } C;
    inline struct Fortran_t { } Fortran; } // namespace layout

  /// Type of layout (the general categories which are invariant by slicing).
  //enum class layout { C, Fortran, Custom};

  // -----------------------------------------------------------------------------------
  /**
   *
   * The map of the indices to linear index.
   *
  ` * It is a set of lengths and strides for each dimensions, and a shift.
   *
   *
   * */
  template <int Rank> class idx_map {

    static_assert(Rank < 64, "Rank must be < 64"); // constraint of slice implementation. ok...

    std::array<long, Rank> len, str;
    long _offset = 0;
    //using layout_t = std::array<int, Rank>;

    public:
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
    bool is_layout_C() const noexcept {
      bool r = true;
      for (int i = 1; i < rank(); ++i) r &= (str[i - 1] >= str[i]);
      return r;
    }

    ///
    bool is_layout_Fortran() const noexcept {
      bool r = true;
      for (int i = 1; i < rank(); ++i) r &= (str[i - 1] <= str[i]);
      return r;
    }

    /**
     * Computes the layout. 
     * Return an array/vector layout such that
     * layout[0] is the slowest index (i.e. largest stride)
     * layout[rank] the fastest index (i.e. smallest stride)\
     * 
     * NB. Not very quick, it sorts the strides
     */
    std::array<int, Rank> layout() const noexcept {
      std::array<int, Rank> lay;
      std::array<std::pair<int, int>, Rank> lay1;
      // Compute the permutation
      for (int i = 0; i < rank(); ++i) lay1[i] = {str[i], i};
      std::sort(lay1.begin(), lay1.end(), std::greater<>{});
      for (int i = 0; i < rank(); ++i) lay[i] = lay1[i].second;
      return lay;
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
    idx_map(std::array<long, Rank> const &lengths, std::array<int, Rank> const &layout) noexcept : len(lengths) {
      len = lengths;
      // compute the strides for a compact array
      long s = 1;
      for (int v = this->rank() - 1; v >= 0; --v) { // rank() is constexpr ...
        int u  = layout[v];
        str[u] = s;
        s *= len[u];
      }
      assert(str == size());
    }

    /// From the lengths. A compact set of indices, in C
    idx_map(std::array<long, Rank> const &lengths, layout::C_t) noexcept {
      len    = lengths;
      long s = 1;
      for (int v = this->rank() - 1; v >= 0; --v) { // rank() is constexpr ...
        str[v] = s;
        s *= len[v];
      }
    }

    /// From the lengths. A compact set of indices, in Fortran
    idx_map(std::array<long, Rank> const &lengths, layout::Fortran_t) noexcept {
      len    = lengths;
      long s = 1;
      for (int v = 0; v < this->rank(); --v) { // rank() is constexpr ...
        str[v] = s;
        s *= len[v];
      }
    }

    /// From the lengths. A compact set of indices, in C order.
    idx_map(std::array<long, Rank> const &lengths) noexcept : idx_map{lengths, layout::C} {}

    // trap
    template <int R> idx_map(std::array<long, R>) { static_assert(R == Rank, "WRONG"); }

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
    // call implementation
    template <typename... Args, size_t... Is> FORCEINLINE long call_impl(std::index_sequence<Is...>, Args const &... args) const noexcept {
      return ((args * str[Is]) + ...);
    }

#ifdef NDA_ENFORCE_BOUNDCHECK
    static constexpr bool enforce_bound_check = false;
#else
    static constexpr bool enforce_bound_check = true;
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
    template <typename... Args> FORCEINLINE auto operator()(Args const &... args) const noexcept(enforce_bound_check) {

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
        return _offset
           + call_impl(std::make_index_sequence<sizeof...(Args)>{}, args...); // NB do not use index_sequence_for : one instantation only by # args.
      } else {                                                                // otherwise we make a  new sliced idx_map
        return slice_static::slice(std::make_index_sequence<Rank - n_args_long>{}, std::make_index_sequence<Rank>{},
                                   std::make_index_sequence<sizeof...(Args)>{}, *this, args...);
      }
    }

    // ----------------  Iterator -------------------------

    using iterator = idx_map_iterator<Rank>;

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
    template <class Archive> void serialize(Archive &ar, const unsigned int) { ar &len &str &_offset; }

  }; // idx_map class

  // ---------------- Transposition -------------------------

  template <int Rank> idx_map<Rank> transpose(idx_map<Rank> const &idx, std::array<int, Rank> const &perm) {
    std::array<long, Rank> l, s;
    for (int u = 0; u < idx.rank(); ++u) {
      l[perm[u]] = idx.lengths()[u];
      s[perm[u]] = idx.strides()[u];
    }
    return {l, s, idx.offset()};
  }

  // ----------------  More complex iterators -------------------------

  template <int Rank, bool WithIndices, typename TraversalOrder> struct _changed_iter {
    idx_map<Rank> const *im;
    using iterator = idx_map_iterator<Rank, WithIndices, TraversalOrder>;
    iterator begin() const { return {im}; }
    iterator cbegin() const { return {im}; }
    typename iterator::end_sentinel_t end() const { return {}; }
    typename iterator::end_sentinel_t cend() const { return {}; }
  };

  // for (auto [pos, idx] : enumerate(idxmap)) : iterate on position and indices
  template <int Rank> _changed_iter<Rank, true, traversal::C_t> enumerate_indices(idx_map<Rank> const &x) { return {&x}; }

  //template <int Rank> _changed_iter<Rank, false, traversal::Dynamical_t> in_layout_order(idx_map<Rank> const &x) { return {&x}; }
  //template <int Rank> _changed_iter<Rank, true, traversal::Dynamical_t> enumerate_indices_in_layout_order(idx_map<Rank> const &x) { return {&x}; }

  // ----------------  foreach  -------------------------

  template <int R, typename... Args> FORCEINLINE void for_each(idx_map<R> const &idx, Args &&... args) {
    for_each(idx.lengths(), std::forward<Args>(args)...);
  }

} // namespace nda
