#pragma once
#include <vector>
#include <array>
#include <numeric>

#include "./slice_worker.hpp"
#include "./bound_check_worker.hpp"
#include "./idx_map_iterator.hpp"
#include "./for_each.hpp"

namespace boost::serialization {
  class access;
}

namespace nda {

  /// ------------ General definitions -----------------

  ///
  static constexpr int DynamicalRank = -1;

  /// Type of all shapes
  template <int Rank> using shape_t = std::array<long, Rank>;

  /// Shape factory
  template <typename... T> shape_t<sizeof...(T)> make_shape(T... args) { return {args...}; }

  // ------------  details for differences between static and dynamic rank  -----------------

  // FIXME : Optimisation : replace vector with a vector with SSO.
  template <typename T, int Rank> struct vec_or_array_t { using type = std::array<T, Rank>; };
  template <typename T> struct vec_or_array_t<T, -1> { using type = std::vector<T>; };

  template <typename T, int Rank> using vec_or_array = typename vec_or_array_t<T, Rank>::type;

  // -------------   the method rank is constexpr or dynamical ---------------------

  template <int Rank> struct _rank_injector {
    static constexpr int rank() { return Rank; }
  };

  template <> struct _rank_injector<-1> {
    int rank() const { return _rank; }

    private:
    int _rank = 0;
  };

  /// Type of layout (the general categories which are invariant by slicing).
  //enum class layout { C, Fortran, Custom};

  // -----------------------------------------------------------------------------------
  /**
   *
   * The map of the indices to linear index long.
   *
  ` * It is a set of lengths and strides for each dimensions, and a shift.
   *
   * Rank = -1 : Rank is dynamical, with accessor rank()
   *
   * */
  template <int Rank> class idx_map : public _rank_injector<Rank> {

    static_assert(Rank < 64, "Rank must be < 64"); // constraint of slice implementation. ok...

    //static constexpr int _Rank_in_tpl = Rank;
    using l_t      = vec_or_array<long, Rank>;
    using layout_t = vec_or_array<int, Rank>;
    l_t len, str;
    long _offset = 0;

    public:
    static constexpr bool is_rank_dynamical() { return Rank == -1; }

    // ----------------  Access lengths, strides, offset -------------------------

    /** 
     * Lengths of each dimension.
     * @return std::array<long, Rank> if Rank > 0 else std::vector<long>
     *
     * */
    auto const &lengths() const { return len; }

    /** 
     * Strides of each dimension.
     * @return std::array<long, Rank> if Rank > 0 else std::vector<long>
     *
     * */
    auto const &strides() const { return str; }

    /// Total number of elements (products of lengths in each dimension).
    long size() const { return std::accumulate(len.cbegin(), len.cend(), 1, std::multiplies<long>()); }

    /// Shift from origin
    long offset() const { return _offset; }

    /// Is the data contiguous in memory ?
    bool is_contiguous() const {
      auto it = std::min_element(str.begin(), str.end());
      int slowest_index = std::distance(str.begin(), it);
      return (str[slowest_index] * len[slowest_index] == size());
    }

    ///
    bool is_layout_C() const {
      bool r = true;
      for (int i = 1; i < rank(); ++i) r &= (str[i - 1] >= str[i]);
      return r;
    }

    ///
    bool is_layout_Fortran() const {
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
    layout_t layout() const {
      layout_t lay;
      vec_or_array<std::pair<int, int>, Rank> lay1;
      if constexpr (is_rank_dynamical()) {
        lay.resize(this->rank());
        lay1.resize(this->rank());
      }
      // Compute the permutation
      for (int i = 0; i < this->rank(); ++i) lay1[i] = {str[i], i};
      std::sort(lay1.begin(), lay1.end());
      for (int i = 0; i < this->rank(); ++i) lay[i] = lay1[i].second;
      return lay;
    }

    // ----------------  Constructors -------------------------

    /// Default constructor. Lengths and Strides are not initiliazed.
    idx_map() = default;

    private: // for slicing only
    idx_map(int r) : {
      if constexpr (is_rank_dynamical()) {
        len.resize(r);
        str.resize(r);
      }
    }

    // all friend with each other (for slicing).
    template <int> friend class idx_map;

    public:
    /// Copy
    idx_map(idx_map const &) = default;

    /// Move
    idx_map(idx_map &&) = default;

    /// Copy =
    idx_map &operator=(idx_map const &) = default;

    /// Move =
    idx_map &operator=(idx_map &&) = default;

    /** 
     * Construction from the length, the stride, offset, ml
     * @param lengths
     * @param strides
     * @param offset
     * */
    idx_map(l_t const &lengths, l_t const &strides, long offset) : len(lengths), str(strides), _offset(offset) {}

    /// From the lengths. A compact set of indices, in layout order
    idx_map(l_t const &lengths, layout_t const &layout) : len(lengths) {
      // compute the strides for a compact array
      long str = 1;
      for (int v = this->rank() - 1; v >= 0; --v) { // rank() is constexpr ...
        int u   = layout[v];
        str[u] = str;
        str *= len[u];
      }
      assert(str == size());
    }

    /// From the lengths. A compact set of indices, in C
    idx_map(l_t const &lengths, layout::C_t) {
      long str = 1;
      for (int v = this->rank() - 1; v >= 0; --v) { // rank() is constexpr ...
        str[v] = str;
        str *= len[v];
      }
    }

    /// From the lengths. A compact set of indices, in Fortran
    idx_map(l_t const &lengths, layout::Fortran_t) {
      long str = 1;
      for (int v = 0; v < this->rank(); --v) { // rank() is constexpr ...
        str[v] = str;
        str *= len[v];
      }
    }

    /// From the lengths. A compact set of indices, in C order.
    idx_map(l_t const &lengths) : idx_map{lengths, layout::C} {}

    /// Cross construction
    template <int R> explicit idx_map(idx_map<R> const &m) {
      if constexpr (is_rank_dynamical()) {                            // accept anything
        static_assert(R != -1, "Internal error : should not happen"); // should not match
        this->_rank = m.rank();
        len.resize(this->_rank);
        str.resize(this->_rank);
      } else {                            // this has static rank
        if constexpr (m.is_dynamical()) { //
          if (m.rank() != Rank) NDA_RUNTIME_ERROR << "Can not construct from a dynamical rank " << m.rank() << " while expecting " << Rank;
        } else {
          static_assert(R == Rank, "array : mismatch rank in construction, with static rank");
        }
      }
      for (int u = 0; u < this->rank() - 1; ++u) {
        len[u]  = m.lengths()[u];
        str[u] = m.strides()[u];
      }
      _offset = m.offset();
    }

    // ----------------  Call operator -------------------------

    private:
    // call implementation
    template <typename... Args, size_t... Is> FORCEINLINE long call_impl(std::index_sequence<Is...>, Args const &... args) const {
      return ((args * str[Is]) + ...);
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
     * */
    template <typename... Args> auto operator()(Args const &... args) const {

      static_assert(((((std::is_base_of_v<range_tag, Args> or std::is_constructible_v<long, Args>) ? 0 : 1) + ...) == 0),
                    "Slice arguments must be convertible to range, Ellipsis, or long");

      static_assert(((((std::is_base_of_v<ellipsis, Args>) ? 1 : 0) + ...) <= 1), "Only one ellipsis argument is authorized");

      static constexpr int n_args_range_ellipsis = ((std::is_same_v<Args, range> or std::is_same_v<Args, ellipsis>)+...);

#ifdef NDA_ENFORCE_BOUNDCHECK
      details::assert_in_bounds(this->rank(), len.data(), args...);

      if constexpr (n_args_range_ellipsis == 0) {
        if constexpr (is_rank_dynamical()) {
          if (sizeof...(Args) != this->rank())
            NDA_RUNTIME_ERROR << "Incorrect number of argument in array call : expected " << this->rank() << " got  " << sizeof...(Args);
        } else {
          static_assert((sizeof...(Args) == Rank), "Incorrect number of argument in array call ");
        }
      }
#endif

      // === Case 1 :  no range, ellipsis, we simply compute the linear position
      if constexpr (n_args_range_ellipsis == 0) {
        return _offset + call_impl(std::index_sequence_for<Args...>{}, args...);
      } else {
        // === Case 2 : there is a range/ellipsis in the arguments : we make a slice

        static constexpr int n_args_long = (std::is_constructible_v<long, Args> + ...);

        // result : argument is ignored in the static case. cf private constructor.
        idx_map<(is_rank_dynamical() ? DynamicalRank : Rank - n_args_long)> result{this->rank() - n_args_long};

        if constexpr (is_rank_dynamical()) {
          return slice_worker_static::slice({len.data(), str.data(), result.len.data(), result.str.data()}, this->rank(), args...);
        } else {
          return slice_worker_static::slice<Rank>(std::make_index_sequence<Rank - n_args_long>{}, std::make_index_sequence<Rank>{}, *this, args...);
        }
      }
    } // namespace nda

    // ----------------  Iterator -------------------------

    using iterator = idx_map_iterator<Rank>;

    iterator begin() const { return {this}; }
    iterator cbegin() const { return {this}; }
    typename iterator::end_sentinel_t end() const { return {}; }
    typename iterator::end_sentinel_t cend() const { return {}; }

    // ----------------  Comparison -------------------------
    bool operator==(idx_map const &x) { return (len == x.len) and (str == x.str) and (_layout == x._layout) and (_offset == x._offset); }

    bool operator!=(idx_map const &x) { return !(operator==(x)); }

    // ----------------  Private -------------------------
    private:
    //  BOOST Serialization
    friend class boost::serialization::access;
    template <class Archive> void serialize(Archive &ar, const unsigned int version) { ar &len &str &_offset &_layout; }
  }; // namespace nda

  // ---------------- Transposition -------------------------

  template <int Rank> idx_map<Rank> transpose(idx_map<Rank> const &m, std::array<int, Rank> const &perm) {
    typename idx_map<Rank>::l_t l, s, lay;
    for (int u = 0; u < m.rank(); ++u) {
      l[perm[u]] = m.lengths()[u];
      s[perm[u]] = m.strides()[u];
      lay[u]     = perm[m.layout()[u]]; // FIXME : EXPLAIN
    }
    return {l, s, m.offset(), lay};
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

  template <int Rank> _changed_iter<Rank, false, traversal::Dynamical_t> in_layout_order(idx_map<Rank> const &x) { return {&x}; }
  template <int Rank> _changed_iter<Rank, true, traversal::Dynamical_t> enumerate_indices_in_layout_order(idx_map<Rank> const &x) { return {&x}; }

  // ----------------  foreach  -------------------------

  template <int R, typename... Args> FORCEINLINE void for_each(idx_map<R> const &idx, Args &&... args) {
    // FIXME : Dynamical
    for_each(idx.lengths(), std::forward<Args>(args)...);
  }

  // FIXME : clean at the end
  //template <int R1, int R2> bool compatible_for_assignment(idx_map<R1> const &m1, idx_map<R2> const &m2) { return m1.lengths() == m2.lengths(); }

  //template <int R1, int R2> bool raw_copy_possible(idx_map<R1> const &m1, idx_map<R2> const &m2) {
  //  return ((m1.layout() == m2.layout()) && m1.is_contiguous() && m2.is_contiguous() && (m1.size() == m2.size()));
  //}

} // namespace nda
