#pragma once
#include "./layout.hpp"
#include <vector>
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

    //static constexpr int _Rank_in_tpl = Rank;
    using l_t = vec_or_array<long, Rank>;
    l_t _lengths, _strides;
    layout_t<Rank> _layout;
    long _offset = 0;

    public:
    static constexpr bool is_rank_dynamical() { return Rank == -1; }

    // ----------------  Access lengths, strides, layout, offset -------------------------

    /** 
     * Lengths of each dimension.
     * @return std::array<long, Rank> if Rank > 0 else std::vector<long>
     *
     * */
    auto const &lengths() const { return _lengths; }

    /** 
     * Strides of each dimension.
     * @return std::array<long, Rank> if Rank > 0 else std::vector<long>
     *
     * */
    auto const &strides() const { return _strides; }

    /// Total number of elements (products of lengths in each dimension).
    long size() const { return product_of_elements(_lengths); }

    /// Shift from origin
    long offset() const { return _offset; }

    /// Is the data contiguous in memory ?
    bool is_contiguous() const {
      int slowest_index = _layout[0];
      return (_strides[slowest_index] * _lengths[slowest_index] == size());
    }

    layout_t<Rank> const &layout() const { return _layout; }

    // ----------------  Constructors -------------------------

    /// Default constructor. Lengths and Strides are not initiliazed.
    idx_map() = default;

    private: // for slicing only
    idx_map(int r) : _layout(r) {
      if constexpr (is_rank_dynamical()) {
        _lengths.resize(r);
        _strides.resize(r);
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
     * @param layout
     * @param offset
     * */
    idx_map(l_t const &lengths, l_t const &strides, long offset, layout_t<Rank> const &layout)
       : _lengths(lengths), _strides(strides), _layout(layout), _offset(offset) {}

    /// From the lengths and the layout. A compact set of indices, in layout order
    idx_map(l_t const &lengths, layout_t<Rank> const &layout) : _lengths(lengths), _layout(layout) {
      // compute the strides for a compact array
      long str = 1;
      for (int v = this->rank() - 1; v >= 0; --v) { // rank() is constexpr ...
        int u       = _layout[v];
        _strides[u] = str;
        str *= _lengths[u];
      }
      assert(str == size());
    }

    /// From the lengths. A compact set of indices, in C
    idx_map(l_t const &lengths, layout::C_t) : idx_map{lengths, layout_t<Rank>{layout::C, int(lengths.size())}} {}

    /// From the lengths. A compact set of indices, in Fortran
    idx_map(l_t const &lengths, layout::Fortran_t) : idx_map{lengths, layout_t<Rank>{layout::Fortran, int(lengths.size())}} {}

    /// From the lengths. A compact set of indices, in C order
    idx_map(l_t const &lengths) : idx_map{lengths, layout::C} {}

    /// Cross construction
    template <int R> explicit idx_map(idx_map<R> const &m) {
      if constexpr (is_rank_dynamical()) {                            // accept anything
        static_assert(R != -1, "Internal error : should not happen"); // should not match
        this->_rank = m.rank();
        _lengths.resize(this->_rank);
        _strides.resize(this->_rank);
      } else {                            // this has static rank
        if constexpr (m.is_dynamical()) { //
          if (m.rank() != Rank) NDA_RUNTIME_ERROR << "Can not construct from a dynamical rank " << m.rank() << " while expecting " << Rank;
        } else {
          static_assert(R == Rank, "array : mismatch rank in construction, with static rank");
        }
      }
      for (int u = 0; u < this->rank() - 1; ++u) {
        _lengths[u] = m.lengths()[u];
        _strides[u] = m.strides()[u];
        _layout[u]  = m.layout()[u];
      }
      _offset = m.offset();
    }

    // ----------------  Call operator -------------------------

    private:
    // call implementation
    template <typename... Args, size_t... Is> FORCEINLINE long call_impl(std::index_sequence<Is...>, Args const &... args) const {
      return ((args * _strides[Is]) + ...);
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

      static constexpr int n_args_range_ellipsis = ((std::is_same_v<Args, range> or std::is_same_v<Args, ellipsis>)+...);

#ifdef NDA_ENFORCE_BOUNDCHECK
      details::assert_in_bounds(this->rank(), _lengths.data(), args...);

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

        vec_or_array<int, Rank> imap;
        if constexpr (is_rank_dynamical()) { imap.resize(this->rank()); }

        auto w = details::slice_worker{_lengths.data(), _strides.data(), result._lengths.data(), result._strides.data(), imap.data(), result._offset};
        if constexpr (is_rank_dynamical()) {
          w.process_dynamic(this->rank(), args...);
        } else {
          w.process_static<Rank>(args...);
        }

        // Compute the new layout
        for (int i = 0, j = 0; j < this->rank(); ++j) {
          auto k = imap[_layout[j]];
          if (k != -1) result._layout[i++] = k;
        }
        return result;
      }
    } // namespace nda

    // ----------------  Iterator -------------------------

    using iterator = idx_map_iterator<Rank>;

    iterator begin() const { return {this}; }
    iterator cbegin() const { return {this}; }
    typename iterator::end_sentinel_t end() const { return {}; }
    typename iterator::end_sentinel_t cend() const { return {}; }

    // ----------------  Comparison -------------------------
    bool operator==(idx_map const &x) {
      return (_lengths == x._lengths) and (_strides == x._strides) and (_layout == x._layout) and (_offset == x._offset);
    }

    bool operator!=(idx_map const &x) { return !(operator==(x)); }

    // ----------------  Private -------------------------
    private:
    //  BOOST Serialization
    friend class boost::serialization::access;
    template <class Archive> void serialize(Archive &ar, const unsigned int version) { ar &_lengths &_strides &_offset &_layout; }
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
