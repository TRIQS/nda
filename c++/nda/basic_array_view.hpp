#pragma once
#include <clef/clef.hpp>
#include "declarations.hpp"
#include "assignment.hpp"
#include "iterators.hpp"
#include "layout/slice_static.hpp"

// The std::swap is WRONG for a view because of the copy/move semantics of view.
// Use swap instead (the correct one, found by ADL).
namespace std {
  template <typename V, int R, typename L, char A, typename Al, typename Ow, typename V2, int R2, typename L2, char A2, typename Al2, typename Ow2>
  void swap(nda::basic_array_view<V, R, L, A, Al, Ow> &a, nda::basic_array_view<V2, R2, L2, A2, Al2, Ow2> &b) =
     delete; // std::swap disabled for basic_array_view. Use nda::swap iinstead (or simply swap, found by ADL).
}

namespace nda {

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  class basic_array_view {

    public:
    /// ValueType FIXME
    using value_t    = ValueType;
    using value_type = ValueType;

    //using value_as_template_arg_t = ValueType;
    using storage_t = typename OwningPolicy::template handle<ValueType>;
    using idx_map_t = typename Layout::template mapping<Rank>;

    ///
    using regular_t =
       basic_array<ValueType, Rank, basic_layout<encode(idx_map_t::static_extents), encode(idx_map_t::stride_order), layout_prop_e::contiguous>,
                   Algebra, heap>;
    ///
    using view_t = basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>;
    ///
    using const_view_t = basic_array_view<ValueType const, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>;
    ///
    using no_const_view_t = basic_array_view<std::remove_const_t<ValueType>, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>;

    static constexpr int rank      = Rank;
    static constexpr bool is_view  = true;
    static constexpr bool is_const = std::is_const_v<ValueType>;

    private:
    template <typename IdxMap>
    using my_view_template_t =
       basic_array_view<ValueType, IdxMap::rank(), basic_layout<encode(IdxMap::static_extents), encode(IdxMap::stride_order), IdxMap::layout_prop>,
                        Algebra, AccessorPolicy, OwningPolicy>;

    idx_map_t _idx_m;
    storage_t _storage;

    public:
    // ------------------------------- constructors --------------------------------------------

    /// Construct an empty view.
    basic_array_view() = default;

    ///
    basic_array_view(basic_array_view &&) = default;

    /// Shallow copy. It copies the *view*, not the data.
    basic_array_view(basic_array_view const &) = default;

    /** 
     * From a view of non const ValueType.
     * Only valid when ValueType is const
     *
     * @param v a view 
     */
    basic_array_view(basic_array_view const &v) NDA_REQUIRES(is_const) : basic_array_view(v.indexmap(), v.storage()) {}

    /**
     *  [Advanced] From an indexmap and a storage handle
     *  @param idxm index map
     *  @st  storage (memory handle)
     */
    basic_array_view(idx_map_t const &idxm, storage_t st) : _idx_m(idxm), _storage(std::move(st)) {}

    /** 
     * From other containers and view : array, matrix, matrix_view.
     *
     * @tparam A an array/array_view or matrix/vector type
     * @param a array or view
     */
    template <typename A>                                          //explicit
    basic_array_view(A const &a) NDA_REQUIRES(is_regular_or_view_v<A>) //and layout_are_compatible_for_view<A::idx_map_t,idx_map_t>)
       : basic_array_view(idx_map_t{a.indexmap()}, a.storage()) {}

    /** 
     * [Advanced] From a pointer to contiguous data, and a shape.
     * NB : no control obvious on the dimensions given.  
     *
     * @param p Pointer to the data
     * @param shape Shape of the view (contiguous)
     */
    basic_array_view(std::array<long, Rank> const &shape, ValueType *p) : basic_array_view(idx_map_t{shape}, p) {}

    /** 
     * [Advanced] From a pointer to data, and an idx_map 
     * NB : no control obvious on the dimensions given.  
     *
     * @param p Pointer to the data 
     * @param idxm Index Map (view can be non contiguous). If the offset is non zero, the view starts at p + idxm.offset()
     */
    basic_array_view(idx_map_t const &idxm, ValueType *p) : _idx_m(idxm), _storage{p} {}
    //basic_array_view(idx_map<Rank, StrideOrder> const &idxm, ValueType *p) : _idx_m(idxm), _storage{p, size_t(idxm.size() + idxm.offset())} {}

    // Move assignment not defined : will use the copy = since view must copy data

    // ------------------------------- assign --------------------------------------------

    /**
     * Copies the content of rhs into the view.
     * Pseudo code : 
     *     for all i,j,k,l,... : this[i,j,k,l] = rhs(i,j,k,l)
     *
     * The dimension of RHS must be large enough or behaviour is undefined.
     * 
     * If NDA_BOUNDCHECK is defined, the bounds are checked.
     *
     * @tparam RHS A scalar or an object modeling the concept NDArray
     * @param rhs Right hand side of the = operation
     */
    template <typename RHS>
    basic_array_view &operator=(RHS const &rhs) {
      assign_from(*this, rhs);
      return *this;
    }

    /// Same as the general case
    /// [C++ oddity : this case must be explicitly coded too]
    basic_array_view &operator=(basic_array_view const &rhs) {
      assign_from(*this, rhs);
      return *this;
    }

    // ------------------------------- rebind --------------------------------------------

    /// Rebind the view
    void rebind(basic_array_view const &a) { //value_t is NEVER const
      _idx_m   = a._idx_m;
      _storage = a._storage;
    }

    /// Rebind view
    void rebind(no_const_view_t const &a) NDA_REQUIRES(is_const) {
      //static_assert(is_const, "Can not rebind a view of const ValueType to a view of ValueType");
      _idx_m   = idx_map_t{a.indexmap()};
      _storage = storage_t{a.storage()};
    }
    //check https://godbolt.org/z/G_QRCU

    // ------------------------------- swap --------------------------------------------

    /**
     * Swaps the *views* a and b, without copying data
     * @param a
     * @param b
     */
    friend void swap(basic_array_view &a, basic_array_view &b) {
      std::swap(a._idx_m, b._idx_m);
      std::swap(a._storage, b._storage);
    }


    // FIXME : pull out !!
    // copy view, no need of friend !


    /**
     * Swaps the *views* a and b, without copying data
     * @param a
     * @param b
     */
    friend void deep_swap(basic_array_view &a, basic_array_view &b) {
      // FIXME Is this optimal ??
      // Do we want to keep this function ?? Used only in det_manip, in 1d
      //
      auto tmp = make_regular(a);
      a        = b;
      b        = tmp;
    }

    // FIXME useless !!
    // on or both can be a temporary !
    /**
     * Swaps the *views* a and b, without copying data
     * @param a
     * @param b
     */
    friend void deep_swap(basic_array_view &&a, basic_array_view &b) { deep_swap(a, b); }

    /**
     * Swaps the *views* a and b, without copying data
     * @param a
     * @param b
     */
    friend void deep_swap(basic_array_view &a, basic_array_view &&b) { deep_swap(a, b); }

    /**
     * Swaps the *views* a and b, without copying data
     * @param a
     * @param b
     */
    friend void deep_swap(basic_array_view &&a, basic_array_view &&b) { deep_swap(a, b); }

    //----------------------------------------------------

#include "./_impl_basic_array_view_common.hpp"
  };

} // namespace nda
