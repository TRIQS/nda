#pragma once
#include "./storage/handle.hpp"
#include "./indexmap/idx_map.hpp"
#include "./basic_functions.hpp"
#include "./assignment.hpp"
#include "./iterator_adapter.hpp"

namespace nda {

  // ---------------------- declare array and view  --------------------------------

  template <typename ValueType, int Rank, uint64_t Layout = 0>
  class array;
  template <typename ValueType, int Rank, uint64_t Guarantees = 0, uint64_t Layout = 0>
  class array_view;

  // ---------------------- is_array_or_view_container  --------------------------------

  template <typename ValueType, int Rank>
  inline constexpr bool is_regular_v<array<ValueType, Rank>> = true;

  template <typename ValueType, int Rank>
  inline constexpr bool is_regular_or_view_v<array<ValueType, Rank>> = true;

  template <typename ValueType, int Rank, uint64_t Guarantees, uint64_t Layout>
  inline constexpr bool is_regular_or_view_v<array_view<ValueType, Rank, Guarantees, Layout>> = true;

  // ---------------------- concept  --------------------------------

  template <typename ValueType, int Rank, uint64_t Layout>
  inline constexpr bool is_ndarray_v<array<ValueType, Rank, Layout>> = true;

  template <typename ValueType, int Rank, uint64_t Guarantees, uint64_t Layout>
  inline constexpr bool is_ndarray_v<array_view<ValueType, Rank, Guarantees, Layout>> = true;

  // ---------------------- algebra --------------------------------

  template <typename ValueType, int Rank, uint64_t Layout>
  inline constexpr char get_algebra<array<ValueType, Rank, Layout>> = 'A';

  template <typename ValueType, int Rank, uint64_t Guarantees, uint64_t Layout>
  inline constexpr char get_algebra<array_view<ValueType, Rank, Guarantees, Layout>> = 'A';

  // ---------------------- guarantees --------------------------------

  template <typename ValueType, int Rank>
  inline constexpr uint64_t get_guarantee<array<ValueType, Rank>> = array<ValueType, Rank>::guarantees;

  template <typename ValueType, int Rank, uint64_t Guarantees, uint64_t Layout>
  inline constexpr uint64_t get_guarantee<array_view<ValueType, Rank, Guarantees, Layout>> = Guarantees;

  // ---------------------- array_view  --------------------------------

  // Try to put the const/mutable in the TYPE

  template <typename ValueType, int Rank, uint64_t Guarantees, uint64_t Layout>
  class array_view {

    public:
    /// ValueType, without const if any
    using value_t = std::remove_const_t<ValueType>;
    ///
    using regular_t = array<value_t, Rank, Layout>;
    ///
    using view_t = array_view<value_t, Rank, Guarantees, Layout>;
    ///
    using const_view_t = array_view<value_t const, Rank, Guarantees, Layout>;

    //using value_as_template_arg_t = ValueType;
    using storage_t = mem::handle<value_t, 'B'>;
    using idx_map_t = idx_map<Rank, Layout>;

    static constexpr int rank      = Rank;
    static constexpr bool is_view  = true;
    static constexpr bool is_const = std::is_const_v<ValueType>;

    static constexpr uint64_t guarantees = Guarantees; // for the generic shared with array
//    static constexpr uint64_t layout = Layout;
 
    // fIXME : FIRST STEP.
    static_assert(Guarantees ==0, "Not implemented");

    // FIXME : h5
    // static std::string hdf5_scheme() { return "array<" + triqs::h5::get_hdf5_scheme<value_t>() + "," + std::to_string(rank) + ">"; }

    private:
    template <typename IdxMap>
    using my_view_template_t = array_view<value_t, IdxMap::rank(), IdxMap::flags, permutations::encode(IdxMap::layout)>;

    idx_map_t _idx_m;
    storage_t _storage;

    public:
    // ------------------------------- constructors --------------------------------------------

    /// Construct an empty view.
    array_view() = default;

    ///
    array_view(array_view &&) = default;

    /// Shallow copy. It copies the *view*, not the data.
    array_view(array_view const &) = default;

    /** 
     * From a view of non const ValueType.
     * Only valid when ValueType is const
     *
     * @param v a view 
     */
    array_view(array_view<value_t, Rank> const &v) REQUIRES(is_const) : array_view(v.indexmap(), v.storage()) {}

    /**
     *  [Advanced] From an indexmap and a storage handle
     *  @param idxm index map
     *  @st  storage (memory handle)
     */
    array_view(idx_map<Rank, Layout> const &idxm, storage_t st) : _idx_m(idxm), _storage(std::move(st)) {}

    /** 
     * From other containers and view : array, matrix, matrix_view.
     *
     * @tparam A an array/array_view or matrix/vector type
     * @param a array or view
     */
    template <typename A> //explicit
    array_view(A const &a) REQUIRES(is_regular_or_view_v<A>) : array_view(a.indexmap(), a.storage()) {}

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
    array_view &operator=(RHS const &rhs) {
      nda::details::assignment(*this, rhs);
      return *this;
    }

    /// Same as the general case
    /// [C++ oddity : this case must be explicitly coded too]
    array_view &operator=(array_view const &rhs) {
      nda::details::assignment(*this, rhs);
      return *this;
    }

    // ------------------------------- rebind --------------------------------------------

    /// Rebind the view
    void rebind(array_view<value_t, Rank> const &a) { //value_t is NEVER const 
      _idx_m   = a._idx_m;
      _storage = a._storage;
    }

    /// Rebind view 
    void rebind(array_view<value_t const, Rank> const &a) { 
      static_assert(is_const, "Can not rebind a view of const ValueType to a view of ValueType"); 
      _idx_m   = a._idx_m;
      _storage = a._storage;
    }
    //check https://godbolt.org/z/G_QRCU

    //----------------------------------------------------

#include "./_regular_view_common.hpp"
  };

  /*
  template <typename ValueType, int Rank, uint64_t Flags, uint64_t Layout> class array_view : public array_view<ValueType, Rank, 0, Layout> {

    using B = array_view<ValueType, Rank, 0, Layout>;

    public:
   
    using B::B;
    using B::operator=;

    // redefine oeprator() 
 // add a cross constructor

  };
*/
  /// Aliases
  template <typename ValueType, int Rank, uint64_t Guarantees = 0, uint64_t Layout = 0>
  using array_const_view = array_view<ValueType const, Rank, Layout, Guarantees>;

} // namespace nda
