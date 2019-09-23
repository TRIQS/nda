#pragma once
#include "storage/policies.hpp"
#include "layout/idx_map.hpp"
#include "layout/policies.hpp"
#include "basic_functions.hpp"
#include "assignment.hpp"
#include "accessors.hpp"
#include "iterators.hpp"
#include "mpi.hpp"
#include <clef/clef.hpp>

namespace nda {

  // ---------------------- declare array and view  --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  class basic_array;

  template <typename ValueType, int Rank, typename Layout, char Algebra = 'A', typename AccessorPolicy = nda::default_accessor, typename OwningPolicy= nda::borrowed>
  class basic_array_view;

  // ---------------------- User aliases  --------------------------------

  template <typename ValueType, int Rank, typename Layout = C_layout>
  using array = basic_array<ValueType, Rank, Layout, 'A', heap>;

  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using array_view = basic_array_view<ValueType, Rank, Layout, 'A', default_accessor, borrowed>;

  template <typename ValueType, typename Layout = C_layout> // CLayout or FLayout
  using matrix = basic_array<ValueType, 2, Layout, 'M', heap>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using matrix_view = basic_array_view<ValueType, 2, Layout, 'M', default_accessor, borrowed>;

  template <typename ValueType, typename Layout = C_layout> // CLayout or FLayout
  using vector = basic_array<ValueType, 1, Layout, 'A', heap>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using vector_view = basic_array_view<ValueType, 1, Layout, 'A', default_accessor, borrowed>;

  // ---------------------- is_array_or_view_container  --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr bool is_regular_v<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = true;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr bool is_view_v<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = true;

  // ---------------------- concept  --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr bool is_ndarray_v<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = true;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr bool is_ndarray_v<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = true;

  // ---------------------- algebra --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr char get_algebra<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = Algebra;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr char get_algebra<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = Algebra;

  // ---------------------- get_layout_info --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr layout_info_t get_layout_info<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> =
     basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>::idx_map_t::layout_info;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr layout_info_t get_layout_info<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> =
     basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>::idx_map_t::layout_info;

  // ---------------------- basic_array_view  --------------------------------

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
    using regular_t = basic_array<ValueType, Rank, layout<idx_map_t::stride_order_encoded, layout_prop_e::contiguous>, Algebra, heap>;
    ///
    using view_t = basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>;
    ///
    using const_view_t = basic_array_view<ValueType const, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>;
    ///
    using no_const_view_t = basic_array_view<std::remove_const_t<ValueType>, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>;

    static constexpr int rank      = Rank;
    static constexpr bool is_view  = true;
    static constexpr bool is_const = std::is_const_v<ValueType>;

    //    static constexpr uint64_t stride_order = StrideOrder;

    // FIXME : h5
    // static std::string hdf5_format() { return "array<" + triqs::h5::get_hdf5_format<ValueType>() + "," + std::to_string(rank) + ">"; }

    private:
    template <typename IdxMap>
    using my_view_template_t =
       basic_array_view<ValueType, IdxMap::rank(), layout<IdxMap::stride_order_encoded, IdxMap::layout_prop>, Algebra, AccessorPolicy, OwningPolicy>;

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
    basic_array_view(basic_array_view const &v) REQUIRES(is_const) : basic_array_view(v.indexmap(), v.storage()) {}

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
    template <typename A> //explicit
    basic_array_view(A const &a) REQUIRES(is_regular_or_view_v<A>) : basic_array_view(a.indexmap(), a.storage()) {}

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
    void rebind(no_const_view_t const &a) REQUIRES(is_const) {
      //static_assert(is_const, "Can not rebind a view of const ValueType to a view of ValueType");
      _idx_m   = idx_map_t{a.indexmap()};
      _storage = storage_t{a.storage()};
    }
    //check https://godbolt.org/z/G_QRCU

    //----------------------------------------------------

#include "./_regular_view_common.hpp"
  };

} // namespace nda
