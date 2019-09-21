#pragma once
#include "./array_view.hpp"

namespace nda {

  // // UNCOMMENT_FOR_MATRIX
  // /// Class template argument deduction
  // template <typename T>
  // matrix(T)->matrix<get_value_t<std::decay_t<T>>>;

  // BEGIN_REMOVE_FOR_MATRIX
  // Class template argument deduction
  template <typename T>
  basic_array(T)->basic_array<get_value_t<std::decay_t<T>>, get_rank<std::decay_t<T>>, C_layout, 'A', heap>;

  // FIXME : in array as static ?
  namespace details {

    template <int Is>
    using _long_anyway = long; // to unpack below

    template <typename R, typename Initializer, size_t... Is>
    inline constexpr bool _is_a_good_lambda_for_init(std::index_sequence<Is...>) {
      return std::is_invocable_r_v<R, Initializer, _long_anyway<Is>...>;
    }
  } // namespace details
  // END_REMOVE_FOR_MATRIX

  // ---------------------- array--------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  class basic_array {
    static_assert(!std::is_const<ValueType>::value, "ValueType can not be const. WHY ?");

    public:
    ///
    using value_t    = ValueType;
    using value_type = ValueType;

    ///
    using regular_t = basic_array;
    ///
    using view_t = basic_array_view<ValueType, Rank, Layout, Algebra, default_accessor, borrowed>;
    ///
    using const_view_t = basic_array_view<ValueType const, Rank, Layout, Algebra, default_accessor, borrowed>;

    using storage_t = heap::handle<ValueType>;
    using idx_map_t = typename Layout::template mapping<Rank>;

    //    static constexpr uint64_t stride_order = StrideOrder;
    //  static constexpr layout_prop_e stride_order= layout_prop_e::contiguous;

    static constexpr int rank      = Rank;
    static constexpr bool is_const = false;
    static constexpr bool is_view  = false;

    private:
    // details for the common code with view
    using AccessorPolicy = default_accessor;

    template <typename IdxMap>
    using my_view_template_t =
       basic_array_view<ValueType, IdxMap::rank(), layout<IdxMap::stride_order_encoded, IdxMap::layout_prop>, Algebra, default_accessor, borrowed>;

    idx_map_t _idx_m;
    storage_t _storage;

    public:
    // ------------------------------- constructors --------------------------------------------

    /// Empty array
    basic_array() = default;

    /// Makes a deep copy, since array is a regular type
    basic_array(basic_array const &x) : _idx_m(x.indexmap()), _storage(x.storage()) {}

    ///
    basic_array(basic_array &&X) = default;

    /** 
     * Construct with a shape [i0, is ...]. 
     * Int must be convertible to long, and there must be exactly Rank arguments.
     * 
     * @param i0, is ... lengths in each dimensions
     * @example array_constructors
     */
    template <typename... Int>
    explicit basic_array(long i0, Int... is) {
      static_assert((std::is_convertible_v<Int, long> and ...), "Arguments must be convertible to long");
      static_assert(sizeof...(Int) + 1 == Rank, "Incorrect number of arguments : should be exactly Rank. ");
      _idx_m   = idx_map_t{{i0, is...}};
      _storage = storage_t{_idx_m.size()};
      // It would be more natural to construct _idx_m, storage from the start, but the error message in case of false # of parameters (very common)
      // is better like this. FIXME to be tested in benchs
    }

    /** 
     * Construct with the given shape
     * 
     * @param shape  Shape of the array (lengths in each dimension)
     */
    explicit basic_array(shape_t<Rank> const &shape) : _idx_m(shape), _storage(_idx_m.size()) {}

    /** 
     * [Advanced] Construct from an indexmap and a storage handle.
     *
     * @param idxm index map
     * @param mem_handle  memory handle
     * NB: make a new copy.
     */
    //template <char RBS>
    //array(idx_map<Rank> const &idxm, mem::handle<ValueType, RBS> mem_handle) : _idx_m(idxm), _storage(std::move(mem_handle)) {}

    /// Construct from anything that has an indexmap and a storage compatible with this class
    //template <typename T> basic_array(T const &a) REQUIRES(XXXX): basic_array(a.indexmap(), a.storage()) {}

    /** 
     * From any type modeling NdArray
     * Constructs from x.shape() and then assign from the evaluation of x.
     * 
     * @tparam A A type modeling NdArray
     * @param x 
     */
    template <typename A>
    basic_array(A const &x) REQUIRES(is_ndarray_v<A>) : basic_array{x.shape()} {
      static_assert(std::is_convertible_v<get_value_t<A>, value_t>,
                    "Can not construct the array. ValueType can not be constructed from the value_t of the argument");
      assign_from(*this, x);
    }

    /** 
     * [Advanced] From any type which has a .shape() and can be assigned from
     * Used for simple lazy operations, like mpi, and or other lazy transformation.
     * Constructs from x.shape() and use assign. Cf code
     * 
     * @tparam Lazy A type modeling IsAssignRHS
     * @param lazy 
     */
    template <typename Lazy>
    basic_array(Lazy const &lazy) REQUIRES(is_assign_rhs<Lazy>) : basic_array{lazy.shape()} {
      assign_from(*this, lazy);
    }

    /** 
     * [Advanced] From a shape and a storage handle (for reshaping)
     * NB: make a new copy.
     *
     * @param shape  Shape of the array (lengths in each dimension)
     * @param mem_handle  memory handle
     */
    //template <char RBS>
    //basic_array(shape_t<Rank> const &shape, mem::handle<ValueType, RBS> mem_handle) : basic_array(idx_map_t{shape}, mem_handle) {}

    // --- with initializers

    // BEGIN_REMOVE_FOR_MATRIX
    /**
     * Construct from the initializer list 
     *
     * @tparam T Any type from which ValueType is constructible
     * @param l Initializer list
     *
     * @requires Rank == 1 and ValueType is constructible from T
     */
    template <typename T>
    basic_array(std::initializer_list<T> const &l) //
       REQUIRES((Rank == 1) and std::is_constructible_v<value_t, T>)
       : basic_array{shape_t<Rank>{long(l.size())}} {
      long i = 0;
      for (auto const &x : l) (*this)(i++) = x;
    }
    // END_REMOVE_FOR_MATRIX

    private: // impl. detail for next function
    template <typename T>
    static shape_t<2> _comp_shape_from_list_list(std::initializer_list<std::initializer_list<T>> const &ll) {
      long s = -1;
      for (auto const &l1 : ll) {
        if (s == -1)
          s = l1.size();
        else if (s != l1.size())
          throw std::runtime_error("initializer list not rectangular !");
      }
      return {long(ll.size()), s};
    }

    public:
    /**
     * Construct from the initializer list of list 
     *
     * @tparam T Any type from which ValueType is constructible
     * @param ll Initializer list of list
     * @requires Rank == 2 and ValueType is constructible from T
     */
    template <typename T>
    basic_array(std::initializer_list<std::initializer_list<T>> const &ll) //
       REQUIRES((Rank == 2) and std::is_constructible_v<value_t, T>)
       : basic_array(_comp_shape_from_list_list(ll)) {
      long i = 0, j = 0;
      for (auto const &l1 : ll) {
        for (auto const &x : l1) { (*this)(i, j++) = x; }
        j = 0;
        ++i;
      }
    }

    /**
     * [Advanced] Construct from shape and a Lambda to initialize the elements. 
     * a(i,j,k,...) is initialized to initializer(i,j,k,...) at construction.
     * Specially useful for non trivially constructible type
     *
     * @tparam Initializer  a callable on Rank longs which returns something is convertible to ValueType
     * @param shape  Shape of the array (lengths in each dimension)
     * @param initializer The lambda
     */
    template <typename Initializer>
    explicit basic_array(shape_t<Rank> const &shape, Initializer initializer)
       REQUIRES(details::_is_a_good_lambda_for_init<ValueType, Initializer>(std::make_index_sequence<Rank>()))
       : _idx_m(shape), _storage{_idx_m.size(), mem::do_not_initialize} {
      nda::for_each(_idx_m.lengths(), [&](auto const &... x) { _storage.init_raw(_idx_m(x...), initializer(x...)); });
    }

    //------------------ Assignment -------------------------

    ///
    basic_array &operator=(basic_array &&x) = default;

    /// Deep copy (array is a regular type). Invalidates all references to the storage.
    basic_array &operator=(basic_array const &X) = default;

    /** 
     * Resizes the array (if necessary).
     * Invalidates all references to the storage.
     *
     * @tparam RHS A scalar or an object modeling NdArray
     */
    template <typename RHS>
    basic_array &operator=(RHS const &rhs) {
      static_assert(is_ndarray_v<RHS> or is_scalar_for_v<RHS, basic_array>, "Assignment : RHS not supported");
      if constexpr (is_ndarray_v<RHS>) resize(rhs.shape());
      assign_from(*this, rhs);
      return *this;
    }

    /** 
     * [Advanced] Assign from any type which has a .shape() and can be assigned from
     * Used for simple lazy operations, like mpi, and or other lazy transformation.
     * 
     * @tparam Lazy A type modeling NdArray
     * @param lazy 
     */
    template <typename Lazy>
    basic_array &operator=(Lazy const &lazy) REQUIRES(is_assign_rhs<Lazy>) {
      resize(lazy.shape());
      assign_from(*this, lazy);
      return *this;
    }

    //------------------ resize  -------------------------
    /** 
     * Resizes the array.
     * Invalidates all references to the storage.
     * Content is undefined, makes no copy of previous data.
     *
     * @tparam Int Integer type
     * @param i0 New dimension
     * @param is New dimension
     */
    template <typename... Int>
    void resize(long i0, Int const &... is) {
      static_assert((std::is_convertible_v<Int, long> and ...), "Arguments must be convertible to long");
      static_assert(sizeof...(is) + 1 == Rank, "Incorrect number of arguments for resize. Should be Rank");
      static_assert(std::is_copy_constructible_v<ValueType>, "Can not resize an array if its value_t is not copy constructible");
      resize(shape_t<Rank>{i0, is...});
    }

    /** 
     * Resizes the array.
     * Invalidates all references to the storage.
     * Content is undefined, makes no copy of previous data.
     *
     * @param shape  New shape of the array (lengths in each dimension)
     */
    [[gnu::noinline]] void resize(shape_t<Rank> const &shape) {
      _idx_m = idx_map_t(shape);
      // Construct a storage only if the new index is not compatible (size mismatch).
      if (_storage.is_null() or (_storage.size() != _idx_m.size())) _storage = mem::handle<ValueType, 'R'>{_idx_m.size()};
    }

    // --------------------------

#include "./_regular_view_common.hpp"
  };

} // namespace nda
