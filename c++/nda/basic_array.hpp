#pragma once
#include <algorithm>
#include "basic_array_view.hpp"

namespace nda {

  /// Class template argument deduction
  template <typename T>
  basic_array(T) -> basic_array<get_value_t<std::decay_t<T>>, get_rank<std::decay_t<T>>, C_layout, 'A', heap>;

  // forward for friend declaration
  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename NewLayoutType>
  auto map_layout_transform(basic_array<T, R, L, Algebra, ContainerPolicy> &&a, NewLayoutType const &new_layout);

  // ---------------------- array--------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  class basic_array {

    static_assert(!std::is_const<ValueType>::value, "ValueType can not be const. WHY ?");

    // details for the common code with view
    using self_t                   = basic_array;
    using AccessorPolicy           = default_accessor;
    using OwningPolicy             = borrowed;
    static constexpr bool is_const = false;
    static constexpr bool is_view  = false;

    public:
    ///
    using value_type = ValueType;

    // FIXME layout_t
    using layout_t = typename Layout::template mapping<Rank>;

    private:
    // FIXME : mem_handle_t
    using storage_t = typename ContainerPolicy::template handle<ValueType, layout_t::ce_size()>;

    public:
    static constexpr int rank = Rank;

    private:
    layout_t lay;
    storage_t sto;

    template <typename T, int R, typename L, char A, typename C, typename NewLayoutType>
    friend auto map_layout_transform(basic_array<T, R, L, A, C> &&a, NewLayoutType const &new_layout);

    // private constructor for the friend
    basic_array(layout_t const &idxm, storage_t &&mem_handle) noexcept : lay(idxm), sto(std::move(mem_handle)) {}

    public:
    // ------------------------------- constructors --------------------------------------------

    /// Empty array
    basic_array() = default;

    /// Makes a deep copy, since array is a regular type
    basic_array(basic_array const &x) noexcept : lay(x.indexmap()), sto(x.sto) {}

    ///
    basic_array(basic_array &&X) = default;

    /** 
     * Construct with a shape [i0, is ...]. 
     * Int are integers (convertible to long), and there must be exactly R arguments.
     * 
     * @param i0, is ... are the extents (lengths) in each dimension
     */
    template <CONCEPT(std::integral)... Int>
    explicit basic_array(Int... is) noexcept REQUIRES17((std::is_convertible_v<Int, long> and ...)) {
      //static_assert((std::is_convertible_v<Int, long> and ...), "Arguments must be convertible to long");
      static_assert(sizeof...(Int) == Rank, "Incorrect number of arguments : should be exactly Rank. ");
      lay = layout_t{{long(is)...}};
      sto = storage_t{lay.size()};
      // It would be more natural to construct lay, storage from the start, but the error message in case of false # of parameters (very common)
      // is better like this. FIXME to be tested in benchs
    }

    /** 
     * Construct with the given shape and default construct elements
     * 
     * @param shape  Shape of the array (lengths in each dimension)
     */
    explicit basic_array(std::array<long, Rank> const &shape) noexcept REQUIRES(std::is_default_constructible_v<ValueType>)
       : lay(shape), sto(lay.size()) {}

    /** 
     * Constructs from a.shape() and then assign from the evaluation of a
     */
    template <CONCEPT(ArrayOfRank<Rank>) A>
    basic_array(A const &a) noexcept REQUIRES17(is_ndarray_v<A>) : lay(a.shape()), sto{lay.size(), mem::do_not_initialize} {
      static_assert(std::is_convertible_v<get_value_t<A>, value_type>,
                    "Can not construct the array. ValueType can not be constructed from the value_type of the argument");
      if constexpr (std::is_trivial_v<ValueType> or mem::is_complex_v<ValueType>) {
        // simple type. the initialization was not necessary anyway.
        // we use the assign, including the optimization (1d strided, contiguous) possibly
        assign_from_ndarray(a);
      } else {
        // in particular ValueType may or may not be default constructible
        // so we do not init memory, and make the placement new now, directly with the value returned by a
        nda::for_each(lay.lengths(), [&](auto const &... is) { new (sto.data() + lay(is...)) ValueType{a(is...)}; });
      }
    }

    /** 
     * Initialize with any type modelling ArrayInitializer, typically a 
     * delayed operation (mpi operation, matmul) that requires 
     * the knowledge of the data pointer to execute
     *
     */
    template <CONCEPT(ArrayInitializer) Initializer> // can not be explicit
    basic_array(Initializer const &initializer) noexcept(noexcept(initializer.invoke(*this))) REQUIRES17(is_assign_rhs<Initializer>)
       : basic_array{initializer.shape()} {
      initializer.invoke(*this);
    }

    private: // impl. detail for next function
    static std::array<long, 1> shape_from_init_list(std::initializer_list<ValueType> const &l) noexcept { return {long(l.size())}; }

    template <typename L>
    static auto shape_from_init_list(std::initializer_list<L> const &l) noexcept {
      const auto [min, max] =
         std::minmax_element(std::begin(l), std::end(l), [](auto &&x, auto &&y) { return shape_from_init_list(x) == shape_from_init_list(y); });
      EXPECTS_WITH_MESSAGE(shape_from_init_list(*min) == shape_from_init_list(*max), "initializer list not rectangular !");
      return nda::front_append(shape_from_init_list(*max), long(l.size()));
    }

    public:
    ///
    basic_array(std::initializer_list<ValueType> const &l) noexcept //
       REQUIRES(Rank == 1)
       : lay(std::array<long, 1>{long(l.size())}), sto{lay.size(), mem::do_not_initialize} {
      long i = 0;
      // We can not assume that ValueType is default constructible. As before, we do not initialize,
      // and use placement new
      // https://godbolt.org/z/Lwic2o. Same code as = for basic type
      // Alternative : if constexpr (std::is_trivial_v<ValueType> or mem::is_complex<ValueType>::value) for (auto const &x : l) *(sto.data() + lay(i++)) = x;
      for (auto const &x : l) { new (sto.data() + lay(i++)) ValueType{x}; }
    }

    ///
    basic_array(std::initializer_list<std::initializer_list<ValueType>> const &l2) noexcept //
       REQUIRES((Rank == 2))
       : lay(shape_from_init_list(l2)), sto{lay.size(), mem::do_not_initialize} {
      long i = 0, j = 0;
      for (auto const &l1 : l2) {
        for (auto const &x : l1) { new (sto.data() + lay(i, j++)) ValueType{x}; } // cf dim1
        j = 0;
        ++i;
      }
    }

    ///
    basic_array(std::initializer_list<std::initializer_list<std::initializer_list<ValueType>>> const &l3) noexcept //
       : lay(shape_from_init_list(l3)), sto{lay.size(), mem::do_not_initialize} {
      long i = 0, j = 0, k = 0;
      static_assert(Rank == 3, "?");
      for (auto const &l2 : l3) {
        for (auto const &l1 : l2) {
          for (auto const &x : l1) { new (sto.data() + lay(i, j, k++)) ValueType{x}; } // cf dim1
          k = 0;
          ++j;
        }
        j = 0;
        ++i;
      }
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
    template <CONCEPT(ArrayOfRank<Rank>) RHS>
    basic_array &operator=(RHS const &rhs) noexcept REQUIRES17(is_ndarray_v<RHS>) {
      static_assert(!is_const, "Cannot assign to a const !");
      resize(rhs.shape());
      assign_from_ndarray(rhs); // common code with view, private
      return *this;
    }

    /** 
     * Resizes the array (if necessary).
     * Invalidates all references to the storage.
     *
     * @tparam RHS A scalar or an object modeling NdArray
     */
    template <typename RHS>
    // FIXME : explode this notion
    basic_array &operator=(RHS const &rhs) noexcept REQUIRES(is_scalar_for_v<RHS, basic_array>) {
      static_assert(!is_const, "Cannot assign to a const !");
      assign_from_scalar(rhs); // common code with view, private
      return *this;
    }

    /** 
     * 
     */
    template <CONCEPT(ArrayInitializer) Initializer>
    basic_array &operator=(Initializer const &initializer) noexcept REQUIRES17(is_assign_rhs<Initializer>) {
      resize(initializer.shape());
      initializer.invoke(*this);
      return *this;
    }

    //------------------ resize  -------------------------
    /** 
     * Resizes the array.
     * Invalidates all references to the storage.
     * Content is undefined, makes no copy of previous data.
     *
     */
    template <CONCEPT(std::integral)... Int>
    void resize(Int const &... extent) REQUIRES17(std::is_convertible_v<Int, long> and...) {
      static_assert(std::is_copy_constructible_v<ValueType>, "Can not resize an array if its value_type is not copy constructible");
      static_assert(sizeof...(extent) == Rank, "Incorrect number of arguments for resize. Should be Rank");
      resize(std::array<long, Rank>{long(extent)...});
    }

    /** 
     * Resizes the array.
     * Invalidates all references to the storage.
     * Content is undefined, makes no copy of previous data.
     *
     * @param shape  New shape of the array (lengths in each dimension)
     */
    [[gnu::noinline]] void resize(std::array<long, Rank> const &shape) {
      lay = layout_t(shape);
      // Construct a storage only if the new index is not compatible (size mismatch).
      if (sto.is_null() or (sto.size() != lay.size())) sto = storage_t{lay.size()};
    }

#include "./_impl_basic_array_view_common.hpp"
  };

} // namespace nda
