// Copyright (c) 2018-2021 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include <algorithm>
#include <memory>
#include <ranges>
#include <random>
#include "basic_array_view.hpp"

namespace nda {

  /// Class template argument deduction
  template <typename T>
  basic_array(T) -> basic_array<get_value_t<std::decay_t<T>>, get_rank<std::decay_t<T>>, C_layout, 'A', heap<>>;

  // forward for friend declaration
  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename NewLayoutType>
  auto map_layout_transform(basic_array<T, R, L, Algebra, ContainerPolicy> &&a, NewLayoutType const &new_layout);

  // ---------------------- array--------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  class basic_array {

    static_assert(!std::is_const<ValueType>::value, "ValueType of basic_array cannot be const.");
    static_assert((Algebra != 'M') or (Rank == 2), " Internal error : Algebra 'M' only makes sense for rank 2");
    static_assert((Algebra != 'V') or (Rank == 1), " Internal error : Algebra 'V' only makes sense for rank 1");

    public:
    /// Type of the array's values
    using value_type          = ValueType;
    /// Type of the memory handle
    using storage_t           = typename ContainerPolicy::template handle<ValueType>;
    /// Type of the memory layout
    using layout_t            = typename Layout::template mapping<Rank>;
    /// The associated regular type
    using regular_type        = basic_array;
    /// The number of dimensions of the array
    static constexpr int rank = Rank;

    static_assert(has_contiguous(layout_t::layout_prop), "Non sense. A basic_array is a contiguous object");

    private:
    // details for the common code with view
    using self_t                   = basic_array;
    using AccessorPolicy           = default_accessor; // no_alias_accessor
    using OwningPolicy             = borrowed<storage_t::address_space>;
    static constexpr bool is_const = false;
    static constexpr bool is_view  = false;

    layout_t lay;
    storage_t sto;

    template <typename U, int R, typename L, char A, typename C, typename NewLayoutType>
    friend auto map_layout_transform(basic_array<U, R, L, A, C> &&a, NewLayoutType const &new_layout);

    // private constructor for the friend
    basic_array(layout_t const &idxm, storage_t &&mem_handle) noexcept : lay{idxm}, sto{std::move(mem_handle)} {}

    template <std::integral Int = long>
    basic_array(std::array<Int, Rank> const &shape, mem::init_zero_t) noexcept : lay{shape}, sto{lay.size(), mem::init_zero} {}

    public:
    // backward : FIXME : temporary to be removed
    [[deprecated]] basic_array_view<ValueType, Rank, Layout, 'A', AccessorPolicy, OwningPolicy> as_array_view() { return {*this}; };
    [[deprecated]] basic_array_view<const ValueType, Rank, Layout, 'A', AccessorPolicy, OwningPolicy> as_array_view() const { return {*this}; };

    [[deprecated]] auto transpose() requires(Rank == 2) { return permuted_indices_view<encode(std::array<int, 2>{1, 0})>(*this); }
    [[deprecated]] auto transpose() const requires(Rank == 2) { return permuted_indices_view<encode(std::array<int, 2>{1, 0})>(*this); }

    // ------------------------------- constructors --------------------------------------------

    /// Empty array
    // Caution! We need to provide a user-defined constructor (over =default)
    // to avoid value initialization of the sso buffer
    basic_array(){};

    /// Makes a deep copy, since array is a regular type
    explicit basic_array(basic_array const &x) noexcept : lay(x.indexmap()), sto(x.sto) {}

    /// Makes a deep copy, given a basic_array with a different container policy
    template <char Algebra_other, typename ContainerPolicy_other>
    explicit basic_array(basic_array<ValueType, Rank, Layout, Algebra_other, ContainerPolicy_other> const &x) noexcept : lay(x.indexmap()), sto(x.storage()) {}

    ///
    basic_array(basic_array &&X) = default;

    /** 
     * Construct with a shape [i0, is ...]. 
     * Int are integers (convertible to long), and there must be exactly R arguments.
     * 
     * @param i0, is ... are the extents (lengths) in each dimension
     */
    template <std::integral... Int>
    explicit basic_array(Int... is) noexcept {
      //static_assert((std::is_convertible_v<Int, long> and ...), "Arguments must be convertible to long");
      static_assert(sizeof...(Int) == Rank, "Incorrect number of arguments : should be exactly Rank. ");
      lay = layout_t{std::array{long(is)...}};
      sto = storage_t{lay.size()};
      // It would be more natural to construct lay, storage from the start, but the error message in case of false # of parameters (very common)
      // is better like this. FIXME to be tested in benchs
    }

    /**
     * Construct one-dimensional array with a shape [i0]
     * with all elements initialized to val
     * Int is an integer (convertible to long)
     *
     * @param i0 is the extents of the only dimension
     */
    template <std::integral Int, typename RHS>
    explicit basic_array(Int i, RHS const &val) noexcept requires((Rank == 1 and is_scalar_for_v<RHS, basic_array>)) {
      lay = layout_t{std::array{long(i)}};
      sto = storage_t{lay.size()};
      assign_from_scalar(val);
    }

    /** 
     * Construct with the given shape and default construct elements
     * 
     * @param shape  Shape of the array (lengths in each dimension)
     */
    template <std::integral Int = long>
    explicit basic_array(std::array<Int, Rank> const &shape) noexcept requires(std::is_default_constructible_v<ValueType>)
       : lay(shape), sto(lay.size()) {}

    /// Construct from the layout
    explicit basic_array(layout_t const &layout) noexcept requires(std::is_default_constructible_v<ValueType>) : lay(layout), sto(lay.size()) {}

    /** 
     * Constructs from a.shape() and then assign from the evaluation of a
     */
    template <ArrayOfRank<Rank> A>
    basic_array(A const &a) noexcept //
       requires(HasValueTypeConstructibleFrom<A, value_type>)
          
       : lay(a.shape()), sto{lay.size(), mem::do_not_initialize} {
      static_assert(std::is_constructible_v<value_type, get_value_t<A>>,
                    "Can not construct the array. ValueType can not be constructed from the value_type of the argument");
      if constexpr (std::is_trivial_v<ValueType> or mem::is_complex_v<ValueType>) {
        // simple type. the initialization was not necessary anyway.
        // we use the assign, including the optimization (1d strided, contiguous) possibly
        assign_from_ndarray(a);
      } else {
        // in particular ValueType may or may not be default constructible
        // so we do not init memory, and make the placement new now, directly with the value returned by a
        nda::for_each(lay.lengths(), [&](auto const &...is) { new (sto.data() + lay(is...)) ValueType{a(is...)}; });
      }
    }

    /** 
     * Initialize with any type modelling ArrayInitializer, typically a 
     * delayed operation (mpi operation, matmul) that requires 
     * the knowledge of the data pointer to execute
     *
     */
    template <ArrayInitializer Initializer> // can not be explicit
    basic_array(Initializer const &initializer) noexcept(noexcept(initializer.invoke(basic_array{}))) 
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
      return stdutil::front_append(shape_from_init_list(*max), long(l.size()));
    }

    public:
    ///
    basic_array(std::initializer_list<ValueType> const &l) noexcept //
       requires(Rank == 1)
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
       requires(Rank == 2)
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
       requires(Rank == 3)
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

    /////
    //template <typename U>
    //explicit basic_array(std::initializer_list<std::initializer_list<U>> const &l2) noexcept //
    //requires((Rank == 1) and (std::is_constructible_v<ValueType, std::initializer_list<U>>))
    //: lay(l2.size()), sto{lay.size(), mem::do_not_initialize} {
    //long i = 0;
    //for (auto const &l1 : l2) { new (sto.data() + lay(i++)) ValueType{l1}; }
    //}

    /////
    //template <typename U>
    //explicit basic_array(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const &l3) noexcept //
    //requires((Rank == 2) and (std::is_constructible_v<ValueType, std::initializer_list<U>>))
    //: lay(pop(shape_from_init_list(l3))), sto{lay.size(), mem::do_not_initialize} {
    //long i = 0, j = 0;
    //static_assert(Rank == 3, "?");
    //for (auto const &l2 : l3) {
    //for (auto const &l1 : l2) { new (sto.data() + lay(i, j++)) ValueType{l1}; }
    //j = 0;
    //++i;
    //}
    /*}*/

    /// Allows to move a array of Rank 2 into a matrix and vice versa
    /// Beware that for stack/sso array, it will copy the data (but move them for heap allocated array).
    /// \trailing_requires
    template <char Algebra2>
    explicit basic_array(basic_array<ValueType, 2, Layout, Algebra2, ContainerPolicy> &&am) noexcept
       requires(Rank == 2) // NB Rank =2 since matrix/array for the moment. generalize if needed
       : basic_array{am.indexmap(), std::move(am).storage()} {}

    //------------------ Factory -------------------------

    /// Make an array of the given dimensions and zero-initialized values / memory
    template <std::integral Int = long>
    static basic_array zeros(std::array<Int, Rank> const &shape)
       requires(std::is_standard_layout_v<ValueType> &&std::is_trivially_copyable_v<ValueType>) {
      return basic_array{stdutil::make_std_array<long>(shape), mem::init_zero};
    }

    /// Make an array of the given dimensions holding 'scalar ones'
    template <std::integral Int = long>
    static basic_array ones(std::array<Int, Rank> const &shape) requires(nda::is_scalar_v<ValueType>) {
      auto res = basic_array{stdutil::make_std_array<long>(shape)};
      res()    = ValueType{1};
      return res;
    }

    /// Create an array the given dimensions and populate it with random
    /// samples from a uniform distribution over [0, 1)
    template <std::integral Int = long>
    static basic_array rand(std::array<Int, Rank> const &shape) requires(std::is_floating_point_v<ValueType>) {

      auto static gen  = std::mt19937(std::random_device{}());
      auto static dist = std::uniform_real_distribution<>(0.0, 1.0);

      auto res = basic_array{shape};
      for (auto &x : res) x = dist(gen);

      return res;
    }

    //------------------ Assignment -------------------------

    ///
    basic_array &operator=(basic_array &&x) = default;

    /// Deep copy (array is a regular type). Invalidates all references to the storage.
    basic_array &operator=(basic_array const &X) = default;

    /// Deep copy assignment given array with different algebra and/or container policy
    template <char Algebra_other, typename ContainerPolicy_other>
    basic_array &operator=(basic_array<ValueType, Rank, Layout, Algebra_other, ContainerPolicy_other> const &x) {
      *this = basic_array{x};
      return *this;
    }

    /** 
     * Resizes the array (if necessary).
     * Invalidates all references to the storage.
     *
     * @tparam RHS A scalar or an object modeling NdArray
     */
    template <ArrayOfRank<Rank> RHS>
    basic_array &operator=(RHS const &rhs) noexcept  {
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
    basic_array &operator=(RHS const &rhs) noexcept requires(is_scalar_for_v<RHS, basic_array>) {
      static_assert(!is_const, "Cannot assign to a const !");
      assign_from_scalar(rhs); // common code with view, private
      return *this;
    }

    /** 
     * 
     */
    template <ArrayInitializer Initializer>
    basic_array &operator=(Initializer const &initializer) noexcept  {
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
    template <std::integral... Int>
    void resize(Int const &...extent) {
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

#include "./layout_transforms.hpp"
