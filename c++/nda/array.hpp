/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2014 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once
#include "./array_view.hpp"

namespace nda {

  // Class template argument deduction
  template <typename T>
  array(T)->array<get_value_t<std::decay_t<T>>, get_rank<std::decay_t<T>>>;

  namespace details {
    template <typename R, typename Initializer, size_t... Is>
    inline constexpr bool _is_a_good_lambda(std::index_sequence<Is...>) {
      return std::is_invocable_r_v<R, Initializer, std::conditional_t<Is, long, long>...>;
    }
  } // namespace details

  // ---------------------- array--------------------------------

  template <typename ValueType, int Rank>
  class array {
    static_assert(!std::is_const<ValueType>::value, "no const type");

    public:
    using value_t   = ValueType;
    using storage_t = mem::handle<ValueType, 'R'>;
    using idx_map_t = idx_map<Rank, 0, flags::contiguous | flags::zero_offset | flags::smallest_stride_is_one>;

    using regular_t    = array<ValueType, Rank>;
    using view_t       = array_view<ValueType, Rank>;
    using const_view_t = array_view<ValueType const, Rank>;

    static constexpr int rank      = Rank;
    static constexpr bool is_const = false;
    static constexpr bool is_view  = false;

    private:
    template <typename IdxMap>
    using my_view_template_t = array_view<value_t, IdxMap::rank(), IdxMap::flags, permutations::encode(IdxMap::layout)>;

    idx_map_t _idx_m;
    storage_t _storage;

    public:
    // ------------------------------- constructors --------------------------------------------

    /// Empty array
    array() = default;

    /// Makes a deep copy, since array is a regular type
    array(array const &x) : _idx_m(x.indexmap()), _storage(x.storage()) {}

    ///
    array(array &&X) = default;

    /** 
     * Construct with a shape [i0, is ...]. 
     * Int must be convertible to long, and there must be exactly Rank arguments.
     * 
     * @param i0, is ... lengths in each dimensions
     */
    template <typename... Int>
    explicit array(long i0, Int... is) {
      //    template <typename... Int> explicit array(long i0, Int... is) : _idx_m{{i0, is...}}, _storage{_idx_m.size()} {
      static_assert(sizeof...(Int) + 1 == Rank, "Incorrect number of arguments : should be exactly Rank. ");
      _idx_m   = idx_map_t{{i0, is...}};
      _storage = storage_t{_idx_m.size()};
      // NB first impl is more natural, but error message in case of false # of parameters (very common)
      // is better like this. FIXME to be tested in benchs
    }

    /** 
     * Construct with the given shape (memory is in C order).
     * @param shape  Shape of the array (lengths in each dimension)
     */
    explicit array(shape_t<Rank> const &shape) : _idx_m(shape), _storage(_idx_m.size()) {}

    /** 
     * [Advanced] Construct from an indexmap and a storage handle.
     *
     * @param idx index map
     * @param mem_handle  memory handle
     * NB: make a new copy.
     */
    template <char RBS>
    array(idx_map<Rank> const &idx, mem::handle<ValueType, RBS> mem_handle) : _idx_m(idx), _storage(std::move(mem_handle)) {}

    /// Construct from anything that has an indexmap and a storage compatible with this class
    //template <typename T> array(T const &a) REQUIRES(XXXX): array(a.indexmap(), a.storage()) {}

    /** 
     * Build a new array from x.shape() and fill it with by evaluating x. 
     * T should model NdArray
     */
    template <typename T>
    array(T const &x) REQUIRES(is_ndarray_v<T> ) : array{x.shape()} {
      static_assert(std::is_convertible_v<get_value_t<T>, value_t>, "Can not construct the array. ValueType can be constructed from the value_t of the argument");
      nda::details::assignment(*this, x);
    }

    /** 
     * [Advanced] From a shape and a storage handle (for reshaping)
     *
     * @param shape  Shape of the array (lengths in each dimension)
     * @param mem_handle  memory handle
     * NB: make a new copy.
     */
    /// From a temporary storage and an indexmap. Used for reshaping a temporary array
    template <char RBS>
    array(shape_t<Rank> const &shape, mem::handle<ValueType, RBS> mem_handle) : array(idx_map_t{shape}, mem_handle) {}

    // --- with initializers

    /**
     * Construct from the initializer list 
     *
     * @tparam T Any type from which ValueType is constructible
     * @param l Initializer list
     *
     * @requires Rank == 1
     * T  Constructor from an initializer list for Rank 1
     */
    template <typename T>
    array(std::initializer_list<T> const &l) //
       REQUIRES((Rank == 1) and std::is_constructible_v<value_t, T>)
       : array{shape_t<Rank>{long(l.size())}} {
      long i = 0;
      for (auto const &x : l) (*this)(i++) = x;
    }

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
     *
     * @requires Rank == 2
     * T  Constructor from an initializer list for Rank 1
     */
    template <typename T>
    array(std::initializer_list<std::initializer_list<T>> const &ll) //
       REQUIRES((Rank == 2) and std::is_constructible_v<value_t, T>)
       : array(_comp_shape_from_list_list(ll)) {
      long i = 0, j = 0;
      for (auto const &l1 : ll) {
        for (auto const &x : l1) { (*this)(i, j++) = x; }
        j = 0;
        ++i;
      }
    }

    /**
     * [Advanced] Construct from shape and a Lambda to initialize the elements. 
     *
     * @tparam Initializer A lambda callable on Rank longs and whose return is convertible to ValueType
     *
     * @param shape  Shape of the array (lengths in each dimension)
     * @param initializer The lambda
     *
     * a(i,j,k,l) is initialized to initializer(i,j,k,l) at construction.
     * Specially useful for non trivially constructible type
     *
     * @example array_init0.cpp
     *
     */
    template <typename Initializer>
    explicit array(shape_t<Rank> const &shape, Initializer initializer)
       REQUIRES(details::_is_a_good_lambda<ValueType, Initializer>(std::make_index_sequence<Rank>()))
       : _idx_m(shape), _storage{_idx_m.size(), mem::do_not_initialize} {
      nda::for_each(_idx_m.lengths(), [&](auto const &... x) { _storage.init_raw(_idx_m(x...), initializer(x...)); });
    }

    //------------------ Assignment -------------------------

    /// Move assignment
    array &operator=(array &&x) = default;

    /// Assignment. Copys the storage. All references to the storage are therefore invalidated.
    array &operator=(array const &X) = default;

    /** 
     * Assignement resizes the array (if necessary).
     * All references to the storage are therefore invalidated.
     * NB : to avoid that, do make_view(A) = X instead of A = X
     */
    template <typename RHS>
    array &operator=(RHS const &rhs) {
      static_assert(is_ndarray_v<RHS> or is_scalar_for_v<RHS,array> , "Assignment : RHS not supported");
      if constexpr (is_ndarray_v<RHS>) resize(rhs.shape());
      nda::details::assignment(*this, rhs);
      return *this;
    }

    //------------------ resize  -------------------------
    /** 
     * Resizes the array. NB : all references to the storage is invalidated.
     * Does not initialize the array
     * Content is undefined
     */
    template <typename... Args>
    void resize(Args const &... args) {
      static_assert(sizeof...(args) == Rank, "Incorrect number of arguments for resize. Should be Rank");
      static_assert(std::is_copy_constructible<ValueType>::value, "Can not resize an array if its value_t is not copy constructible");
      resize(shape_t<Rank>{args...});
    }

    void resize(shape_t<Rank> const &shape) {
      _idx_m = idx_map<Rank>(shape);
      // build a new one with the lengths of IND BUT THE SAME layout !
      // optimisation. Construct a storage only if the new index is not compatible (size mismatch).
      if (_storage.size() != _idx_m.size()) _storage = mem::handle<ValueType, 'R'>{_idx_m.size()};
    }

    // --------------------------

#include "./_regular_view_common.hpp"
  };

} // namespace nda
