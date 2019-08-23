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

  // ---------------------- array--------------------------------

  template <typename ValueType, int Rank> class array<ValueType, Rank> : tag::array {
    static_assert(!std::is_const<ValueType>::value, "no const type");

    public:
    using value_t   = ValueType;
    using storage_t = mem::handle<ValueType, 'R'>;
    using idx_m_t   = idx_map<Rank>;
    using shape_t   = idx_map<Rank>::l_t;

    using regular_type    = array<ValueType, Rank>;
    using view_type       = array_view<ValueType, Rank>;
    using const_view_type = array_view<ValueType const, Rank>;

    //static constexpr int rank      = Rank;
    //static constexpr bool is_const = IsConst;
    // value_type not in concept

    private:
    idx_m_t _idx_m;
    storage_t _storage;

    // ------------------------------- constructors --------------------------------------------

    /// Empty array
    array() = default;

    /// Copy constructor. Makes a deep copy
    array(array const &x) : _idx_m(x.indexmap(), x.storage()) {}

    /// Move
    array(array &&X) = default;

    /// Construct from an indexmap and a storage handle. Always make a new copy.
    template <char RBS> array(idx_map<Rank> im, mem::handle<ValueType, RBS> st) : _idx_m(std::move(im)), _storage(std::move(st)) {}

    /// Build from anything that has an indexmap and a storage compatible with this class
    template <typename T> array(T const &X) : array(X.indexmap(), X.storage()) {}

    /// Empty array.
    //explicit array(layout_t<Rank> ml = layout_t<Rank>{}) : array(ml) {}

    /// From a shape
    explicit array(shape_t const &shape) : _idx_m(shape), _storage(_idx_m.size()) {}

    /// From a shape and a layout
    explicit array(shape_t const &shape, layout_t<Rank> ml) : _idx_m(shape, ml), _storage(_idx_m.size()) {}

    /// From shape and a Lambda to initialize the element
    template <typename InitLambda>
    explicit array(shape_t const &shape, layout_t<Rank> ml, InitLambda &&lambda)
       : _idx_m(shape, ml), _storage{_idx_m.size(), mem::do_not_initialize} {
      nda::for_each(_idx_m.lengths(), [&](auto const &... x) { _storage.init_raw(_idx_m(x...), lambda(x...)); });
    }
    
    /// From shape and a Lambda to initialize the element
    template <typename InitLambda>
    explicit array(shape_t const &shape, InitLambda &&lambda) : array(shape, std::forward<InitLambda>(lambda)){}


    /// For lengths as integer and optionally layout::C_t or Fortran_t
    template <typename... T> explicit array(long i0, T... is) : _idx_m{i0, is...}, _storage{_idx_m.size()} {}

    // Makes a true (deep) copy of the data with a different layout
    explicit array(array const &X, layout_t<Rank> ml) : array(X.indexmap(), ml) { triqs_arrays_assign_delegation(*this, X); }

    // from a temporary storage and an indexmap. Used for reshaping a temporary array
    explicit array(typename indexmap_type::domain_type const &dom, storage_type &&sto, layout_t<Rank> ml = layout_t<Rank>{})
       : IMPL_TYPE(indexmap_type(dom, ml), std::move(sto)) {}

    /** 
     * Build a new array from X.domain() and fill it with by evaluating X. X can be : 
     *  - another type of array, array_view, matrix,.... (any <IndexMap, Storage> pair)
     *  - the memory layout will be as given (ml)
     *  - a expression : e.g. array<int> A = B+ 2*C;
     */
    template <typename T>
    array(T const &X, layout_t<Rank> ml) //
       REQUIRES(ImmutableCuboidArray<T>::value &&std::is_convertible<typename T::value_type, value_type>::value)
       : array{get_shape(X), ml} {
      triqs_arrays_assign_delegation(*this, X);
    }

    /** 
     * Build a new array from X.domain() and fill it with by evaluating X. X can be : 
     *  - another type of array, array_view, matrix,.... (any <IndexMap, Storage> pair)
     *  - the memory layout will be deduced from X if possible, or default constructed
     *  - a expression : e.g. array<int> A = B+ 2*C;
     */
    template <typename T>
    array(T const &X) //
       REQUIRES(ImmutableCuboidArray<T>::value &&std::is_convertible<typename T::value_type, value_type>::value)
       : array{get_shape(X), get_memory_layout<Rank, T>::invoke(X)} {
      triqs_arrays_assign_delegation(*this, X);
    }

    /// Constructor from an initializer list for Rank 1
    template <typename T>
    array(std::initializer_list<T> const &l) //
       REQUIRES((Rank == 1) and std::is_constructible<value_type, T>::value >)
       : array{shape_t{l.size()}} {
      long i = 0;
      for (auto const &x : l) (*this)(i++) = x;
    }

    /// Constructor from an initializer list of list for Rank 2
    template <typename T, int R = Rank>
    array(std::initializer_list<std::initializer_list<T>> const &l) //
       REQUIRES((Rank == 2) and std::is_constructible<value_type, T>::value >) {
      long i = 0, j = 0;
      int s = -1;
      for (auto const &l1 : l) {
        if (s == -1)
          s = l1.size();
        else if (s != l1.size())
          throw std::runtime_error("initializer list not rectangular !");
      }
      this->resize(shape_t{l.size(), s});
      for (auto const &l1 : l) {
        for (auto const &x : l1) { (*this)(i, j++) = x; }
        j = 0;
        ++i;
      }
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
    template <typename RHS> array &operator=(RHS const &X) {
      static_assert(ImmutableCuboidArray<RHS>::value, "Assignment : RHS not supported");
      resize(get_shape(X));
      triqs_arrays_assign_delegation(*this, X);
      return *this;
    }

    //------------------ resize  -------------------------
    /** 
     * Resizes the array. NB : all references to the storage is invalidated.
     * Does not initialize the array
     * Content is undefined
     */
    template <typename... Args> void resize(Args const &... args) {
      static_assert(std::is_copy_constructible<ValueType>::value, "Can not resize an array if its value_type is not copy constructible");
      _idx_m = idx_map<Rank>(shape_t<Rank>(args...), _idx_m.memory_layout());
      // build a new one with the lengths of IND BUT THE SAME layout !
      // optimisation. Construct a storage only if the new index is not compatible (size mismatch).
      if (_storage.size != _idx_m.size()) _storage = mem::handle<ValueType, 'R'>{_idx_m.size()};
    }

    // ------------------------------- data access --------------------------------------------

    // The Index Map object
    idx_map<Rank> const &indexmap() const { return _idx_m; }

    // The storage handle
    mem::handle<ValueType, S_B> const &storage() const { return _storage; }
    mem::handle<ValueType, S_B> &storage() { return _storage; }

    // Memory layout
    layout_t<Rank> const &layout() const { return _idx_m.layout(); }

    /// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
    ValueType const *restrict data_start() const { return _storage.data + _idx_m.offset(); }

    /// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
    ValueType *restrict data_start() { return _storage.data + _idx_m.offset(); }

    /// FIXME : auto : type is not good ...
    //auto const &shape() const { return _idx_m.lengths(); }

    /// FIXME same as shape()[i] : redondant
    //size_t shape(size_t i) const { return _idx_m.lengths()[i]; }

    /// Number of elements in the array
    long size() const { return _idx_m.size(); }

    /// FIXME : REMOVE size ? TRIVIAL
    bool is_empty() const { return size() == 0; }

    // ------------------------------- Iterators --------------------------------------------

    using const_iterator = iterator_adapter<true, idx_map<Rank>::iterator, storage_t>;
    using iterator       = iterator_adapter<false, idx_map<Rank>::iterator, storage_t>;
    const_iterator begin() const { return const_iterator(indexmap(), storage(), false); }
    const_iterator end() const { return const_iterator(indexmap(), storage(), true); }
    const_iterator cbegin() const { return const_iterator(indexmap(), storage(), false); }
    const_iterator cend() const { return const_iterator(indexmap(), storage(), true); }
    iterator begin() { return iterator(indexmap(), storage(), false); }
    iterator end() { return iterator(indexmap(), storage(), true); }

    // ------------------------------- Operations --------------------------------------------

    TRIQS_DEFINE_COMPOUND_OPERATORS(array);
    // to forbid serialization of views...
    //template<class Archive> void serialize(Archive & ar, const unsigned int version) = delete;
  };

} // namespace nda

