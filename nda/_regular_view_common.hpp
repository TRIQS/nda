// ------------------------------- data access --------------------------------------------

/// The Index Map object
auto const &indexmap() const { return _idx_m; }

/// The storage handle
storage_t const &storage() const { return _storage; }
storage_t &storage() { return _storage; }

/// Memory layout
auto layout() const { return _idx_m.layout(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
ValueType const *data_start() const { return _storage.data() + _idx_m.offset(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
ValueType *data_start() { return _storage.data() + _idx_m.offset(); }

/// Shape of this
shape_t<Rank> const &shape() const { return _idx_m.lengths(); }

/// Number of elements
long size() const { return _idx_m.size(); }

/// size() == 0
[[deprecated]] bool is_empty() const { return size() == 0; }

/// Same as shape()[i]
[[deprecated]] long shape(int i) const { return _idx_m.lengths()[i]; }

// -------------------------------  operator () --------------------------------------------

// one can factorize the last part in a private static method, but I find clearer to have the repetition
// here. In particular to check the && case carefully.

// Internal only. A special case for optimization
decltype(auto) operator()(_linear_index_t x) const { return _storage[x.value]; }
decltype(auto) operator()(_linear_index_t x) { return _storage[x.value]; }

/**
 * Access the array, make a lazy expression or slice of it depending on the arguments
 *
 * @tparam T Can be long, range, range_all or ellipsis, of clef lazy (placeholder or expression)
 * @param x
 * @example array_call
 */
template <typename... T> decltype(auto) operator()(T const &... x) const & {
  if constexpr (sizeof...(T) == 0)
    return view_t{_idx_m, _storage};
  else {

    static_assert((Rank == -1) or (sizeof...(T) == Rank) or (ellipsis_is_present<T...> and (sizeof...(T) <= Rank)),
                  "Incorrect number of parameters in call");
    //if constexpr (clef::is_any_lazy_v<T...>) return clef::make_expr_call(*this, std::forward<T>(x)...);

    auto idx_or_pos = _idx_m.template slice_or_position<guarantees>(x...);              // we call the index map
    if constexpr (std::is_same_v<decltype(idx_or_pos), long>)                           // Case 1: we got a long, hence access a element
      return _storage[idx_or_pos];                                                      //
    else                                                                                // Case 2: we got a slice
      return my_view_template_t<decltype(idx_or_pos)>{std::move(idx_or_pos), _storage}; //
  }
}

///
template <typename... T> decltype(auto) operator()(T const &... x) & {
  if constexpr (sizeof...(T) == 0)
    return view_t{_idx_m, _storage};
  else {

    static_assert((Rank == -1) or (sizeof...(T) == Rank) or (ellipsis_is_present<T...> and (sizeof...(T) <= Rank)),
                  "Incorrect number of parameters in call");
    //if constexpr (clef::is_any_lazy_v<T...>) return clef::make_expr_call(*this, std::forward<T>(x)...);

    auto idx_or_pos = _idx_m.template slice_or_position<guarantees>(x...);              // we call the index map
    if constexpr (std::is_same_v<decltype(idx_or_pos), long>)                           // Case 1: we got a long, hence access a element
      return _storage[idx_or_pos];                                                      //
    else                                                                                // Case 2: we got a slice
      return my_view_template_t<decltype(idx_or_pos)>{std::move(idx_or_pos), _storage}; //
  }
}

///
template <typename... T> decltype(auto) operator()(T const &... x) && {
  if constexpr (sizeof...(T) == 0)
    return view_t{_idx_m, _storage};
  else {

    static_assert((Rank == -1) or (sizeof...(T) == Rank) or (ellipsis_is_present<T...> and (sizeof...(T) <= Rank)),
                  "Incorrect number of parameters in call");
    //if constexpr (clef::is_any_lazy_v<T...>) return clef::make_expr_call(std::move(*this), std::forward<T>(x)...);

    auto idx_or_pos = _idx_m.template slice_or_position<guarantees>(x...); // we call the index map

    if constexpr (std::is_same_v<decltype(idx_or_pos), long>) // Case 1: we got a long, hence access a element
      if constexpr (is_view)                                  //
        return _storage[idx_or_pos];                          // We return a REFERENCE here. Ok since underlying array is still alive
      else                                                    //
        return ValueType{_storage[idx_or_pos]};               // We return a VALUE here, the array is about be destroyed.
    else                                                      // Case 2: we got a slice
      return my_view_template_t<decltype(idx_or_pos)>{std::move(idx_or_pos), _storage}; //
  }
}

// ------------------------------- Iterators --------------------------------------------

///
using const_iterator = iterator_adapter<ValueType const, idx_map_t>;

///
using iterator       = iterator_adapter<ValueType, idx_map_t>;

///
const_iterator begin() const { return {indexmap().cbegin(), storage().data()}; }
///
const_iterator cbegin() const { return {indexmap().cbegin(), storage().data()}; }
///
iterator begin() { return {indexmap().cbegin(), storage().data()}; }

///
typename const_iterator::end_sentinel_t end() const { return {}; }
///
typename const_iterator::end_sentinel_t cend() const { return {}; }
///
typename iterator::end_sentinel_t end() { return {}; }

// ------------------------------- Operations --------------------------------------------

// FIXME : find a way to regroup on the same page in RST ?

/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS> auto &operator+=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  details::compound_assign_impl<'A'>(*this, rhs);
  return *this;
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS> auto &operator-=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  details::compound_assign_impl<'S'>(*this, rhs);
  return *this;
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS> auto &operator*=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  details::compound_assign_impl<'M'>(*this, rhs);
  return *this;
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS> auto &operator/=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  details::compound_assign_impl<'D'>(*this, rhs);
  return *this;
}
