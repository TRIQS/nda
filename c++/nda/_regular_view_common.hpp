// ------------------------------- data access --------------------------------------------

/// The Index Map object
auto const &indexmap() const { return _idx_m; }

/// The storage handle
storage_t const &storage() const { return _storage; }
storage_t &storage() { return _storage; }

/// Memory layout
auto layout() const { return _idx_m.layout(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
ValueType const *data_start() const { return _storage.data(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
ValueType *data_start() { return _storage.data(); }

/// Shape of this
shape_t<rank> const &shape() const { return _idx_m.lengths(); }

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

private:
// impl of call. Only different case is if Self is &&

template <bool SelfIsRvalue, typename Self, typename... T>
FORCEINLINE static decltype(auto) __call__impl(Self &&self, T const &... x) {

  //if constexpr (clef::is_any_lazy_v<T...>) return clef::make_expr_call(*this, std::forward<T>(x)...);
  // else
  if constexpr (sizeof...(T) == 0)
    return view_t{self._idx_m, self._storage};
  else {
    static_assert(((((std::is_base_of_v<range_tag, T> or std::is_constructible_v<long, T>) ? 0 : 1) + ...) == 0),
                  "Slice arguments must be convertible to range, Ellipsis, or long");

    static constexpr int n_args_long = (std::is_constructible_v<long, T> + ...);

    // case 1 : all arguments are long, we simply compute the offset
    if constexpr (n_args_long == rank) {         // no range, ellipsis, we simply compute the linear position
      long offset = self._idx_m(x...);           // compute the offset
      if constexpr (is_view or not SelfIsRvalue) //
        return self._storage[offset];            // We return a REFERENCE here. Ok since underlying array is still alive
      else                                       //
        return ValueType{self._storage[offset]}; // We return a VALUE here, the array is about be destroyed.
    }
    // case 2 : we have to make a slice
    else {
      // Static rank
      auto const [offset, idxm] = slice_static::slice_layout(self._idx_m, x...);

      return my_view_template_t<decltype(idxm)>{std::move(idxm), {self._storage, offset}};
    }
  }
}

public:
/**
 * Access the array, make a lazy expression or slice of it depending on the arguments
 *
 * @tparam T Can be long, range, range_all or ellipsis, of clef lazy (placeholder or expression)
 * @param x
 * @example array_call
 */
template <typename... T>
decltype(auto) operator()(T const &... x) const & {
  static_assert((rank == -1) or (sizeof...(T) == rank) or (sizeof...(T) == 0) or (ellipsis_is_present<T...> and (sizeof...(T) <= rank)),
                "Incorrect number of parameters in call");
  return __call__impl<false>(*this, x...);
}

///
template <typename... T>
decltype(auto) operator()(T const &... x) & {
  static_assert((rank == -1) or (sizeof...(T) == rank) or (sizeof...(T) == 0) or (ellipsis_is_present<T...> and (sizeof...(T) <= rank)),
                "Incorrect number of parameters in call");
  return __call__impl<false>(*this, x...);
}

///
template <typename... T>
decltype(auto) operator()(T const &... x) && {
  static_assert((rank == -1) or (sizeof...(T) == rank) or (sizeof...(T) == 0) or (ellipsis_is_present<T...> and (sizeof...(T) <= rank)),
                "Incorrect number of parameters in call");
  return __call__impl<true>(*this, x...);
}

// ------------------------------- Iterators --------------------------------------------

///
using const_iterator = iterator_adapter<ValueType const, idx_map_t>;

///
using iterator = iterator_adapter<ValueType, idx_map_t>;

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
template <typename RHS>
auto &operator+=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  details::compound_assign_impl<'A'>(*this, rhs);
  return *this;
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator-=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  details::compound_assign_impl<'S'>(*this, rhs);
  return *this;
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator*=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  details::compound_assign_impl<'M'>(*this, rhs);
  return *this;
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator/=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  details::compound_assign_impl<'D'>(*this, rhs);
  return *this;
}
