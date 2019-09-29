// ------------------------------- data access --------------------------------------------

/// The Index Map object
[[nodiscard]] constexpr auto const &indexmap() const noexcept { return _idx_m; }

/// The storage handle
[[nodiscard]] storage_t const &storage() const noexcept { return _storage; }
storage_t &storage() { return _storage; }

/// Memory stride_order
[[nodiscard]] constexpr auto stride_order() const noexcept { return _idx_m.stride_order(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
[[nodiscard]] ValueType const *data_start() const { return _storage.data(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
ValueType *data_start() { return _storage.data(); }

/// Shape of this
[[nodiscard]] shape_t<rank> const &shape() const { return _idx_m.lengths(); }

/// Number of elements
[[nodiscard]] long size() const { return _idx_m.size(); }

/// size() == 0
//[[deprecated]]
[[nodiscard]] bool is_empty() const { return _storage.is_null(); }

/// Same as shape()[i]
//[[deprecated]]
[[nodiscard]] long shape(int i) const { return _idx_m.lengths()[i]; }

[[nodiscard]] long extent(int i) const { return _idx_m.lengths()[i]; }

///
static constexpr bool is_stride_order_C() { return idx_map_t::is_stride_order_C(); }

///
static constexpr bool is_stride_order_Fortran() { return idx_map_t::is_stride_order_Fortran(); }

// -------------------------------  operator () --------------------------------------------

// one can factorize the last part in a private static method, but I find clearer to have the repetition
// here. In particular to check the && case carefully.

// Internal only. A special case for optimization
decltype(auto) operator()(_linear_index_t x) const {
  //NDA_PRINT(idx_map_t::layout_prop);
  if constexpr (idx_map_t::layout_prop == layout_prop_e::strided_1d) return _storage[x.value * _idx_m.min_stride()];
  if constexpr (idx_map_t::layout_prop == layout_prop_e::contiguous) return _storage[x.value]; // min_stride is 1
  // other case : should not happen, let it be a compilation error.
}
decltype(auto) operator()(_linear_index_t x) {
  //NDA_PRINT(idx_map_t::layout_prop);
  if constexpr (idx_map_t::layout_prop == layout_prop_e::strided_1d) return _storage[x.value * _idx_m.min_stride()];
  if constexpr (idx_map_t::layout_prop == layout_prop_e::contiguous) return _storage[x.value]; // min_stride is 1
  // other case : should not happen, let it be a compilation error.
}

private:
// impl of call. Only different case is if Self is &&

template <bool SelfIsRvalue, typename Self, typename... T>
FORCEINLINE static decltype(auto) __call__impl(Self &&self, T const &... x) {

  if constexpr (clef::is_any_lazy<T...>)
    return clef::make_expr_call(std::forward<Self>(self), x...);
  else if constexpr (sizeof...(T) == 0)
    return view_t{self._idx_m, self._storage};
  else {
    static_assert(((((std::is_base_of_v<range_tag, T> or std::is_constructible_v<long, T>) ? 0 : 1) + ...) == 0),
                  "Slice arguments must be convertible to range, Ellipsis, or long");

    static constexpr int n_args_long = (std::is_constructible_v<long, T> + ...);

    // case 1 : all arguments are long, we simply compute the offset
    if constexpr (n_args_long == rank) {         // no range, ellipsis, we simply compute the linear position
      long offset = self._idx_m(x...);           // compute the offset
      if constexpr (is_view or not SelfIsRvalue) //
        return AccessorPolicy::template accessor<ValueType>::access(self._storage.data(),
                                                                    offset); // We return a REFERENCE here. Ok since underlying array is still alive
      else                                                                   //
        return ValueType{self._storage[offset]};                             // We return a VALUE here, the array is about be destroyed.
    }
    // case 2 : we have to make a slice
    else {
      // Static rank
      auto const [offset, idxm] = slice_static::slice_stride_order(self._idx_m, x...);

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

/**
 * Access the array, make a lazy expression or slice of it depending on the arguments
 *
 * @tparam T Can be long, range, range_all or ellipsis, of clef lazy (placeholder or expression)
 * @param x
 * @example array_call
 */
template <typename T>
decltype(auto) operator[](T const &x) const & {
  static_assert((rank == 1), " [ ] operator is only available for rank 1 in C++17/20");
  return __call__impl<false>(*this, x);
}

///
template <typename T>
decltype(auto) operator[](T const &x) & {
  static_assert((rank == 1), " [ ] operator is only available for rank 1 in C++17/20");
  return __call__impl<false>(*this, x);
}

///
template <typename T>
decltype(auto) operator[](T const &x) && {
  static_assert((rank == 1), " [ ] operator is only available for rank 1 in C++17/20");
  return __call__impl<true>(*this, x);
}

// ------------------------------- Iterators --------------------------------------------

static constexpr int iterator_rank = (idx_map_t::layout_prop & layout_prop_e::strided_1d ? 1 : Rank);

///
using const_iterator = array_iterator<iterator_rank, ValueType const, typename AccessorPolicy::template accessor<ValueType>::pointer>;

///
using iterator = array_iterator<iterator_rank, ValueType, typename AccessorPolicy::template accessor<ValueType>::pointer>;

private:
template <typename Iterator>
[[nodiscard]] auto _make_iterator(bool at_end) const {
  //if constexpr (iterator_rank == Rank)
  //NDA_PRINT("USING general iterator");
  //else
  //NDA_PRINT("USING 1d iterator");

  if constexpr (iterator_rank == Rank)
    return Iterator{indexmap().lengths(), indexmap().strides(), storage().data(), at_end};
  else
    return Iterator{std::array<long, 1>{size()}, std::array<long, 1>{indexmap().min_stride()}, storage().data(), at_end};
}

public:
///
[[nodiscard]] const_iterator begin() const { return _make_iterator<const_iterator>(false); }
///
[[nodiscard]] const_iterator cbegin() const { return _make_iterator<const_iterator>(false); }
///
iterator begin() { return _make_iterator<iterator>(false); }

///
[[nodiscard]] const_iterator end() const { return _make_iterator<const_iterator>(true); }
///
[[nodiscard]] const_iterator cend() const { return _make_iterator<const_iterator>(true); }
///
iterator end() { return _make_iterator<iterator>(true); }

// ------------------------------- Operations --------------------------------------------

/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator+=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  return operator=(*this + rhs);
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator-=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  return operator=(*this - rhs);
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator*=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  return operator=(*this *rhs);
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator/=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  return operator=(*this / rhs);
}
