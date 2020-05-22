// ------------------------------- data access --------------------------------------------

// FIXME get_layout

/// The Index Map object
[[nodiscard]] constexpr auto const &indexmap() const noexcept { return lay; }

/// \private
[[nodiscard]] storage_t const &storage() const noexcept { return sto; }

/// \private
storage_t &storage() { return sto; }

/// Memory stride_order
[[nodiscard]] constexpr auto stride_order() const noexcept { return lay.stride_order(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
[[nodiscard]] ValueType const *data_start() const noexcept { return sto.data(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
ValueType *data_start() noexcept { return sto.data(); }

/// Shape
[[nodiscard]] std::array<long, rank> const &shape() const noexcept { return lay.lengths(); }

///
[[nodiscard]] long size() const noexcept { return lay.size(); }

/// size() == 0
[[nodiscard]] bool empty() const { return sto.is_null(); }

//[[deprecated]]
[[nodiscard]] bool is_empty() const noexcept { return sto.is_null(); }

/// Same as shape()[i]
//[[deprecated]]
[[nodiscard]] long shape(int i) const noexcept { return lay.lengths()[i]; }

[[nodiscard]] long extent(int i) const noexcept { return lay.lengths()[i]; }

///
static constexpr bool is_stride_order_C() noexcept { return layout_t::is_stride_order_C(); }

///
static constexpr bool is_stride_order_Fortran() noexcept { return layout_t::is_stride_order_Fortran(); }

// -------------------------------  operator () --------------------------------------------

// impl details : optimization
// can NOT be put private, since used by expr template e.g. forwarding argument.
// but it is not for the user directly

/// \private NO DOC
decltype(auto) operator()(_linear_index_t x) const noexcept {
  //NDA_PRINT(layout_t::layout_prop);
  if constexpr (layout_t::layout_prop == layout_prop_e::strided_1d) return sto[x.value * lay.min_stride()];
  if constexpr (layout_t::layout_prop == layout_prop_e::contiguous) return sto[x.value]; // min_stride is 1
  // other case : should not happen, let it be a compilation error.
}

/// \private NO DOC
decltype(auto) operator()(_linear_index_t x) noexcept {
  //NDA_PRINT(layout_t::layout_prop);
  if constexpr (layout_t::layout_prop == layout_prop_e::strided_1d) return sto[x.value * lay.min_stride()];
  if constexpr (layout_t::layout_prop == layout_prop_e::contiguous) return sto[x.value]; // min_stride is 1
  // other case : should not happen, let it be a compilation error.
}

private:
// impl of call. Only different case is if Self is &&

#ifdef NDA_ENFORCE_BOUNDCHECK
  static constexpr bool has_no_boundcheck = false;
#else 
  static constexpr bool has_no_boundcheck = true;
#endif
 
template <bool SelfIsRvalue, typename Self, typename... T>
FORCEINLINE static decltype(auto) __call__impl(Self &&self, T const &... x) noexcept(has_no_boundcheck) {

  using r_v_t = std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, ValueType const, ValueType>;

  if constexpr (::clef::is_any_lazy<T...>) return ::clef::make_expr_call(std::forward<Self>(self), x...);

  // () returns a full view
  else if constexpr (sizeof...(T) == 0) {
    return basic_array_view<r_v_t, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>{self.lay, self.sto};
  }

  else {
    static_assert(((((std::is_base_of_v<range_tag, T> or std::is_constructible_v<long, T>) ? 0 : 1) + ...) == 0),
                  "Slice arguments must be convertible to range, Ellipsis, or long");

    static constexpr int n_args_long = (std::is_constructible_v<long, T> + ...);

    // case 1 : all arguments are long, we simply compute the offset
    if constexpr (n_args_long == rank) {         // no range, ellipsis, we simply compute the linear position
      long offset = self.lay(x...);              // compute the offset
      if constexpr (is_view or not SelfIsRvalue) //
        return AccessorPolicy::template accessor<ValueType>::access(self.sto.data(),
                                                                    offset); // We return a REFERENCE here. Ok since underlying array is still alive
      else                                                                   //
        return ValueType{self.sto[offset]};                                  // We return a VALUE here, the array is about be destroyed.
    }
    // case 2 : we have to make a slice
    else {
      // Static rank
      auto const [offset, idxm] = slice_static::slice_stride_order(self.lay, x...);

      using r_layout_t = decltype(idxm);
      using r_view_t =
         // FIXME  basic_array_view<r_v_t, r_layout_t::rank(),
         basic_array_view<ValueType, r_layout_t::rank(),
                          basic_layout<encode(r_layout_t::static_extents), encode(r_layout_t::stride_order), r_layout_t::layout_prop>, Algebra,
                          AccessorPolicy, OwningPolicy>;

      return r_view_t{std::move(idxm), {self.sto, offset}};
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
decltype(auto) operator()(T const &... x) const &noexcept(has_no_boundcheck) {
  static_assert((rank == -1) or (sizeof...(T) == rank) or (sizeof...(T) == 0) or (ellipsis_is_present<T...> and (sizeof...(T) <= rank)),
                "Incorrect number of parameters in call");
  return __call__impl<false>(*this, x...);
}

///
template <typename... T>
decltype(auto) operator()(T const &... x) &noexcept(has_no_boundcheck) {
  static_assert((rank == -1) or (sizeof...(T) == rank) or (sizeof...(T) == 0) or (ellipsis_is_present<T...> and (sizeof...(T) <= rank)),
                "Incorrect number of parameters in call");
  return __call__impl<false>(*this, x...);
}

///
template <typename... T>
decltype(auto) operator()(T const &... x) &&noexcept(has_no_boundcheck) {
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
decltype(auto) operator[](T const &x) const &noexcept(has_no_boundcheck) {
  static_assert((rank == 1), " [ ] operator is only available for rank 1 in C++17/20");
  return __call__impl<false>(*this, x);
}

///
template <typename T>
decltype(auto) operator[](T const &x) &noexcept(has_no_boundcheck) {
  static_assert((rank == 1), " [ ] operator is only available for rank 1 in C++17/20");
  return __call__impl<false>(*this, x);
}

///
template <typename T>
decltype(auto) operator[](T const &x) &&noexcept(has_no_boundcheck) {
  static_assert((rank == 1), " [ ] operator is only available for rank 1 in C++17/20");
  return __call__impl<true>(*this, x);
}

// ------------------------------- Iterators --------------------------------------------

static constexpr int iterator_rank = (layout_t::layout_prop & layout_prop_e::strided_1d ? 1 : Rank);

///
using const_iterator = array_iterator<iterator_rank, ValueType const, typename AccessorPolicy::template accessor<ValueType>::pointer>;

///
using iterator = array_iterator<iterator_rank, ValueType, typename AccessorPolicy::template accessor<ValueType>::pointer>;

private:
template <typename Iterator>
[[nodiscard]] auto _make_iterator(bool at_end) const noexcept {
  if constexpr (iterator_rank == Rank)
    return Iterator{indexmap().lengths(), indexmap().strides(), sto.data(), at_end};
  else
    return Iterator{std::array<long, 1>{size()}, std::array<long, 1>{indexmap().min_stride()}, sto.data(), at_end};
}

public:
///
[[nodiscard]] const_iterator begin() const noexcept { return _make_iterator<const_iterator>(false); }
///
[[nodiscard]] const_iterator cbegin() const noexcept { return _make_iterator<const_iterator>(false); }
///
iterator begin() noexcept { return _make_iterator<iterator>(false); }

///
[[nodiscard]] const_iterator end() const noexcept { return _make_iterator<const_iterator>(true); }
///
[[nodiscard]] const_iterator cend() const noexcept { return _make_iterator<const_iterator>(true); }
///
iterator end() noexcept { return _make_iterator<iterator>(true); }

// ------------------------------- Operations --------------------------------------------

/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator+=(RHS const &rhs) noexcept {
  static_assert(not is_const, "Can not assign to a const view");
  return operator=(*this + rhs);
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator-=(RHS const &rhs) noexcept {
  static_assert(not is_const, "Can not assign to a const view");
  return operator=(*this - rhs);
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator*=(RHS const &rhs) noexcept {
  static_assert(not is_const, "Can not assign to a const view");
  return operator=(*this *rhs);
}
/**
 * @tparam RHS A scalar or a type modeling NdArray
 * @param rhs
 */
template <typename RHS>
auto &operator/=(RHS const &rhs) noexcept {
  static_assert(not is_const, "Can not assign to a const view");
  return operator=(*this / rhs);
}

// ------------------------------- Assignment --------------------------------------------

private:
template <typename RHS>
void assign_from_ndarray(RHS const &rhs) noexcept {

#ifdef NDA_DEBUG
  if (this->shape() != rhs.shape())
    NDA_RUNTIME_ERROR << "Size mismatch in = "
                      << " : LHS " << *this << " \n RHS = " << rhs;
#endif

  // FIXME : what if RHS has no memory ??
  // firt if SPECIFIC to container ...
  // check call -> _linear_inde ?
  // general case if RHS is not a scalar (can be isp, expression...)
  static_assert(std::is_assignable_v<value_type &, get_value_t<RHS>>, "Assignment impossible for the type of RHS into the type of LHS");

  // If LHS and RHS are both 1d strided order or contiguous, and have the same stride order
  // we can make a 1d loop
  if constexpr ((get_layout_info<self_t>.stride_order == get_layout_info<RHS>.stride_order) // same stride order and both contiguous ...
                and has_layout_strided_1d<self_t> and has_layout_strided_1d<RHS>) {
    //static_assert(is_regular_or_view_v<RHS>, "oops");
    // In general, has_layout_strided_1d is FALSE by default
    // VALID ALSO FOR EXPRESSION !!!
    long L = size();
    for (long i = 0; i < L; ++i) (*this)(_linear_index_t{i}) = rhs(_linear_index_t{i});
  } else {
    auto l = [this, &rhs](auto const &... args) { (*this)(args...) = rhs(args...); };
    nda::for_each(shape(), l);
  }
}

// -----------------------------------------------------

template <typename Scalar>
void fill_with_scalar(Scalar const &scalar) noexcept {
  // we make a special implementation if the array is 1d strided or contiguous
  if constexpr (has_layout_strided_1d<self_t>) { // possibly contiguous
    const long L             = size();
    auto *__restrict const p = data_start(); // no alias possible here !
    if constexpr (has_layout_contiguous<self_t>) {
      for (long i = 0; i < L; ++i) p[i] = scalar;
    } else {
      const long stri = indexmap().min_stride();
      for (long i = 0; i < L; i += stri) p[i] = scalar;
    }
  } else {
    auto l = [this, scalar](auto const &... args) { (*this)(args...) = scalar; };
    nda::for_each_static<layout_t::static_extents_encoded, layout_t::stride_order_encoded>(shape(), l);
  }
}

// -----------------------------------------------------

template <typename Scalar>
void assign_from_scalar(Scalar const &scalar) noexcept {

  static_assert(!is_const, "Cannot assign to a const view !");

  if constexpr (Algebra != 'M') {
    fill_with_scalar(scalar);
  } else {
    //  A scalar has to be interpreted as a unit matrix
    // FIXME : A priori faster to put 0 everywhere and then change the diag to avoid the if.
    // FIXME : Benchmark and confirm
    if constexpr (is_scalar_or_convertible_v<Scalar>)
      fill_with_scalar(0);
    else
      fill_with_scalar(Scalar{0 * scalar}); //FIXME : improve this
    // on diagonal only
    const long imax = extent(0);
    for (long i = 0; i < imax; ++i) operator()(i, i) = scalar;
  }
}
