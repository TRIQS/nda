// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Miguel Morales, Olivier Parcollet, Nils Wentzell

/**
 * @brief Get the memory layout of the view/array.
 * @return nda::idx_map specifying the layout of the view/array.
 */
[[nodiscard]] constexpr auto const &indexmap() const noexcept { return lay; }

/**
 * @brief Get the data storage of the view/array.
 * @return A const reference to the memory handle of the view/array.
 */
[[nodiscard]] storage_t const &storage() const & noexcept { return sto; }

/**
 * @brief Get the data storage of the view/array.
 * @return A reference to the memory handle of the view/array.
 */
[[nodiscard]] storage_t &storage() & noexcept { return sto; }

/**
 * @brief Get the data storage of the view/array.
 * @return A copy of the memory handle of the view/array.
 */
[[nodiscard]] storage_t storage() && noexcept { return std::move(sto); }

/**
 * @brief Get the stride order of the memory layout of the view/array (see nda::idx_map for more details on how we
 * define stride orders).
 *
 * @return `std::array<int, Rank>` object specifying the stride order.
 */
[[nodiscard]] constexpr auto stride_order() const noexcept { return lay.stride_order; }

/**
 * @brief Get a pointer to the actual data (in general this is not the beginning of the memory block for a view).
 * @return Const pointer to the first element of the view/array.
 */
[[nodiscard]] ValueType const *data() const noexcept { return sto.data(); }

/**
 * @brief Get a pointer to the actual data (in general this is not the beginning of thr memory block for a view).
 * @return Pointer to the first element of the view/array.
 */
[[nodiscard]] ValueType *data() noexcept { return sto.data(); }

/**
 * @brief Get the shape of the view/array.
 * @return `std::array<long, Rank>` object specifying the shape of the view/array.
 */
[[nodiscard]] auto const &shape() const noexcept { return lay.lengths(); }

/**
 * @brief Get the strides of the view/array (see nda::idx_map for more details on how we define strides).
 * @return `std::array<long, Rank>` object specifying the strides of the view/array.
 */
[[nodiscard]] auto const &strides() const noexcept { return lay.strides(); }

/**
 * @brief Get the total size of the view/array.
 * @return Number of elements contained in the view/array.
 */
[[nodiscard]] long size() const noexcept { return lay.size(); }

/**
 * @brief Is the memory layout of the view/array contiguous?
 * @return True if the nda::idx_map is contiguous, false otherwise.
 */
[[nodiscard]] long is_contiguous() const noexcept { return lay.is_contiguous(); }

/**
 * @brief Is the view/array empty?
 * @return True if the view/array does not contain any elements.
 */
[[nodiscard]] bool empty() const { return sto.is_null(); }

/// @deprecated Use empty() instead.
[[nodiscard]] bool is_empty() const noexcept { return sto.is_null(); }

/**
 * @brief Get the extent of the i<sup>th</sup> dimension.
 * @return Number of elements along the i<sup>th</sup> dimension.
 */
[[nodiscard]] long extent(int i) const noexcept {
#ifdef NDA_ENFORCE_BOUNDCHECK
  if (i < 0 || i >= rank) {
    std::cerr << "Error in extent: Dimension " << i << " is incompatible with array of rank " << rank << std::endl;
    std::terminate();
  }
#endif
  return lay.lengths()[i];
}

/// @deprecated Use `extent(i)` or `shape()[i]` instead.
[[nodiscard]] long shape(int i) const noexcept { return extent(i); }

/**
 * @brief Get a range that generates all valid index tuples.
 * @return An `itertools::multiplied` range that can be used to iterate over all valid index tuples.
 */
[[nodiscard]] auto indices() const noexcept { return itertools::product_range(shape()); }

/**
 * @brief Is the stride order of the view/array in C-order?
 * @return True if the stride order of the nda::idx_map is C-order, false otherwise.
 */
static constexpr bool is_stride_order_C() noexcept { return layout_t::is_stride_order_C(); }

/**
 * @brief Is the stride order of the view/array in Fortran-order?
 * @return True if the stride order of the nda::idx_map is Fortran-order, false otherwise.
 */
static constexpr bool is_stride_order_Fortran() noexcept { return layout_t::is_stride_order_Fortran(); }

/**
 * @brief Access the element of the view/array at the given nda::_linear_index_t.
 *
 * @details The linear index specifies the position of the element in the view/array and not the position of the
 * element w.r.t. to the data pointer (i.e. any possible strides should not be taken into account).
 *
 * @param idx nda::_linear_index_t object.
 * @return Const reference to the element at the given linear index.
 */
decltype(auto) operator()(_linear_index_t idx) const noexcept {
  if constexpr (layout_t::layout_prop == layout_prop_e::strided_1d)
    return sto[idx.value * lay.min_stride()];
  else if constexpr (layout_t::layout_prop == layout_prop_e::contiguous)
    return sto[idx.value];
  else
    static_assert(always_false<layout_t>, "Internal error in array/view: Calling this type with a _linear_index_t is not allowed");
}

/// Non-const overload of @ref nda::basic_array_view::operator()(_linear_index_t) const.
decltype(auto) operator()(_linear_index_t idx) noexcept {
  if constexpr (layout_t::layout_prop == layout_prop_e::strided_1d)
    return sto[idx.value * lay.min_stride()];
  else if constexpr (layout_t::layout_prop == layout_prop_e::contiguous)
    return sto[idx.value];
  else
    static_assert(always_false<layout_t>, "Internal error in array/view: Calling this type with a _linear_index_t is not allowed");
}

private:
// Constexpr variable that is true if bounds checking is disabled.
#ifdef NDA_ENFORCE_BOUNDCHECK
static constexpr bool has_no_boundcheck = false;
#else
static constexpr bool has_no_boundcheck = true;
#endif

public:
/**
 * @brief Implementation of the function call operator.
 *
 * @details This function is an implementation detail an should be private. Since the Green's function library in
 * TRIQS uses this function, it is kept public (for now).
 *
 * @tparam ResultAlgebra Algebra of the resulting view/array.
 * @tparam SelfIsRvalue True if the view/array is an rvalue.
 * @tparam Self Type of the calling view/array.
 * @tparam T Types of the arguments.
 *
 * @param self Calling view.
 * @param idxs Multi-dimensional index consisting of `long`, `nda::range`, `nda::range::all_t`, nda::ellipsis or lazy
 * arguments.
 * @return Result of the function call depending on the given arguments and type of the view/array.
 */
template <char ResultAlgebra, bool SelfIsRvalue, typename Self, typename... Ts>
FORCEINLINE static decltype(auto) call(Self &&self, Ts const &...idxs) noexcept(has_no_boundcheck) {
  // resulting value type
  using r_v_t = std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, ValueType const, ValueType>;

  // behavior depends on the given arguments
  if constexpr (clef::is_any_lazy<Ts...>) {
    // if there are lazy arguments, e.g. as in A(i_) << i_, a lazy expression is returned
    return clef::make_expr_call(std::forward<Self>(self), idxs...);
  } else if constexpr (sizeof...(Ts) == 0) {
    // if no arguments are given, a full view is returned
    return basic_array_view<r_v_t, Rank, LayoutPolicy, Algebra, AccessorPolicy, OwningPolicy>{self.lay, self.sto};
  } else {
    // otherwise we check the arguments and either access a single element or make a slice
    static_assert(((layout_t::template argument_is_allowed_for_call_or_slice<Ts> + ...) > 0),
                  "Error in array/view: Slice arguments must be convertible to range, ellipsis, or long (or string if the layout permits it)");

    // number of arguments convertible to long
    static constexpr int n_args_long = (layout_t::template argument_is_allowed_for_call<Ts> + ...);

    if constexpr (n_args_long == rank) {
      // access a single element
      long offset = self.lay(idxs...);
      if constexpr (is_view or not SelfIsRvalue) {
        // if the calling object is a view or an lvalue, we return a reference
        return AccessorPolicy::template accessor<r_v_t>::access(self.sto.data(), offset);
      } else {
        // otherwise, we return a copy of the value
        return ValueType{self.sto[offset]};
      }
    } else {
      // access a slice of the view/array
      auto const [offset, idxm]      = self.lay.slice(idxs...);
      static constexpr auto res_rank = decltype(idxm)::rank();
      // resulting algebra
      static constexpr char newAlgebra = (ResultAlgebra == 'M' and (res_rank == 1) ? 'V' : ResultAlgebra);
      // resulting layout policy
      using r_layout_p = typename detail::layout_to_policy<std::decay_t<decltype(idxm)>>::type;
      return basic_array_view<r_v_t, res_rank, r_layout_p, newAlgebra, AccessorPolicy, OwningPolicy>{std::move(idxm), {self.sto, offset}};
    }
  }
}

public:
/**
 * @brief Function call operator to access the view/array.
 *
 * @details Depending on the type of the calling object and the given arguments, this function call does the following:
 * - If any of the arguments is lazy, an nda::clef::expr with the nda::clef::tags::function tag is returned.
 * - If no arguments are given, a full view of the calling object is returned:
 *   - If the calling object itself or its value type is const, a view with a const value type is returned.
 *   - Otherwise, a view with a non-const value type is returned.
 * - If the number of arguments is equal to the rank of the calling object and all arguments are convertible to `long`,
 * a single element is accessed:
 *   - If the calling object is a view or an lvalue, a (const) reference to the element is returned.
 *   - Otherwise, a copy of the element is returned.
 * - Otherwise a slice of the calling object is returned with the same value type and accessor and owning policies as
 * the calling object. The algebra of the slice is the same as well, except if a 1-dimensional slice of a matrix is
 * taken. In this case, the algebra is changed to 'V'.
 *
 * @tparam Ts Types of the function arguments.
 * @param idxs Multi-dimensional index consisting of `long`, `nda::range`, `nda::range::all_t`, nda::ellipsis or lazy
 * arguments.
 * @return Result of the function call depending on the given arguments and type of the view/array.
 */
template <typename... Ts>
FORCEINLINE decltype(auto) operator()(Ts const &...idxs) const & noexcept(has_no_boundcheck) {
  static_assert((rank == -1) or (sizeof...(Ts) == rank) or (sizeof...(Ts) == 0) or (ellipsis_is_present<Ts...> and (sizeof...(Ts) <= rank + 1)),
                "Error in array/view: Incorrect number of parameters in call operator");
  return call<Algebra, false>(*this, idxs...);
}

/// Non-const overload of `nda::basic_array_view::operator()(Ts const &...) const &`.
template <typename... Ts>
FORCEINLINE decltype(auto) operator()(Ts const &...idxs) & noexcept(has_no_boundcheck) {
  static_assert((rank == -1) or (sizeof...(Ts) == rank) or (sizeof...(Ts) == 0) or (ellipsis_is_present<Ts...> and (sizeof...(Ts) <= rank + 1)),
                "Error in array/view: Incorrect number of parameters in call operator");
  return call<Algebra, false>(*this, idxs...);
}

/// Rvalue overload of `nda::basic_array_view::operator()(Ts const &...) const &`.
template <typename... Ts>
FORCEINLINE decltype(auto) operator()(Ts const &...idxs) && noexcept(has_no_boundcheck) {
  static_assert((rank == -1) or (sizeof...(Ts) == rank) or (sizeof...(Ts) == 0) or (ellipsis_is_present<Ts...> and (sizeof...(Ts) <= rank + 1)),
                "Error in array/view: Incorrect number of parameters in call operator");
  return call<Algebra, true>(*this, idxs...);
}

/**
 * @brief Subscript operator to access the 1-dimensional view/array.
 *
 * @details Depending on the type of the calling object and the given argument, this subscript operation does the
 * following:
 * - If the argument is lazy, an nda::clef::expr with the nda::clef::tags::function tag is returned.
 * - If the argument is convertible to `long`, a single element is accessed:
 *   - If the calling object is a view or an lvalue, a (const) reference to the element is returned.
 *   - Otherwise, a copy of the element is returned.
 * - Otherwise a slice of the calling object is returned with the same value type, algebra and accessor and owning
 * policies as the calling object.
 *
 * @tparam T Type of the argument.
 * @param idx 1-dimensional index that is either a `long`, `nda::range`, `nda::range::all_t`, nda::ellipsis or a lazy
 * argument.
 * @return Result of the subscript operation depending on the given argument and type of the view/array.
 */
template <typename T>
decltype(auto) operator[](T const &idx) const & noexcept(has_no_boundcheck) {
  static_assert((rank == 1), "Error in array/view: Subscript operator is only available for rank 1 views/arrays in C++17/20");
  return call<Algebra, false>(*this, idx);
}

/// Non-const overload of `nda::basic_array_view::operator[](T const &) const &`.
template <typename T>
decltype(auto) operator[](T const &x) & noexcept(has_no_boundcheck) {
  static_assert((rank == 1), "Error in array/view: Subscript operator is only available for rank 1 views/arrays in C++17/20");
  return call<Algebra, false>(*this, x);
}

/// Rvalue overload of `nda::basic_array_view::operator[](T const &) const &`.
template <typename T>
decltype(auto) operator[](T const &x) && noexcept(has_no_boundcheck) {
  static_assert((rank == 1), "Error in array/view: Subscript operator is only available for rank 1 views/arrays in C++17/20");
  return call<Algebra, true>(*this, x);
}

/// Rank of the nda::array_iterator for the view/array.
static constexpr int iterator_rank = (has_strided_1d(layout_t::layout_prop) ? 1 : Rank);

/// Const iterator type of the view/array.
using const_iterator = array_iterator<iterator_rank, ValueType const, typename AccessorPolicy::template accessor<ValueType>::pointer>;

/// Iterator type of the view/array.
using iterator = array_iterator<iterator_rank, ValueType, typename AccessorPolicy::template accessor<ValueType>::pointer>;

private:
// Make an iterator for the view/array depending on its type.
template <typename Iterator>
[[nodiscard]] auto make_iterator(bool at_end) const noexcept {
  if constexpr (iterator_rank == Rank) {
    // multi-dimensional iterator
    if constexpr (layout_t::is_stride_order_C()) {
      // C-order case (array_iterator already traverses the data in C-order)
      return Iterator{indexmap().lengths(), indexmap().strides(), sto.data(), at_end};
    } else {
      // general case (we need to permute the shape and the strides according to the stride order of the layout)
      return Iterator{nda::permutations::apply(layout_t::stride_order, indexmap().lengths()),
                      nda::permutations::apply(layout_t::stride_order, indexmap().strides()), sto.data(), at_end};
    }
  } else {
    // 1-dimensional iterator
    return Iterator{std::array<long, 1>{size()}, std::array<long, 1>{indexmap().min_stride()}, sto.data(), at_end};
  }
}

public:
/// Get a const iterator to the beginning of the view/array.
[[nodiscard]] const_iterator begin() const noexcept { return make_iterator<const_iterator>(false); }

/// Get a const iterator to the beginning of the view/array.
[[nodiscard]] const_iterator cbegin() const noexcept { return make_iterator<const_iterator>(false); }

/// Get an iterator to the beginning of the view/array.
iterator begin() noexcept { return make_iterator<iterator>(false); }

/// Get a const iterator to the end of the view/array.
[[nodiscard]] const_iterator end() const noexcept { return make_iterator<const_iterator>(true); }

/// Get a const iterator to the end of the view/array.
[[nodiscard]] const_iterator cend() const noexcept { return make_iterator<const_iterator>(true); }

/// Get an iterator to the end of the view/array.
iterator end() noexcept { return make_iterator<iterator>(true); }

/**
 * @brief Addition assignment operator.
 *
 * @details It first performs the (lazy) addition with the right hand side operand and then assigns the result to the
 * left hand side view/array.
 *
 * See nda::operator+(L &&, R &&) and nda::operator+(A &&, S &&) for more details.
 *
 * @tparam RHS nda::Scalar or nda::Array type.
 * @param rhs Right hand side operand of the addition assignment operation.
 * @return Reference to this object.
 */
template <typename RHS>
auto &operator+=(RHS const &rhs) noexcept {
  static_assert(not is_const, "Error in array/view: Can not assign to a const view");
  return operator=(*this + rhs);
}

/**
 * @brief Subtraction assignment operator.
 *
 * @details It first performs the (lazy) subtraction with the right hand side operand and then assigns the result to
 * the left hand side view/array.
 *
 * See nda::operator-(L &&, R &&) and nda::operator-(A &&, S &&) for more details.
 *
 * @tparam RHS nda::Scalar or nda::Array type.
 * @param rhs Right hand side operand of the subtraction assignment operation.
 * @return Reference to this object.
 */
template <typename RHS>
auto &operator-=(RHS const &rhs) noexcept {
  static_assert(not is_const, "Error in array/view: Can not assign to a const view");
  return operator=(*this - rhs);
}

/**
 * @brief Multiplication assignment operator.
 *
 * @details It first performs the (lazy) multiplication with the right hand side operand and then assigns the result
 * to the left hand side view/array.
 *
 * See nda::operator*(L &&, R &&) and nda::operator*(A &&, S &&) for more details.
 *
 * @tparam RHS nda::Scalar or nda::Array type.
 * @param rhs Right hand side operand of the multiplication assignment operation.
 * @return Reference to this object.
 */
template <typename RHS>
auto &operator*=(RHS const &rhs) noexcept {
  static_assert(not is_const, "Error in array/view: Can not assign to a const view");
  return operator=((*this) * rhs);
}

/**
 * @brief Division assignment operator.
 *
 * @details It first performs the (lazy) division with the right hand side operand and then assigns the result to the
 * left hand side view/array.
 *
 * See nda::operator/(L &&, R &&) and nda::operator/(A &&, S &&) for more details.
 *
 * @tparam RHS nda::Scalar or nda::Array type.
 * @param rhs Right hand side operand of the division assignment operation.
 * @return Reference to this object.
 */
template <typename RHS>
auto &operator/=(RHS const &rhs) noexcept {
  static_assert(not is_const, "Error in array/view: Can not assign to a const view");
  return operator=(*this / rhs);
}

/**
 * @brief Assignment operator makes a deep copy of a general contiguous range and assigns it to the 1-dimensional
 * view/array.
 *
 * @tparam R Range type.
 * @param rhs Right hand side range object.
 * @return Reference to this object.
 */
template <std::ranges::contiguous_range R>
auto &operator=(R const &rhs) noexcept
  requires(Rank == 1 and not MemoryArray<R> and not is_scalar_for_v<R, self_t>)
{
  *this = array_const_view<std::ranges::range_value_t<R>, 1>{rhs};
  return *this;
}

private:
// Implementation of the assignment from an n-dimensional array type.
template <typename RHS>
void assign_from_ndarray(RHS const &rhs) { // FIXME noexcept {
#ifdef NDA_ENFORCE_BOUNDCHECK
  if (this->shape() != rhs.shape())
    NDA_RUNTIME_ERROR << "Error in assign_from_ndarray: Size mismatch:"
                      << "\n LHS.shape() = " << this->shape() << "\n RHS.shape() = " << rhs.shape();
#endif
  // compile-time check if assignment is possible
  static_assert(std::is_assignable_v<value_type &, get_value_t<RHS>>, "Error in assign_from_ndarray: Incompatible value types");

  // are both operands nda::MemoryArray types?
  static constexpr bool both_in_memory = MemoryArray<self_t> and MemoryArray<RHS>;

  // do both operands have the same stride order?
  static constexpr bool same_stride_order = get_layout_info<self_t>.stride_order == get_layout_info<RHS>.stride_order;

  // prefer optimized options if possible
  if constexpr (both_in_memory and same_stride_order) {
    if (rhs.empty()) return;
    // are both operands strided in 1d?
    static constexpr bool both_1d_strided = has_layout_strided_1d<self_t> and has_layout_strided_1d<RHS>;
    if constexpr (mem::on_host<self_t, RHS> and both_1d_strided) {
      // vectorizable copy on host
      for (long i = 0; i < size(); ++i) (*this)(_linear_index_t{i}) = rhs(_linear_index_t{i});
      return;
    } else if constexpr (!mem::on_host<self_t, RHS> and have_same_value_type_v<self_t, RHS>) {
      // check for block-layout and use mem::memcpy2D if possible
      auto bl_layout_dst = get_block_layout(*this);
      auto bl_layout_src = get_block_layout(rhs);
      if (bl_layout_dst && bl_layout_src) {
        auto [n_bl_dst, bl_size_dst, bl_str_dst] = *bl_layout_dst;
        auto [n_bl_src, bl_size_src, bl_str_src] = *bl_layout_src;
        // check that the total memory size is the same
        if (n_bl_dst * bl_size_dst != n_bl_src * bl_size_src) NDA_RUNTIME_ERROR << "Error in assign_from_ndarray: Incompatible block sizes";
        // if either destination or source consists of a single block, we can chunk it up to make the layouts compatible
        if (n_bl_dst == 1 && n_bl_src > 1) {
          n_bl_dst = n_bl_src;
          bl_size_dst /= n_bl_src;
          bl_str_dst = bl_size_dst;
        }
        if (n_bl_src == 1 && n_bl_dst > 1) {
          n_bl_src = n_bl_dst;
          bl_size_src /= n_bl_dst;
          bl_str_src = bl_size_src;
        }
        // copy only if block-layouts are compatible, otherwise continue to fallback
        if (n_bl_dst == n_bl_src && bl_size_dst == bl_size_src) {
          mem::memcpy2D<mem::get_addr_space<self_t>, mem::get_addr_space<RHS>>((void *)data(), bl_str_dst * sizeof(value_type), (void *)rhs.data(),
                                                                               bl_str_src * sizeof(value_type), bl_size_src * sizeof(value_type),
                                                                               n_bl_src);
          return;
        }
      }
    }
  }
  // otherwise fallback to elementwise assignment
  if constexpr (mem::on_device<self_t> || mem::on_device<RHS>) {
    NDA_RUNTIME_ERROR << "Error in assign_from_ndarray: Fallback to elementwise assignment not implemented for arrays/views on the GPU";
  }
  nda::for_each(shape(), [this, &rhs](auto const &...args) { (*this)(args...) = rhs(args...); });
}

// Implementation to fill a view/array with a constant scalar value.
template <typename Scalar>
void fill_with_scalar(Scalar const &scalar) noexcept {
  // we make a special implementation if the array is strided in 1d or contiguous
  if constexpr (has_layout_strided_1d<self_t>) {
    const long L             = size();
    auto *__restrict const p = data(); // no alias possible here!
    if constexpr (has_contiguous_layout<self_t>) {
      for (long i = 0; i < L; ++i) p[i] = scalar;
    } else {
      const long stri  = indexmap().min_stride();
      const long Lstri = L * stri;
      for (long i = 0; i < Lstri; i += stri) p[i] = scalar;
    }
  } else {
    // no compile-time memory layout guarantees
    for (auto &x : *this) x = scalar;
  }
}

// Implementation of the assignment from a scalar value.
template <typename Scalar>
void assign_from_scalar(Scalar const &scalar) noexcept {
  static_assert(!is_const, "Error in assign_from_ndarray: Cannot assign to a const view");
  if constexpr (Algebra != 'M') {
    // element-wise assignment for non-matrix algebras
    fill_with_scalar(scalar);
  } else {
    // a scalar has to be interpreted as a unit matrix for matrix algebras (the scalar in the shortest diagonal)
    // FIXME : A priori faster to put 0 everywhere and then change the diag to avoid the if.
    // FIXME : Benchmark and confirm.
    if constexpr (is_scalar_or_convertible_v<Scalar>)
      fill_with_scalar(0);
    else
      fill_with_scalar(Scalar{0 * scalar}); // FIXME : improve this
    const long imax = std::min(extent(0), extent(1));
    for (long i = 0; i < imax; ++i) operator()(i, i) = scalar;
  }
}
