// Copyright (c) 2019-2021 Simons Foundation
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

// ------------------------------- data access --------------------------------------------

// FIXME get_layout

/// The Index Map object
[[nodiscard]] constexpr auto const &indexmap() const noexcept { return lay; }

/// \private
[[nodiscard]] storage_t const &storage() const &noexcept { return sto; }

/// \private
[[nodiscard]] storage_t &storage() &noexcept { return sto; }

/// \private
[[nodiscard]] storage_t storage() &&noexcept { return std::move(sto); }

/// Memory stride_order
[[nodiscard]] constexpr auto stride_order() const noexcept { return lay.stride_order; }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
[[nodiscard]] ValueType const *data() const noexcept { return sto.data(); }

/// Starting point of the data. NB : this is NOT the beginning of the memory block for a view in general
ValueType *data() noexcept { return sto.data(); }

/// Shape
[[nodiscard]] std::array<long, rank> const &shape() const noexcept { return lay.lengths(); }

/// Strides
[[nodiscard]] std::array<long, rank> const &strides() const noexcept { return lay.strides(); }

///
[[nodiscard]] long size() const noexcept { return lay.size(); }

///
[[nodiscard]] long is_contiguous() const noexcept { return lay.is_contiguous(); }

/// size() == 0
[[nodiscard]] bool empty() const { return sto.is_null(); }

//[[deprecated]]
[[nodiscard]] bool is_empty() const noexcept { return sto.is_null(); }

[[nodiscard]] long extent(int i) const noexcept {
#ifdef NDA_ENFORCE_BOUNDCHECK
  if (i < 0 || i >= rank) {
    std::cerr << "Dimension i in arr.extent(i) is incompatible with array rank: i=" << i << "  rank=" << rank << std::endl;
    std::terminate();
  }
#endif
  return lay.lengths()[i];
}

/// Same as shape()[i]
//[[deprecated]]
[[nodiscard]] long shape(int i) const noexcept { return extent(i); }

/// Return a range that generates all valid index tuples
[[nodiscard]] auto indices() const noexcept { return itertools::product_range(shape()); }

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
  if constexpr (layout_t::layout_prop == layout_prop_e::strided_1d)
    return sto[x.value * lay.min_stride()];
  else if constexpr (layout_t::layout_prop == layout_prop_e::contiguous)
    return sto[x.value]; // min_stride is 1
  else
    static_assert(always_false<layout_t>, "Internal error in calling this type with a _linear_index_t. One should never reach this !");
}

/// \private NO DOC
decltype(auto) operator()(_linear_index_t x) noexcept {
  //NDA_PRINT(layout_t::layout_prop);
  if constexpr (layout_t::layout_prop == layout_prop_e::strided_1d)
    return sto[x.value * lay.min_stride()];
  else if constexpr (layout_t::layout_prop == layout_prop_e::contiguous)
    return sto[x.value]; // min_stride is 1
  else
    static_assert(always_false<layout_t>, "Internal error in calling this type with a _linear_index_t. One should never reach this !");
  // other case : should not happen, let it be a compilation error.
}

private:
// impl of call. Only different case is if Self is &&

#ifdef NDA_ENFORCE_BOUNDCHECK
static constexpr bool has_no_boundcheck = false;
#else
static constexpr bool has_no_boundcheck = true;
#endif

public:
// I keep this public for the call of gf, which has to be reinterpreted as matrix
// I can construct a matrix at once. Of course, the optimizer may eliminate the copy of handle and idx_map
// but I prefer to keep it shorter here.
// FIXME : consider to make this private, and use a make_matrix (...) on the resut in gf operator()
// and restest carefully with benchmarsk
/// \private
template <char ResultAlgebra, bool SelfIsRvalue, typename Self, typename... T>
FORCEINLINE static decltype(auto) call(Self &&self, T const &... x) noexcept(has_no_boundcheck) {

  using r_v_t = std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, ValueType const, ValueType>;

  if constexpr (clef::is_any_lazy<T...>) return clef::make_expr_call(std::forward<Self>(self), x...);

  // () returns a full view
  else if constexpr (sizeof...(T) == 0) {
    return basic_array_view<r_v_t, Rank, LayoutPolicy, Algebra, AccessorPolicy, OwningPolicy>{self.lay, self.sto};
  }

  else {
    static_assert(((layout_t::template argument_is_allowed_for_call_or_slice<T> + ...) > 0),
                  "Slice arguments must be convertible to range, Ellipsis, or long (or string if the layout permits it)");

    static constexpr int n_args_long = (layout_t::template argument_is_allowed_for_call<T> + ...);

    // case 1 : all arguments are long, we simply compute the offset
    if constexpr (n_args_long == rank) {         // no range, simply compute the linear position. There may be an ellipsis, but it is of zero length !
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
      auto const [offset, idxm] = self.lay.slice(x...);

      using r_layout_t = std::decay_t<decltype(idxm)>;

      static constexpr char newAlgebra = (ResultAlgebra == 'M' and (r_layout_t::rank() == 1) ? 'V' : ResultAlgebra);

      using r_view_t =
         // FIXME  basic_array_view<r_v_t, r_layout_t::rank(),
         basic_array_view<ValueType, r_layout_t::rank(), typename details::layout_to_policy<r_layout_t>::type, newAlgebra, AccessorPolicy,
                          OwningPolicy>;

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
FORCEINLINE decltype(auto) operator()(T const &... x) const &noexcept(has_no_boundcheck) {
  static_assert((rank == -1) or (sizeof...(T) == rank) or (sizeof...(T) == 0) or (ellipsis_is_present<T...> and (sizeof...(T) <= rank + 1)),
                "Incorrect number of parameters in call");
  return call<Algebra, false>(*this, x...);
}

///
template <typename... T>
FORCEINLINE decltype(auto) operator()(T const &... x) &noexcept(has_no_boundcheck) {

  if constexpr (not((rank == -1) or (sizeof...(T) == rank) or (sizeof...(T) == 0)
                    or (ellipsis_is_present<T...> and (sizeof...(T) <= rank + 1)))) { // +1 since ellipsis can be of size 0
    //if ((sizeof...(T) != rank))
    static_assert(with_Args<T...> or with_Array<self_t>, "Incorrect number of parameters in calling Array with Args");
  }
  //static_assert(,
  //              "Incorrect number of parameters in call");
  return call<Algebra, false>(*this, x...);
}

///
template <typename... T>
FORCEINLINE decltype(auto) operator()(T const &... x) &&noexcept(has_no_boundcheck) {
  static_assert((rank == -1) or (sizeof...(T) == rank) or (sizeof...(T) == 0) or (ellipsis_is_present<T...> and (sizeof...(T) <= rank + 1)),
                "Incorrect number of parameters in call");
  return call<Algebra, true>(*this, x...);
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
  return call<Algebra, false>(*this, x);
}

///
template <typename T>
decltype(auto) operator[](T const &x) &noexcept(has_no_boundcheck) {
  static_assert((rank == 1), " [ ] operator is only available for rank 1 in C++17/20");
  return call<Algebra, false>(*this, x);
}

///
template <typename T>
decltype(auto) operator[](T const &x) &&noexcept(has_no_boundcheck) {
  static_assert((rank == 1), " [ ] operator is only available for rank 1 in C++17/20");
  return call<Algebra, true>(*this, x);
}

// ------------------------------- Iterators --------------------------------------------

static constexpr int iterator_rank = (has_strided_1d(layout_t::layout_prop) ? 1 : Rank);

///
using const_iterator = array_iterator<iterator_rank, ValueType const, typename AccessorPolicy::template accessor<ValueType>::pointer>;

///
using iterator = array_iterator<iterator_rank, ValueType, typename AccessorPolicy::template accessor<ValueType>::pointer>;

private:
template <typename Iterator>
[[nodiscard]] auto make_iterator(bool at_end) const noexcept {
  if constexpr (iterator_rank == Rank) {
    // FIXME : remove optimziation becasue of a TRIQS bug
    return Iterator{indexmap().lengths(), indexmap().strides(), sto.data(), at_end};
    if constexpr (layout_t::is_stride_order_C())
      return Iterator{indexmap().lengths(), indexmap().strides(), sto.data(), at_end};
    else
      // general case. In C order, no need to spend time applying the identity permutation
      // the new length used by the iterator is  length[ stride_order[0]], length[ stride_order[1]], ...
      // since stride_order[0] is the slowest, it will traverse the memory in sequential order
      return Iterator{nda::permutations::apply(layout_t::stride_order, indexmap().lengths()),
                      nda::permutations::apply(layout_t::stride_order, indexmap().strides()), sto.data(), at_end};
  } else // 1d iteration
    return Iterator{std::array<long, 1>{size()}, std::array<long, 1>{indexmap().min_stride()}, sto.data(), at_end};
}

public:
///
[[nodiscard]] const_iterator begin() const noexcept { return make_iterator<const_iterator>(false); }
///
[[nodiscard]] const_iterator cbegin() const noexcept { return make_iterator<const_iterator>(false); }
///
iterator begin() noexcept { return make_iterator<iterator>(false); }

///
[[nodiscard]] const_iterator end() const noexcept { return make_iterator<const_iterator>(true); }
///
[[nodiscard]] const_iterator cend() const noexcept { return make_iterator<const_iterator>(true); }
///
iterator end() noexcept { return make_iterator<iterator>(true); }

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

/// Assign from 1D Contiguous Range
template <std::ranges::contiguous_range R>
auto &operator=(R const &rhs) noexcept requires(Rank == 1 and not MemoryArray<R>) {
  *this = basic_array_view{rhs};
  return *this;
}

private:
template <typename RHS>
void assign_from_ndarray(RHS const &rhs) { // FIXME noexcept {

#ifdef NDA_ENFORCE_BOUNDCHECK
  if (this->shape() != rhs.shape())
    NDA_RUNTIME_ERROR << "Size mismatch:"
                      << "\n LHS.shape() = " << this->shape() << "\n RHS.shape() = " << rhs.shape();
#endif

  static_assert(std::is_assignable_v<value_type &, get_value_t<RHS>>, "Assignment impossible for the type of RHS into the type of LHS");

  static constexpr bool both_in_memory    = MemoryArray<self_t> and MemoryArray<RHS>;
  static constexpr bool same_stride_order = get_layout_info<self_t>.stride_order == get_layout_info<RHS>.stride_order;

  if constexpr (both_in_memory and same_stride_order) {
    if (rhs.empty()) return;

    static constexpr bool both_1d_strided = has_layout_strided_1d<self_t> and has_layout_strided_1d<RHS>;

    if constexpr (mem::on_host<self_t, RHS> and both_1d_strided) { // -> vectorizable host copy
      for (long i = 0; i < size(); ++i) (*this)(_linear_index_t{i}) = rhs(_linear_index_t{i});
      return;
    } else if constexpr (!mem::on_host<self_t, RHS> and have_same_value_type_v<self_t, RHS>) {
      // Check for block-layout and use mem::memcpy2D if possible
      auto bl_layout_dst = get_block_layout(*this);
      auto bl_layout_src = get_block_layout(rhs);
      if (bl_layout_dst && bl_layout_src) {
        auto [n_bl_dst, bl_size_dst, bl_str_dst] = *bl_layout_dst;
        auto [n_bl_src, bl_size_src, bl_str_src] = *bl_layout_src;

        // If either destination or source consist of a single block we can chunk it up to make the layouts compatible
        if (n_bl_dst == 1 && n_bl_src > 1) {
          n_bl_dst = n_bl_src;
          bl_size_dst /= n_bl_src;
          bl_str_dst /= n_bl_src;
        }
        if (n_bl_src == 1 && n_bl_dst > 1) {
          n_bl_src = n_bl_dst;
          bl_size_src /= n_bl_dst;
          bl_str_src /= n_bl_dst;
        }

        // Copy only if block-layouts are compatible, otherwise continue to fallback
        if (n_bl_dst == n_bl_src && bl_size_dst == bl_size_src) {
          mem::memcpy2D<mem::get_addr_space<self_t>, mem::get_addr_space<RHS>>(
			  (void *)data(), bl_str_dst * sizeof(value_type), (void *)rhs.data(),
                          bl_str_src * sizeof(value_type), bl_size_src * sizeof(value_type), 
			  n_bl_src); 
          return;
        }
      }
    }
  }
  if constexpr (mem::on_device<self_t> || mem::on_device<RHS>) NDA_RUNTIME_ERROR << "Fallback to elementwise assignment not implemented for arrays on the GPU";
  // Fallback to elementwise assignment
  auto l = [this, &rhs](auto const &...args) { (*this)(args...) = rhs(args...); };
  nda::for_each(shape(), l);
}

// -----------------------------------------------------

template <typename Scalar>
void fill_with_scalar(Scalar const &scalar) noexcept {
  // we make a special implementation if the array is 1d strided or contiguous
  if constexpr (has_layout_strided_1d<self_t>) { // possibly contiguous
    const long L             = size();
    auto *__restrict const p = data(); // no alias possible here !
    if constexpr (has_contiguous_layout<self_t>) {
      for (long i = 0; i < L; ++i) p[i] = scalar;
    } else {
      const long stri  = indexmap().min_stride();
      const long Lstri = L * stri;
      for (long i = 0; i < Lstri; i += stri) p[i] = scalar;
    }
  } else {
    for (auto &x : *this) x = scalar;
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
    const long imax = std::min(extent(0), extent(1));
    for (long i = 0; i < imax; ++i) operator()(i, i) = scalar;
  }
}
