#pragma once
namespace nda {

  // --------------------------- zeros ------------------------

  /// Make a array of zeros with the given dimensions.
  /// Return a scalar for the case of rank zero.
  /// If we want more general array, use the static factory zeros [See also]
  template <typename T, CONCEPT(std::integral) Int, auto Rank>
  REQUIRES(std::is_arithmetic_v<T> or nda::is_complex_v<T>)
  auto zeros(std::array<Int, Rank> const &shape) {
    // For Rank == 0 we should return the underlying scalar_t
    if constexpr (Rank == 0)
      return T{0};
    else
      return array<T, Rank>::zeros(shape);
  }

  ///
  template <typename T, CONCEPT(std::integral)... Int>
  auto zeros(Int... i) {
    return zeros<T>(std::array<long, sizeof...(Int)>{i...});
  }

  // --------------------------- make_regular ------------------------
  // general make_regular
  // FIXME : auto return ?  regular_t<A> ?
  template <typename A>
  basic_array<get_value_t<std::decay_t<A>>, get_rank<A>, C_layout, get_algebra<std::decay_t<A>>, heap> //
  make_regular(A &&x) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return std::forward<A>(x);
  }
  //template <typename A> regular_t<A> make_regular(A &&x) REQUIRES(is_ndarray_v<A>) { return std::forward<A>(x); }

  // --------------------------- resize_or_check_if_view------------------------

  /** 
   * Resize if A is a container, or assert that the view has the right dimension if A is view
   *
   * @tparam A
   * @param a A container or a view
   */
  template <typename A>
  void resize_or_check_if_view(A &a, std::array<long, A::rank> const &sha) REQUIRES(is_regular_or_view_v<A>) {
    if (a.shape() == sha) return;
    if constexpr (is_regular_v<A>) {
      a.resize(sha);
    } else {
      NDA_RUNTIME_ERROR << "Size mismatch : view class shape = " << a.shape() << " expected " << sha;
    }
  }

  // --------------- make_const_view------------------------

  /// Explicitly make a const view REMOVING all compile time guarantee flags
  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  basic_array_view<T const, R, L, Algebra> make_const_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra = 'A', typename AccessorPolicy, typename OwningPolicy>
  basic_array_view<T const, R, L, Algebra, AccessorPolicy, OwningPolicy>
  make_const_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  // --------------- make_matrix_view------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  matrix_view<T, L> make_matrix_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  matrix_view<T, L> make_matrix_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  matrix_view<T const, L> make_matrix_const_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  matrix_view<T const, L> make_matrix_const_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  // --------------- operator == ---------------------

  /// True iif all elements are equal.
  template <typename A, typename B>
  bool operator==(A const &a, B const &b) REQUIRES(is_ndarray_v<A> and is_ndarray_v<B>) {
#if (__cplusplus > 201703L)
    static_assert(StdEqualityComparableWith<get_value_t<A>, get_value_t<B>>, "A == B is only defined when their element can be compared");
#endif
    if (a.shape() != b.shape()) return false;
    bool r = true;
    nda::for_each(a.shape(), [&](auto &&... x) { r &= (a(x...) == b(x...)); });
    return r;
  }

  // --------------- ASSIGN FOREACH ------------------------

  template <typename T, typename F>
  //[[deprecated]] // FIXME : SHALL WE KEEP THIS ?
  void assign_foreach(T &x, F &&f) {
    nda::for_each(x.shape(), std::forward<F>(f));
  }

  // ------------------------------- auto_assign --------------------------------------------

  template <typename A, typename F>
  void clef_auto_assign(A &&a, F &&f) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    nda::for_each(a.shape(), [&a, &f](auto &&... x) { a(x...) = f(x...); });
  }

} // namespace nda
