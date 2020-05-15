#pragma once
namespace nda {

  // --------------------------- make_regular ------------------------
  // general make_regular
  // FIXME : auto return ?  regular_t<A> ?
  template <typename A>
  basic_array<get_value_t<std::decay_t<A>>, get_rank<A>, C_layout, get_algebra<std::decay_t<A>>, heap> //
  make_regular(A &&x) NDA_REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return std::forward<A>(x);
  }
  //template <typename A> regular_t<A> make_regular(A &&x) NDA_REQUIRES(is_ndarray_v<A>) { return std::forward<A>(x); }

  // --------------------------- resize_or_check_if_view------------------------

  /** 
   * Resize if A is a container, or assert that the view has the right dimension if A is view
   *
   * @tparam A
   * @param a A container or a view
   */
  template <typename A>
  void resize_or_check_if_view(A &a, std::array<long, A::rank> const &sha) NDA_REQUIRES(is_regular_or_view_v<A>) {
    if (a.shape() == sha) return;
    if constexpr (is_regular_v<A>) {
      a.resize(sha);
    } else {
      NDA_RUNTIME_ERROR << "Size mismatch : view class shape = " << a.shape() << " expected " << sha;
    }
  }

  /// --------------- make_const_view------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  basic_array_view<T const, R, L, Algebra> make_const_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return {a};
  }

  template <typename T, int R, typename L, char Algebra = 'A', typename AccessorPolicy, typename OwningPolicy>
  basic_array_view<T const, R, L, Algebra, AccessorPolicy, OwningPolicy>
  make_const_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> const &a) {
    return {a};
  }

  /// --------------- make_matrix_view------------------------

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

  /// --------------- operator == ---------------------

  template <typename A, typename B>
  bool operator==(A const &a, B const &b) NDA_REQUIRES(is_ndarray_v<A> and is_ndarray_v<B>) {
    static constexpr bool A_and_B_have_same_element_type = std::is_same_v<get_value_t<A>, get_value_t<B>>;
    static constexpr bool element_type_is_an_integer     = std::is_integral_v<get_value_t<A>>;
    static_assert((A_and_B_have_same_element_type and element_type_is_an_integer), "A == B is only defined when A, B are array of *integers*");
    if (a.shape() != b.shape()) return false;
    bool r = true;
    nda::for_each(a.shape(), [&](auto &&... x) { r &= (a(x...) == b(x...)); });
    return r;
  }

  /// --------------- ASSIGN FOREACH ------------------------

  template <typename T, typename F>
  //[[deprecated]] // FIXME : SHALL WE KEEP THIS ?
  void assign_foreach(T &x, F &&f) {
    nda::for_each(x.shape(), std::forward<F>(f));
  }

  // ------------------------------- auto_assign --------------------------------------------

  template <typename A, typename F>
  void clef_auto_assign(A &&a, F &&f) NDA_REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    nda::for_each(a.shape(), [&a, &f](auto &&... x) { a(x...) = f(x...); });
  }

} // namespace nda
