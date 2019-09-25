#pragma once
namespace nda {

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

  /// --------------- transposed_view------------------------

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  auto transposed_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a) {

    auto new_idx_map    = transpose<Permutation>(a.indexmap());
    using new_idx_map_t = decltype(new_idx_map);
    // FIXME STATICEXTENT
    using layout_policy = generic_layout<encode(new_idx_map_t::stride_order), new_idx_map_t::layout_prop>;

    return basic_array_view<T, R, layout_policy, Algebra, AccessorPolicy, OwningPolicy>{new_idx_map, a.storage()};
  }

  //--------------------

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  auto transposed_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return transposed_view(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed>(a));
  }

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  auto transposed_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a) {
    return transposed_view<Permutation>(basic_array_view<T, R, L, Algebra, default_accessor, borrowed>(a));
  }

  // for matrices ...
  template <typename A>
  auto transpose(A &&a) REQUIRES(is_regular_or_view_v<std::decay_t<A>> and (std::decay_t<A>::rank == 2)) {
    return transposed_view<encode(std::array{1, 0})>(std::forward<A>(a));
  }

  /// --------------- operator == ---------------------

  template <typename A, typename B>
  bool operator==(A const &a, B const &b) REQUIRES(is_ndarray_v<A> and is_ndarray_v<B>) {
    static_assert(std::is_same_v<get_value_t<A>, get_value_t<B>> and std::is_integral_v<get_value_t<A>>, "Only make sense for integers ");
    EXPECTS(a.shape() == b.shape());
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
  void clef_auto_assign(A &&a, F &&f) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    nda::for_each(a.shape(), [&a, &f](auto &&... x) { a(x...) = f(x...); });
  }

} // namespace nda
