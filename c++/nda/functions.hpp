#pragma once
namespace nda {

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
  void clef_auto_assign(A &&a, F&& f) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    nda::for_each(a.shape(), [&a, &f](auto &&... x) { a(x...) = f(x...); });
  }

} // namespace nda
