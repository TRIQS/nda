
// Literal include

template <typename RHS> auto &operator+=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  compound_assign_impl<'A'>(*this, rhs);
  return *this;
}
template <typename RHS> auto &operator-=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  compound_assign_impl<'S'>(*this, rhs);
  return *this;
}
template <typename RHS> auto &operator*=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  compound_assign_impl<'M'>(*this, rhs);
  return *this;
}
template <typename RHS> auto &operator/=(RHS const &rhs) {
  static_assert(not is_const, "Can not assign to a const view");
  compound_assign_impl<'D'>(*this, rhs);
  return *this;
}
