#pragma once
#include "../clef.hpp"
#include <vector>

namespace nda::clef {

  template <typename T, typename RHS>
  void clef_auto_assign__std_vector_impl(T &x, RHS &&rhs) {
    x = std::forward<RHS>(rhs);
  }

  template <typename Expr, int... Is, typename T>
  void clef_auto_assign__std_vector_impl(T &x, make_fun_impl<Expr, Is...> &&rhs) {
    clef_auto_assign_subscript(x, std::forward<make_fun_impl<Expr, Is...>>(rhs));
  }

  template <typename T, typename Fnt>
  void clef_auto_assign_subscript(std::vector<T> &v, Fnt f) {
    for (size_t i = 0; i < v.size(); ++i) clef_auto_assign__std_vector_impl(v[i], f(i));
  }

} // namespace clef
