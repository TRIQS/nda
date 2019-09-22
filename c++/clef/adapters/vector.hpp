#pragma once
#include "../clef.hpp"
#include <vector>

namespace clef {

  template <typename T, typename RHS>
  void triqs_clef_auto_assign__std_vector_impl(T &x, RHS &&rhs) {
    x = std::forward<RHS>(rhs);
  }

  template <typename Expr, int... Is, typename T>
  void triqs_clef_auto_assign__std_vector_impl(T &x, make_fun_impl<Expr, Is...> &&rhs) {
    triqs_clef_auto_assign_subscript(x, std::forward<make_fun_impl<Expr, Is...>>(rhs));
  }

  template <typename T, typename Fnt>
  void triqs_clef_auto_assign_subscript(std::vector<T> &v, Fnt f) {
    for (size_t i = 0; i < v.size(); ++i) triqs_clef_auto_assign__std_vector_impl(v[i], f(i));
  };

  template <typename T>
  std::ostream &triqs_clef_formal_print(std::ostream &out, std::vector<T> const &) {
    return out << "std::vector";
  }

} // namespace clef
