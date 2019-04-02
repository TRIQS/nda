/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2013 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
namespace nda {

  template <typename T> constexpr bool is_scalar = utility::is_in_ZRC<T>::value;

  template <typename A, typename B> using value_type_of_result_t = std::decay_t<decltype(A::value_type{} * B::value_type{})>> ;

  template <size_t... Is, typename A> void __call_with_0(std::index_sequence<Is...>, A &&a) {
    return a((0, std::get<Is>)...); // repeat 0 sizeof...(Is) times
  }
  template <typename A> struct __get_call_const_return_type {
    using type = decltype(__call_with_0(std::make_index_sequence<A::rank()>, std::declval<A const>));
  } template <typename A> using get_call_const_return_t = __get_call_const_return_type<A>::type;

  // nda type with
  // using __concept__ = ;
  // concept_enum_t __concept__ = ImmutableArray

  /// array expression
  template <char OP, typename L, typename R> struct _expr : TRIQS_CONCEPT_TAG_NAME(ImmutableArray) {

    using L_t = std::decay_t<L>;
    using R_t = std::decay_t<R>;

    static constexpr bool l_is_scalar = is_scalar<L_t>;
    static constexpr bool r_is_scalar = is_scalar<R_t>;
    static constexpr char amv         = (l_is_scalar ? R_t::amv : L_t::amv);
    using value_type                  = std::conditional_t<l_is_scalar, R_t::value_type, value_type_of_result_t<L_t, R_t>>;

    static_assert(l_is_scalar || r_is_scalar || get_rank<L>() == get_rank<R>(), "rank mismatch in array operations");

    L l;
    R r;

    constexpr int rank() const {
      if constexpr (l_is_scalar)
        return R_t::rank();
      else
        return L_t::rank();
    }

    // REALLY NEEDED ?
    template <typename LL, typename RR> _expr(LL &&l_, RR &&r_) : l(std::forward<LL>(l_)), r(std::forward<RR>(r_)) {}

    template <typename... Args> auto operator()(Args const &... args) const REQUIRES(not(clef::is_lazy<A> and ...)) {

      if constexpr (OP == '+') {
        if constexpr (l_is_scalar) {
          if constexpr (amv != 'M')
            return l + r(args...);
          else // matrix case: 1 is Identity according to matrix algebra and args is of size 2
            return (operator==(args...) ? l : value_type) + r(args...);
        }
        if constexpr (r_is_scalar) {
          if constexpr (amv != 'M')
            return l(args...) + r;
          else
            return l(args...) + (operator==(args...) ? r : value_type);
        }
        return l(args...) + r(args...);
      }

      if constexpr (OP == '-') {
        if constexpr (l_is_scalar) {
          if constexpr (amv != 'M')
            return l - r(args...);
          else // matrix case: 1 is Identity according to matrix algebra and args is of size 2
            return (operator==(args...) ? l : value_type) - r(args...);
        }
        if constexpr (r_is_scalar) {
          if constexpr (amv != 'M')
            return l(args...) - r;
          else
            return l(args...) - (operator==(args...) ? r : value_type);
        }
        return l(args...) - r(args...);
      }

      if constexpr (OP == '*') {
        if constexpr (l_is_scalar) return l * r(args...);
        if constexpr (r_is_scalar) return l(args...) * r;
        return l(args...) * r(args...);
      }

      if constexpr (OP == '/') {
        if constexpr (l_is_scalar) return l / r(args...);
        if constexpr (r_is_scalar) return l(args...) / r;
        return l(args...) / r(args...);
      }
    }

    TRIQS_CLEF_IMPLEMENT_LAZY_CALL();

    /// [ ] is the same as (). Enable for Vectors only
    template <typename Arg>
    value_type operator[](Arg &&arg) const                                 //
       REQUIRES(l_is_scalar ? ImmutableVector<R_t> : ImmutableVector<L_t>) //
    {
      return operator()(std::forward<Args>(arg));
    }

    // just for better error messages
    template <typename T> void operator=(T &&x)  = delete; // Array expressions are immutable
    template <typename T> void operator+=(T &&x) = delete; // Array expressions are immutable
    template <typename T> void operator-=(T &&x) = delete; // Array expressions are immutable
  };

  // ---------------------------------

  // a special case : the unary operator !

  template <char OP, typename L, char AMV> struct _expr_unary : TRIQS_CONCEPT_TAG_NAME(ImmutableArray) {
    L l;
    static constexpr char amv = (l_is_scalar ? R_t::amv : L_t::amv);
    template <typename LL> _expr_unary(LL &&l_) : l(std::forward<LL>(l_)) {}

    template <typename... Args> value_type operator()(Args &&... args) const REQUIRES(not(clef::is_lazy<L>)) {
      return -l(std::forward<Args>(args)...);
    }

    TRIQS_CLEF_IMPLEMENT_LAZY_CALL();

    // just for better error messages
    template <typename T> void operator=(T &&x)  = delete; // Array expressions are immutable
    template <typename T> void operator+=(T &&x) = delete; // Array expressions are immutable
    template <typename T> void operator-=(T &&x) = delete; // Array expressions are immutable
  };

  //------------  Operator overload

  template <typename L, typename R>
  constexpr bool _ok_for_op = (ImmutableArray<L> and ImmutableArray<R>) or (is_in_ZRC<L> and ImmutableArray<R>)
     or (ImmutableArray<L> and is_in_ZRC<R>);

  template <typename L, typename R> _expr<'+', L, R> operator+(L &&l, R &&r) REQUIRES(_ok_for_op<L, R>) {
    return {std::forward<A1>(a1), std::forward<A2>(a2)};
  }

  template <typename L, typename R> _expr<'-', L, R> operator-(L &&l, R &&r) REQUIRES(_ok_for_op<L, R>) {
    return {std::forward<A1>(a1), std::forward<A2>(a2)};
  }

  template <typename L, typename R> _expr<'*', L, R> operator*(L &&l, R &&r)REQUIRES(_ok_for_op<L, R>) {
    return {std::forward<A1>(a1), std::forward<A2>(a2)};
  }

  template <typename L, typename R> _expr<'/', L, R> operator/(L &&l, R &&r) REQUIRES(_ok_for_op<L, R>) {
    return {std::forward<A1>(a1), std::forward<A2>(a2)};
  }

  template <typename L, typename R> _expr_unary<L> operator-(L &&l) REQUIRES(ImmutableArray<L>) { return {std::forward<L>(l)}; }

  //------------  lazy inverse

  template <class A> _expr<'/', A, int> inverse(A &&a) REQUIRES(ImmutableArray<A>) { return {1, std::forward<A>(a)}; }

  //------------  make_array

  /// Makes a new regular object (array, matrix, vector) from the expression
  template <char OP, typename L, typename R> array<value_type, _expr<OP, L, R>::rank()> make_array(_expr<OP, L, R> const &e) { return e; }

  /// Makes a new regular object (array, matrix, vector) from the expression
  template <typename L> array<value_type, _expr<OP, L, R>::rank()> make_array(_expr_unary<L> const &e) { return e; }

  /// Makes a new regular object (array, matrix, vector) from the expression
  template <char OP, typename L, typename R> auto make_regular(_expr<OP, L, R> const &x) {

    // auto with return if __concept__ or Use category to dispatch ?
    // for matrix
    return make_array(x);
  }

  //------------  Inverse of Matrix

  // anything / matrix ---> anything * inverse(matrix)
  template <typename A, typename M>
  auto operator/(A &&a, M &&m)    //
     REQUIRES(ImmutableMatrix<M>) //
  {
   return std::forward<A>(a) * inverse(std::forward<M>(m);
  }

  //------------  Print
  template <typename L> std::ostream &operator<<(std::ostream &sout, _expr_unary<L> const &expr) { return sout << '-' << expr.l; }

  template <char OP, typename L, typename R> std::ostream &operator<<(std::ostream &sout, _expr const &expr) {
    return sout << "(" << expr.l << " " << OP << " " << expr.r << ")";
  }

} // namespace nda
