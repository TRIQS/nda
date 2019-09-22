/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012-2013 by O. Parcollet
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
#include <tuple>
#include <type_traits>
#include <functional>
#include <memory>
#include <complex>
#include "macros.hpp"

namespace clef {

  //template <typename T>
  //struct remove_cv_ref : std::remove_cv<typename std::remove_reference<T>::type> {};

  using ull_t = unsigned long long;
  namespace tags {
    struct function {};
    struct subscript {};
    struct terminal {};
    struct if_else {};
    struct unary_op {};
    struct binary_op {};
  } // namespace tags

  // Compute the type to put in the expression tree.
  // If T is a lvalue reference, pack it into a reference_wrapper, unless force_copy_in_expr<T>::value == true
  // If T is an rvalue reference, we store it as the type (using move semantics).
  template <typename T>
  constexpr bool force_copy_in_expr = false;

  // --------------------

  template <class T>
  struct expr_storage_impl {
    using type = T;
  };
  template <class T>
  struct expr_storage_impl<T &> : std::conditional<force_copy_in_expr<std::remove_const_t<T>>, std::remove_const_t<T>, std::reference_wrapper<T>> {};
  template <class T>
  struct expr_storage_impl<T &&> {
    using type = T;
  };
  template <class T>
  struct expr_storage_impl<const T &&> {
    using type = T;
  };
  template <class T>
  struct expr_storage_impl<const T> {
    using type = T;
  };

  template <class T>
  using expr_storage_t = typename expr_storage_impl<T>::type; // helper type

  /* ---------------------------------------------------------------------------------------------------
  *  Placeholder and corresponding traits
  *  --------------------------------------------------------------------------------------------------- */
  template <int i, typename T>
  struct pair; // forward
  template <typename Tag, typename... T>
  struct expr; //forward

  // a placeholder is an empty struct, labelled by an int.
  template <int N>
  struct _ph {
    //static_assert( (N>=0) && (N<64) , "Placeholder number limited to [0,63]");
    static_assert((N >= 0), "Invalid placeholder range. Placeholder parameters is to [0,15] for each type of placeholder.");
    static constexpr int index = N;
    template <typename RHS>
    pair<N, RHS> operator=(RHS &&rhs) const {
      return {std::forward<RHS>(rhs)};
    }
    template <typename... T>
    expr<tags::function, _ph, expr_storage_t<T>...> operator()(T &&... x) const {
      return {tags::function{}, *this, std::forward<T>(x)...};
    }
    template <typename T>
    expr<tags::subscript, _ph, expr_storage_t<T>> operator[](T &&x) const {
      return {tags::subscript{}, *this, std::forward<T>(x)};
    }
  };

  //
  constexpr int _ph_flatten_indices(int i, int p) { return (i <= 15 ? p * 16 + i : -1); }

  // user class
  template <int I>
  using placeholder = _ph<_ph_flatten_indices(I, 0)>; // the ordinary placeholder (rank 0) with index [0,15]
  template <int I>
  using placeholder_prime = _ph<_ph_flatten_indices(I, 1)>; // the of placeholder rank 1 with index [0,15]

  // _ph will always be copied (they are empty anyway).
  template <int N>
  constexpr bool force_copy_in_expr<_ph<N>> = true;

  // represent a couple (_ph, value).
  template <int N, typename U>
  struct pair {
    U rhs;
    static constexpr int p = N;
    using value_type       = std::decay_t<U>;
  };

  // ph_set is a trait that given a pack of type, returns the set of _phs they contain
  // it returns a int in binary coding : bit N in the int is 1 iif at least one T is lazy and contains _ph<N>
  template <typename... T>
  struct ph_set;
  template <typename T0, typename... T>
  struct ph_set<T0, T...> {
    static constexpr ull_t value = ph_set<T0>::value | ph_set<T...>::value;
  };
  template <typename T>
  struct ph_set<T> {
    static constexpr ull_t value = 0;
  };
  template <int N>
  struct ph_set<_ph<N>> {
    static constexpr ull_t value = 1ull << N;
  };
  template <int i, typename T>
  struct ph_set<pair<i, T>> : ph_set<_ph<i>> {};

  /* ---------------------------------------------------------------------------------------------------
  * is_lazy and is_any_lazy
  *  --------------------------------------------------------------------------------------------------- */
  template <typename T>
  constexpr bool is_lazy = false;

  template <typename... Args>
  constexpr bool is_any_lazy = (is_lazy<std::decay_t<Args>> or ...);

  template <int N>
  constexpr bool is_lazy<_ph<N>> = true;

  /* ---------------------------------------------------------------------------------------------------
  * Node of the expression tree
  *  --------------------------------------------------------------------------------------------------- */
  template <typename Tag, typename... T>
  struct expr {
    // T can be U, U & (a reference or a value).
    using childs_t = std::tuple<T...>;
    childs_t childs;
    expr(expr const &x) = default;
    expr(expr &&x) noexcept : childs(std::move(x.childs)) {}
    // a constructor with the Tag make it unambiguous with other constructors...
    template <typename... Args>
    expr(Tag, Args &&... args) : childs(std::forward<Args>(args)...) {}
    // [] returns a new lazy expression, with one more layer
    template <typename Args>
    expr<tags::subscript, expr, expr_storage_t<Args>> operator[](Args &&args) const {
      return {tags::subscript(), *this, std::forward<Args>(args)};
    }
    // () also ...
    template <typename... Args>
    expr<tags::function, expr, expr_storage_t<Args>...> operator()(Args &&... args) const {
      return {tags::function(), *this, std::forward<Args>(args)...};
    }
    // assignement is in general deleted
    expr &operator=(expr const &) = delete; // no ordinary assignment
    expr &operator=(expr &&) = default;     // move assign ok

    // however, this is ok in the case f(i,j) = expr, where f is a clef::function
    //template <typename RHS, typename CH = childs_t>
    //void operator=(RHS const &rhs) {
    //static_assert(std::is_base_of<tags::function_class, std::tuple_element_t<0, CH>>::value, "NO");
    //*this << rhs;
    //}
  };
  // set some traits
  template <typename Tag, typename... T>
  struct ph_set<expr<Tag, T...>> : ph_set<T...> {};

  template <typename Tag, typename... T>
  constexpr bool is_lazy<expr<Tag, T...>> = true;

  // if we want that subexpression are copied ?
  template <typename Tag, typename... T>
  constexpr bool force_copy_in_expr<expr<Tag, T...>> = true;

  template <typename Tag, typename... T>
  using expr_node_t = expr<Tag, expr_storage_t<T>...>;

  /* ---------------------------------------------------------------------------------------------------
  * The basic operations put in a template....
  *  --------------------------------------------------------------------------------------------------- */
  template <typename Tag>
  struct operation;

  // a little function to clean the reference_wrapper
  template <typename U>
  [[gnu::always_inline]] U &&_cl(U &&x) {
    return std::forward<U>(x);
  }
  template <typename U>
  [[gnu::always_inline]] decltype(auto) _cl(std::reference_wrapper<U> x) {
    return x.get();
  }

  // Terminal
  template <>
  struct operation<tags::terminal> {
    template <typename L>
    [[gnu::always_inline]] L operator()(L &&l) const {
      return std::forward<L>(l);
    }
  };

  // Function call
  template <>
  struct operation<tags::function> {
    template <typename F, typename... Args>
    [[gnu::always_inline]] decltype(auto) operator()(F &&f, Args &&... args) const {
      return _cl(std::forward<F>(f))(_cl(std::forward<Args>(args))...);
    }
  };

  // [ ] Call
  template <>
  struct operation<tags::subscript> {
    template <typename F, typename Args>
    [[gnu::always_inline]] decltype(auto) operator()(F &&f, Args &&args) const {
      return _cl(std::forward<F>(f))[_cl(std::forward<Args>(args))];
    }
  };

  // all binary operators....
#define CLEF_OPERATION(TAG, OP)                                                                                                                      \
  namespace tags {                                                                                                                                   \
    struct TAG : binary_op {                                                                                                                         \
      static const char *name() { return AS_STRING(OP); }                                                                                            \
    };                                                                                                                                               \
  }                                                                                                                                                  \
  template <typename L, typename R>                                                                                                                  \
  [[gnu::always_inline]] auto operator OP(L &&l, R &&r) REQUIRES(is_any_lazy<L, R>) {                                                                \
    return expr<tags::TAG, expr_storage_t<L>, expr_storage_t<R>>{tags::TAG(), std::forward<L>(l), std::forward<R>(r)};                               \
  }                                                                                                                                                  \
  template <>                                                                                                                                        \
  struct operation<tags::TAG> {                                                                                                                      \
    template <typename L, typename R>                                                                                                                \
    [[gnu::always_inline]] decltype(auto) operator()(L &&l, R &&r) const {                                                                           \
      return _cl(std::forward<L>(l)) OP _cl(std::forward<R>(r));                                                                                     \
    }                                                                                                                                                \
  };

  CLEF_OPERATION(plus, +);
  CLEF_OPERATION(minus, -);
  CLEF_OPERATION(multiplies, *);
  CLEF_OPERATION(divides, /);
  CLEF_OPERATION(greater, >);
  CLEF_OPERATION(less, <);
  CLEF_OPERATION(leq, <=);
  CLEF_OPERATION(geq, >=);
  CLEF_OPERATION(eq, ==);
#undef CLEF_OPERATION

  // all unary operators....
#define CLEF_OPERATION(TAG, OP)                                                                                                                      \
  namespace tags {                                                                                                                                   \
    struct TAG : unary_op {                                                                                                                          \
      static const char *name() { return AS_STRING(OP); }                                                                                            \
    };                                                                                                                                               \
  }                                                                                                                                                  \
  template <typename L>                                                                                                                              \
  [[gnu::always_inline]] auto operator OP(L &&l) REQUIRES(is_any_lazy<L>) {                                                                          \
    return expr<tags::TAG, expr_storage_t<L>>{tags::TAG(), std::forward<L>(l)};                                                                      \
  }                                                                                                                                                  \
  template <>                                                                                                                                        \
  struct operation<tags::TAG> {                                                                                                                      \
    template <typename L>                                                                                                                            \
    [[gnu::always_inline]] decltype(auto) operator()(L &&l) const {                                                                                  \
      return OP _cl(std::forward<L>(l));                                                                                                             \
    }                                                                                                                                                \
  };

  CLEF_OPERATION(unaryplus, +);
  CLEF_OPERATION(negate, -);
  CLEF_OPERATION(loginot, !);
#undef CLEF_OPERATION

  // the only ternary node :  expression if
  template <>
  struct operation<tags::if_else> {
    // A and B MUST be the same
    template <typename C, typename A, typename B>
    [[gnu::always_inline]] A operator()(C const &c, A const &a, B const &b) const {
      return _cl(c) ? _cl(a) : _cl(b);
    }
  };
  // operator is : if_else( Condition, A, B)
  template <typename C, typename A, typename B>
  [[gnu::always_inline]] expr<tags::if_else, expr_storage_t<C>, expr_storage_t<A>, expr_storage_t<B>> if_else(C &&c, A &&a, B &&b) {
    return {tags::if_else(), std::forward<C>(c), std::forward<A>(a), std::forward<B>(b)};
  }

  /* ---------------------------------------------------------------------------------------------------
  * Evaluation of the expression tree.
  *  --------------------------------------------------------------------------------------------------- */

  // Generic case : do nothing (for the leaf of the tree including _ph)
  template <typename T, typename... Pairs>
  struct evaluator {
    static constexpr bool is_lazy = is_any_lazy<T>;
    [[gnu::always_inline]] T const &operator()(T const &k, Pairs const &...) const { return k; }
  };

  // The general eval function for expressions : declaration only
  template <typename T, typename... Pairs>
  decltype(auto) eval(T const &ex, Pairs const &... pairs);

  // _ph
  template <int N, int i, typename T, typename... Pairs>
  struct evaluator<_ph<N>, pair<i, T>, Pairs...> {
    using eval_t                  = evaluator<_ph<N>, Pairs...>;
    static constexpr bool is_lazy = eval_t::is_lazy;
    [[gnu::always_inline]] decltype(auto) operator()(_ph<N>, pair<i, T> const &, Pairs const &... pairs) const {
      return eval_t()(_ph<N>(), pairs...);
    }
  };

  template <int N, typename T, typename... Pairs>
  struct evaluator<_ph<N>, pair<N, T>, Pairs...> {
    static constexpr bool is_lazy = false;
    [[gnu::always_inline]] T operator()(_ph<N>, pair<N, T> const &p, Pairs const &...) const { return p.rhs; }
  };

  // any object hold by reference wrapper is redirected to the evaluator of the object
  template <typename T, typename... Contexts>
  struct evaluator<std::reference_wrapper<T>, Contexts...> {
    static constexpr bool is_lazy = false;
    [[gnu::always_inline]] decltype(auto) operator()(std::reference_wrapper<T> const &x, Contexts const &... contexts) const {
      return eval(x.get(), contexts...);
    }
  };

  // Dispatch the operations : depends it the result is a lazy expression
  template <typename Tag, typename... Args>
  [[gnu::always_inline]] expr<Tag, expr_storage_t<Args>...> op_dispatch(std::true_type, Args &&... args) {
    return {Tag(), std::forward<Args>(args)...};
  }

  template <typename Tag, typename... Args>
  [[gnu::always_inline]] decltype(auto) op_dispatch(std::false_type, Args &&... args) {
    return operation<Tag>()(std::forward<Args>(args)...);
  }

  // the evaluator for an expression
  template <typename Tag, typename... Childs, typename... Pairs>
  struct evaluator<expr<Tag, Childs...>, Pairs...> {
    static constexpr bool is_lazy = (evaluator<Childs, Pairs...>::is_lazy or ...);

    template <size_t... Is>
    [[gnu::always_inline]] decltype(auto) eval_impl(std::index_sequence<Is...>, expr<Tag, Childs...> const &ex, Pairs const &... pairs) const {
      //  if constexpr(is_lazy)
      // return {Tag(), eval(std::get<Is>(ex.childs), pairs...)...};

      return op_dispatch<Tag>(std::integral_constant<bool, is_lazy>{}, eval(std::get<Is>(ex.childs), pairs...)...);
    }

    [[gnu::always_inline]] decltype(auto) operator()(expr<Tag, Childs...> const &ex, Pairs const &... pairs) const {
      return eval_impl(std::make_index_sequence<sizeof...(Childs)>(), ex, pairs...);
    }
  };

  // The general eval function for expressions
  template <typename T, typename... Pairs>
  [[gnu::always_inline]] decltype(auto) eval(T const &ex, Pairs const &... pairs) {
    return evaluator<T, Pairs...>()(ex, pairs...);
  }

  /* ---------------------------------------------------------------------------------------------------
 * Apply a function object to all the leaves of the expression tree
 *  --------------------------------------------------------------------------------------------------- */

  template <typename F>
  struct apply_on_each_leaf_impl {
    F f;

    private:
    template <typename ChildTuple, size_t... Is>
    void _apply_this_on_each(std::index_sequence<Is...>, ChildTuple const &child_tuple) {
      ((*this)(std::get<Is>(child_tuple)), ...);
    }

    public:
    template <typename Tag, typename... T>
    [[gnu::always_inline]] void operator()(expr<Tag, T...> const &ex) {
      _apply_this_on_each(std::make_index_sequence<sizeof...(T)>{}, ex.childs);
    }
    template <typename T>
    [[gnu::always_inline]] void operator()(T const &x) REQUIRES(!is_any_lazy<T>) {
      f(x);
    }
    template <typename T>
    [[gnu::always_inline]] void operator()(std::reference_wrapper<T> const &x) REQUIRES(!is_any_lazy<T>) {
      f(x.get());
    }
  };

  template <typename F, typename Expr>
  [[gnu::always_inline]] void apply_on_each_leaf(F &&f, Expr const &ex) {
    auto impl = apply_on_each_leaf_impl<F>{std::forward<F>(f)};
    impl(ex);
  }

  /* ---------------------------------------------------------------------------------------------------
  * make_function : transform an expression to a function
  *  --------------------------------------------------------------------------------------------------- */

  template <typename Expr, int... Is>
  struct make_fun_impl {
    Expr ex; // keep a copy of the expression
    //make_fun_impl(Expr const & ex_) : ex(ex_) {}

    template <typename... Args>
    [[gnu::always_inline]] decltype(auto) operator()(Args &&... args) const {
      return evaluator<Expr, pair<Is, Args>...>()(ex, pair<Is, Args>{std::forward<Args>(args)}...);
    }
  };

  // values of the ph, excluding the Is ...
  template <ull_t x, int... Is>
  struct ph_filter;
  template <ull_t x, int I0, int... Is>
  struct ph_filter<x, I0, Is...> {
    static constexpr ull_t value = ph_filter<x, Is...>::value & (~(1ull << I0));
  };
  template <ull_t x>
  struct ph_filter<x> {
    static constexpr ull_t value = x;
  };

  template <typename Expr, int... Is>
  struct ph_set<make_fun_impl<Expr, Is...>> {
    static constexpr ull_t value = ph_filter<ph_set<Expr>::value, Is...>::value;
  };

  template <typename Expr, int... Is>
  constexpr bool is_lazy<make_fun_impl<Expr, Is...>> = (ph_set<make_fun_impl<Expr, Is...>>::value != 0);

  template <typename Expr, int... Is>
  constexpr bool force_copy_in_expr<make_fun_impl<Expr, Is...>> = true;

  template <typename Expr, typename... Phs>
  [[gnu::always_inline]] make_fun_impl<std::decay_t<Expr>, Phs::index...> make_function(Expr &&ex, Phs...) {
    return {std::forward<Expr>(ex)};
  }

  template <typename Expr, int... Is, typename... Pairs>
  struct evaluator<make_fun_impl<Expr, Is...>, Pairs...> {
    using e_t                     = evaluator<Expr, Pairs...>;
    static constexpr bool is_lazy = (ph_set<make_fun_impl<Expr, Is...>>::value != ph_set<Pairs...>::value);
    [[gnu::always_inline]] decltype(auto) operator()(make_fun_impl<Expr, Is...> const &f, Pairs const &... pairs) const {
      return make_function(e_t()(f.ex, pairs...), _ph<Is>()...);
    }
  };

  template <int... N>
  struct ph_list {};
  template <int... N>
  ph_list<N...> var(_ph<N>...) {
    return {};
  }

  template <typename Expr, int... N>
  auto operator>>(ph_list<N...> &&, Expr const &ex) -> decltype(make_function(ex, _ph<N>()...)) {
    return make_function(ex, _ph<N>()...);
  }
  // add trailing as a workaround around a clang bug here on xcode 5.1.1 (?)

  /* --------------------------------------------------------------------------------------------------
  *  make_function
  *  x_ >> expression  is the same as make_function(expression,x)
  * --------------------------------------------------------------------------------------------------- */

  template <int N, typename Expr>
  auto operator>>(_ph<N>, Expr &&ex) {
    return make_function(ex, _ph<N>{});
  }

  /* ---------------------------------------------------------------------------------------------------
  * Auto assign for ()
  *  --------------------------------------------------------------------------------------------------- */

  // by default it is deleted = not implemented : every class has to define it...
  //template<typename T, typename F> void clef_auto_assign (T,F) = delete;

  // remove the ref_wrapper, terminal ...
  template <typename T, typename F>
  [[gnu::always_inline]] void clef_auto_assign(std::reference_wrapper<T> R, F &&f) {
    clef_auto_assign(R.get(), std::forward<F>(f));
  }
  template <typename T, typename F>
  [[gnu::always_inline]] void clef_auto_assign(expr<tags::terminal, T> const &t, F &&f) {
    clef_auto_assign(std::get<0>(t.childs), std::forward<F>(f));
  }

  // auto assign of an expr ? (for chain calls) : just reuse the same operator
  template <typename Tag, typename... Childs, typename RHS>
  [[gnu::always_inline]] void clef_auto_assign(expr<Tag, Childs...> &&ex, RHS const &rhs) {
    ex << rhs;
  }

  template <typename Tag, typename... Childs, typename RHS>
  [[gnu::always_inline]] void clef_auto_assign(expr<Tag, Childs...> const &ex, RHS const &rhs) {
    ex << rhs;
  }

  template <typename... Is>
  constexpr bool _all_different(int i0, Is... is) {
    return (((is - i0) * ... * 1) != 0);
  }

  // The case A(x_,y_) = RHS : we form the function (make_function) and call auto_assign (by ADL)
  template <typename F, typename RHS, int... Is>
  [[gnu::always_inline]] void operator<<(expr<tags::function, F, _ph<Is>...> &&ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholders on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
  }
  template <typename F, typename RHS, int... Is>
  [[gnu::always_inline]] void operator<<(expr<tags::function, F, _ph<Is>...> const &ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholders on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
  }
  template <typename F, typename RHS, int... Is>
  [[gnu::always_inline]] void operator<<(expr<tags::function, F, _ph<Is>...> &ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholders on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
  }

  // The case A[x_,y_] = RHS : we form the function (make_function) and call auto_assign (by ADL)
  // template <typename F, typename RHS, int... Is> [[gnu::always_inline]] void operator<<(expr<tags::subscript, F, _tuple<_ph<Is>...>>&& ex, RHS&& rhs) {
  //  clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
  // }
  /*template <typename F, typename RHS, int... Is>
 [[gnu::always_inline]] void operator<<(expr<tags::subscript, F, _ph<Is>...> const& ex, RHS&& rhs) {
  clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
 }
 template <typename F, typename RHS, int... Is> [[gnu::always_inline]] void operator<<(expr<tags::subscript, F, _ph<Is>...>& ex, RHS&& rhs) {
  clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
 }
*/
  // any other case e.g. f(x_+y_) = RHS etc .... which makes no sense : compiler will stop
  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::function, F, T...> &&ex, RHS &&rhs) = delete;
  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::function, F, T...> &ex, RHS &&rhs) = delete;
  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::function, F, T...> const &ex, RHS &&rhs) = delete;

  /* ---------------------------------------------------------------------------------------------------
  * Auto assign for []
  *  --------------------------------------------------------------------------------------------------- */

  // by default it is deleted = not implemented : every class has to define it...
  template <typename T, typename F>
  [[gnu::always_inline]] void clef_auto_assign_subscript(T, F) = delete;

  // remove the ref_wrapper, terminal ...
  template <typename T, typename F>
  [[gnu::always_inline]] void clef_auto_assign_subscript(std::reference_wrapper<T> R, F &&f) {
    clef_auto_assign_subscript(R.get(), std::forward<F>(f));
  }
  template <typename T, typename F>
  [[gnu::always_inline]] void clef_auto_assign_subscript(expr<tags::terminal, T> const &t, F &&f) {
    clef_auto_assign_subscript(std::get<0>(t.childs), std::forward<F>(f));
  }

  // auto assign of an expr ? (for chain calls) : just reuse the same operator
  template <typename Tag, typename... Childs, typename RHS>
  [[gnu::always_inline]] void clef_auto_assign_subscript(expr<Tag, Childs...> &&ex, RHS const &rhs) {
    ex << rhs;
  }

  template <typename Tag, typename... Childs, typename RHS>
  [[gnu::always_inline]] void clef_auto_assign_subscript(expr<Tag, Childs...> const &ex, RHS const &rhs) {
    ex << rhs;
  }

  // Same thing for the  [ ]
  template <typename F, typename RHS, int... Is>
  [[gnu::always_inline]] void operator<<(expr<tags::subscript, F, _ph<Is>...> const &ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholdes on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign_subscript(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
  }
  template <typename F, typename RHS, int... Is>
  [[gnu::always_inline]] void operator<<(expr<tags::subscript, F, _ph<Is>...> &&ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholdes on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign_subscript(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
  }

  template <typename F, typename RHS, int... Is>
  [[gnu::always_inline]] void operator<<(expr<tags::subscript, F, _ph<Is>...> &ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholdes on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign_subscript(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), _ph<Is>()...));
  }

  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::subscript, F, T...> &&ex, RHS &&rhs) = delete;
  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::subscript, F, T...> &ex, RHS &&rhs) = delete;
  template <typename F, typename RHS, typename... T>
  void operator<<(expr<tags::subscript, F, T...> const &ex, RHS &&rhs) = delete;

  /* --------------------------------------------------------------------------------------------------
  * Create a terminal node of an object. the from clone version force copying the object
  * --------------------------------------------------------------------------------------------------- */

  // make a node with the ref, unless it is an rvalue (which is moved).
  template <typename T>
  expr<tags::terminal, expr_storage_t<T>> make_expr(T &&x) {
    return {tags::terminal(), std::forward<T>(x)};
  }

  // make a node from a copy of the object
  template <typename T>
  expr<tags::terminal, std::decay_t<T>> make_expr_from_clone(T &&x) {
    return {tags::terminal(), std::forward<T>(x)};
  }

  /* --------------------------------------------------------------------------------------------------
  * Create a call node of an object
  * The object can be kept as a : a ref, a copy, a view
  * --------------------------------------------------------------------------------------------------- */

  //template <typename T>
  //constexpr int arity = 1;

  template <typename Obj, typename... Args>
  expr<tags::function, expr_storage_t<Obj>, expr_storage_t<Args>...> make_expr_call(Obj &&obj, Args &&... args) REQUIRES(is_any_lazy<Args...>) {
    //static_assert(((arity<Obj> == -1) || (arity<Obj> == sizeof...(Args))), "Object called with a wrong number of arguments");
    return {tags::function{}, std::forward<Obj>(obj), std::forward<Args>(args)...};
  }

  /* --------------------------------------------------------------------------------------------------
  * Create a [] call (subscript) node of an object
  * The object can be kept as a : a ref, a copy, a view
  * --------------------------------------------------------------------------------------------------- */

  template <typename Obj, typename Args>
  expr<tags::subscript, expr_storage_t<Obj>, expr_storage_t<Args>> make_expr_subscript(Obj &&obj, Args &&args) REQUIRES(is_any_lazy<Args>) {
    return {tags::function{}, std::forward<Obj>(obj), std::forward<Args>(args)};
  }

  /* --------------------------------------------------------------------------------------------------
  *  The macro to make any function lazy
  *  CLEF_MAKE_FNT_LAZY (Arity,FunctionName ) : creates a new function in the triqs::lazy namespace
  *  taking expressions (at least one argument has to be an expression)
  *  The lookup happens by ADL, so IT MUST BE USED IN THE clef namespace
  * --------------------------------------------------------------------------------------------------- */
  //#define CLEF_MAKE_FNT_LAZY1(name)                                                                                                              \
  //struct name##_lazy_impl {                                                                                                                          \
    //template <typename... A>                                                                                                                         \
    //decltype(auto) operator()(A &&... __a) const {                                                                                                   \
      //return name(std::forward<A>(__a)...);                                                                                                          \
    //}                                                                                                                                                \
  //};                                                                                                                                                 \
  //template <typename... A>                                                                                                                           \
  //auto name(A &&... __a) DECL_AND_RETURN(make_expr_call(name##_lazy_impl(), std::forward<A>(__a)...))

#define CLEF_MAKE_FNT_LAZY(name)                                                                                                                     \
  template <typename... A>                                                                                                                           \
  auto name(A &&... __a) REQUIRES(is_any_lazy<A...>) {                                                                                               \
    return make_expr_call([](auto const &... __b) -> decltype(auto) { return name(__b...); }, std::forward<A>(__a)...);                              \
  }

#define CLEF_EXTEND_FNT_LAZY(FUN, TRAIT)                                                                                                             \
  template <typename A>                                                                                                                              \
  std::enable_if_t<TRAIT<A>::value, clef::expr_node_t<clef::tags::function, clef::FUN##_lazy_impl, A>> FUN(A &&__a) {                                \
    return {clef::tags::function{}, clef::FUN##_lazy_impl{}, std::forward<A>(__a)};                                                                  \
  }

#define CLEF_IMPLEMENT_LAZY_METHOD(TY, name)                                                                                                         \
  struct __clef_lazy_method_impl_##TY##_##name {                                                                                                     \
    template <typename X, typename... A>                                                                                                             \
    decltype(auto) operator()(X &&__x, A &&... __a) const {                                                                                          \
      return __x.name(std::forward<A>(__a)...);                                                                                                      \
    }                                                                                                                                                \
    friend std::ostream &operator<<(std::ostream &out, __clef_lazy_method_impl_##TY##_##name const &) {                                              \
      return out << "apply_method:" << AS_STRING(name);                                                                                              \
    }                                                                                                                                                \
  };                                                                                                                                                 \
  template <typename... A>                                                                                                                           \
  auto name(A &&... a) DECL_AND_RETURN(make_expr_call(__clef_lazy_method_impl_##TY##_##name{}, *this, std::forward<A>(a)...))

#define CLEF_IMPLEMENT_LAZY_CALL(...)                                                                                                                \
  template <typename... Args>                                                                                                                        \
        auto operator()(Args &&... args) const &DECL_AND_RETURN(make_expr_call(*this, std::forward<Args>(args)...))                                  \
                                                                                                                                                     \
           template <typename... Args>                                                                                                               \
           auto operator()(Args &&... args)                                                                                                          \
        & DECL_AND_RETURN(make_expr_call(*this, std::forward<Args>(args)...))                                                                        \
                                                                                                                                                     \
             template <typename... Args>                                                                                                             \
             auto operator()(Args &&... args)                                                                                                        \
     && DECL_AND_RETURN(make_expr_call(std::move(*this), std::forward<Args>(args)...))

} // namespace clef
