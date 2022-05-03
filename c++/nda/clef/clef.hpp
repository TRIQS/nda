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

#pragma once
#include <tuple>
#include <type_traits>
#include <functional>
#include <memory>
#include <complex>
#include "macros.hpp"

namespace nda::clef {

#define FORCEINLINE __inline__ __attribute__((always_inline))

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
  struct placeholder {
    static constexpr int index = N;
    template <typename RHS>
    pair<N, RHS> operator=(RHS &&rhs) const { // NOLINT It is correct here to return a pair, not the placeholder& as suggested by clang-tidy
      return {std::forward<RHS>(rhs)};
    }
    template <typename... T>
    expr<tags::function, placeholder, expr_storage_t<T>...> operator()(T &&...x) const {
      return {tags::function{}, *this, std::forward<T>(x)...};
    }
    template <typename T>
    expr<tags::subscript, placeholder, expr_storage_t<T>> operator[](T &&x) const {
      return {tags::subscript{}, *this, std::forward<T>(x)};
    }
  };

  // placeholder will always be copied (they are empty anyway).
  template <int N>
  constexpr bool force_copy_in_expr<placeholder<N>> = true;

  // represent a couple (placeholder, value).
  template <int N, typename U>
  struct pair {
    U rhs;
    static constexpr int p = N;
    using value_type       = std::decay_t<U>;
  };

  // ph_set is a trait that given a pack of type, returns the set of placeholders they contain
  // it returns a int in binary coding : bit N in the int is 1 iif at least one T is lazy and contains placeholder<N>
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
  struct ph_set<placeholder<N>> {
    static constexpr ull_t value = 1ull << N;
  };
  template <int i, typename T>
  struct ph_set<pair<i, T>> : ph_set<placeholder<i>> {};

  /* ---------------------------------------------------------------------------------------------------
  * is_lazy and is_any_lazy
  *  --------------------------------------------------------------------------------------------------- */
  template <typename T>
  constexpr bool is_lazy = false;
  template <typename T> requires(!std::is_same_v<T, std::remove_cvref_t<T>>)
  constexpr bool is_lazy<T> = is_lazy<std::remove_cvref_t<T>>;

  template <typename... Args>
  constexpr bool is_any_lazy = (is_lazy<Args> or ...);

  // FIXME : we should be this, much more precise
  template <typename... T>
  constexpr bool is_clef_expression = is_any_lazy<T...>;

  template <int N>
  constexpr bool is_lazy<placeholder<N>> = true;

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
    expr(Tag, Args &&...args) : childs(std::forward<Args>(args)...) {}
    // [] returns a new lazy expression, with one more layer
    template <typename Args>
    expr<tags::subscript, expr, expr_storage_t<Args>> operator[](Args &&args) const {
      return {tags::subscript(), *this, std::forward<Args>(args)};
    }
    // () also ...
    template <typename... Args>
    expr<tags::function, expr, expr_storage_t<Args>...> operator()(Args &&...args) const {
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
  FORCEINLINE U &&_cl(U &&x) {
    return std::forward<U>(x);
  }
  template <typename U>
  FORCEINLINE decltype(auto) _cl(std::reference_wrapper<U> x) {
    return x.get();
  }

  // Terminal
  template <>
  struct operation<tags::terminal> {
    template <typename L>
    FORCEINLINE L operator()(L &&l) const {
      return std::forward<L>(l);
    }
  };

  // Function call
  template <>
  struct operation<tags::function> {
    template <typename F, typename... Args>
    FORCEINLINE decltype(auto) operator()(F &&f, Args &&...args) const {
      return _cl(std::forward<F>(f))(_cl(std::forward<Args>(args))...);
    }
  };

  // [ ] Call
  template <>
  struct operation<tags::subscript> {
    template <typename F, typename Args>
    FORCEINLINE decltype(auto) operator()(F &&f, Args &&args) const {
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
  FORCEINLINE auto operator OP(L &&l, R &&r) CLEF_requires(is_any_lazy<L, R>) {                                                                      \
    return expr<tags::TAG, expr_storage_t<L>, expr_storage_t<R>>{tags::TAG(), std::forward<L>(l), std::forward<R>(r)};                               \
  }                                                                                                                                                  \
  template <>                                                                                                                                        \
  struct operation<tags::TAG> {                                                                                                                      \
    template <typename L, typename R>                                                                                                                \
    FORCEINLINE decltype(auto) operator()(L &&l, R &&r) const {                                                                                      \
      return _cl(std::forward<L>(l)) OP _cl(std::forward<R>(r));                                                                                     \
    }                                                                                                                                                \
  }

  // clang-format off
  CLEF_OPERATION(plus, +);
  CLEF_OPERATION(minus, -);
  CLEF_OPERATION(multiplies, *);
  CLEF_OPERATION(divides, /);
  CLEF_OPERATION(greater, >);
  CLEF_OPERATION(less, <);
  CLEF_OPERATION(leq, <=);
  CLEF_OPERATION(geq, >=);
  CLEF_OPERATION(eq, ==);
  // clang-format on
#undef CLEF_OPERATION

  // all unary operators....
#define CLEF_OPERATION(TAG, OP)                                                                                                                      \
  namespace tags {                                                                                                                                   \
    struct TAG : unary_op {                                                                                                                          \
      static const char *name() { return AS_STRING(OP); }                                                                                            \
    };                                                                                                                                               \
  }                                                                                                                                                  \
  template <typename L>                                                                                                                              \
  FORCEINLINE auto operator OP(L &&l) CLEF_requires(is_any_lazy<L>) {                                                                                \
    return expr<tags::TAG, expr_storage_t<L>>{tags::TAG(), std::forward<L>(l)};                                                                      \
  }                                                                                                                                                  \
  template <>                                                                                                                                        \
  struct operation<tags::TAG> {                                                                                                                      \
    template <typename L>                                                                                                                            \
    FORCEINLINE decltype(auto) operator()(L &&l) const {                                                                                             \
      return OP _cl(std::forward<L>(l));                                                                                                             \
    }                                                                                                                                                \
  }

  CLEF_OPERATION(unaryplus, +);
  CLEF_OPERATION(negate, -);
  CLEF_OPERATION(loginot, !);
#undef CLEF_OPERATION

  // the only ternary node :  expression if
  template <>
  struct operation<tags::if_else> {
    // A and B MUST be the same
    template <typename C, typename A, typename B>
    FORCEINLINE A operator()(C const &c, A const &a, B const &b) const {
      return _cl(c) ? _cl(a) : _cl(b);
    }
  };
  // operator is : if_else( Condition, A, B)
  template <typename C, typename A, typename B>
  FORCEINLINE expr<tags::if_else, expr_storage_t<C>, expr_storage_t<A>, expr_storage_t<B>> if_else(C &&c, A &&a, B &&b) {
    return {tags::if_else(), std::forward<C>(c), std::forward<A>(a), std::forward<B>(b)};
  }

  /* ---------------------------------------------------------------------------------------------------
  * Evaluation of the expression tree.
  *  --------------------------------------------------------------------------------------------------- */

  // Generic case : do nothing (for the leaf of the tree including placeholder)
  template <typename T, typename... Pairs>
  struct evaluator {
    static constexpr bool is_lazy = is_any_lazy<T>;
    FORCEINLINE T const &operator()(T const &k, Pairs const &...) const { return k; }
  };

  // The general eval function for expressions : declaration only
  template <typename T, typename... Pairs>
  decltype(auto) eval(T const &ex, Pairs const &...pairs);

#if 0
  // placeholder
  template <int N, int i, typename T, typename... Pairs>
  struct evaluator<placeholder<N>, pair<i, T>, Pairs...> {
    using eval_t                  = evaluator<placeholder<N>, Pairs...>;
    static constexpr bool is_lazy = eval_t::is_lazy;
    FORCEINLINE decltype(auto) operator()(placeholder<N>, pair<i, T> const &, Pairs const &... pairs) const {
      return eval_t()(placeholder<N>(), pairs...);
    }
  };

  template <int N, typename T, typename... Pairs>
  struct evaluator<placeholder<N>, pair<N, T>, Pairs...> {
    static constexpr bool is_lazy = false;
    FORCEINLINE T operator()(placeholder<N>, pair<N, T> const &p, Pairs const &...) const { return p.rhs; }
  };

#else

  template <int N, int... Is, typename... T>
  struct evaluator<placeholder<N>, pair<Is, T>...> {
    private:
    template <size_t... Ps>
    static constexpr int get_position_of_N(std::index_sequence<Ps...>) {
      return ((Is == N ? int(Ps) + 1 : 0) + ...) - 1;
    }
    static constexpr int N_position = get_position_of_N(std::make_index_sequence<sizeof...(Is)>{});

    public:
    static constexpr bool is_lazy = (N_position == -1);

    FORCEINLINE decltype(auto) operator()(placeholder<N>, pair<Is, T> const &...pairs) const {
      if constexpr (not is_lazy) { // N is one of the Is
        return std::get<N_position>(std::tie(pairs...)).rhs;
      } else { // N is not one of the Is
        return placeholder<N>{};
      }
    }
  };
#endif

  // any object hold by reference wrapper is redirected to the evaluator of the object
  template <typename T, typename... Contexts>
  struct evaluator<std::reference_wrapper<T>, Contexts...> {
    static constexpr bool is_lazy = false;
    FORCEINLINE decltype(auto) operator()(std::reference_wrapper<T> const &x, Contexts const &...contexts) const {
      return eval(x.get(), contexts...);
    }
  };

  // Dispatch the operations : depends it the result is a lazy expression
  template <typename Tag, typename... Args>
  FORCEINLINE expr<Tag, expr_storage_t<Args>...> op_dispatch(std::true_type, Args &&...args) {
    return {Tag(), std::forward<Args>(args)...};
  }

  template <typename Tag, typename... Args>
  FORCEINLINE decltype(auto) op_dispatch(std::false_type, Args &&...args) {
    return operation<Tag>()(std::forward<Args>(args)...);
  }

  // the evaluator for an expression
  template <typename Tag, typename... Childs, typename... Pairs>
  struct evaluator<expr<Tag, Childs...>, Pairs...> {
    static constexpr bool is_lazy = (evaluator<Childs, Pairs...>::is_lazy or ...);

    template <size_t... Is>
    [[nodiscard]] FORCEINLINE decltype(auto) eval_impl(std::index_sequence<Is...>, expr<Tag, Childs...> const &ex, Pairs const &...pairs) const {
      //  if constexpr(is_lazy)
      // return {Tag(), eval(std::get<Is>(ex.childs), pairs...)...};

      return op_dispatch<Tag>(std::integral_constant<bool, is_lazy>{}, eval(std::get<Is>(ex.childs), pairs...)...);
    }

    [[nodiscard]] FORCEINLINE decltype(auto) operator()(expr<Tag, Childs...> const &ex, Pairs const &...pairs) const {
      return eval_impl(std::make_index_sequence<sizeof...(Childs)>(), ex, pairs...);
    }
  };

  // The general eval function for expressions
  template <typename T, typename... Pairs>
  FORCEINLINE decltype(auto) eval(T const &ex, Pairs const &...pairs) {
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
    FORCEINLINE void operator()(expr<Tag, T...> const &ex) {
      _apply_this_on_each(std::make_index_sequence<sizeof...(T)>{}, ex.childs);
    }
    template <typename T>
    FORCEINLINE void operator()(T const &x) CLEF_requires(!is_any_lazy<T>) {
      f(x);
    }
    template <typename T>
    FORCEINLINE void operator()(std::reference_wrapper<T> const &x) CLEF_requires(!is_any_lazy<T>) {
      f(x.get());
    }
  };

  template <typename F, typename Expr>
  FORCEINLINE void apply_on_each_leaf(F &&f, Expr const &ex) {
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
    FORCEINLINE decltype(auto) operator()(Args &&...args) const {
      return evaluator<Expr, pair<Is, Args>...>()(ex, pair<Is, Args>{std::forward<Args>(args)}...);
    }
  };

  // To easily detect such an object in generic code.
  template <typename T>
  inline constexpr bool is_function = false;
  template <typename Expr, int... Is>
  inline constexpr bool is_function<make_fun_impl<Expr, Is...>> = true;

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
  FORCEINLINE make_fun_impl<std::decay_t<Expr>, Phs::index...> make_function(Expr &&ex, Phs...) {
    return {std::forward<Expr>(ex)};
  }

  template <typename Expr, int... Is, typename... Pairs>
  struct evaluator<make_fun_impl<Expr, Is...>, Pairs...> {
    using e_t                     = evaluator<Expr, Pairs...>;
    static constexpr bool is_lazy = (ph_set<make_fun_impl<Expr, Is...>>::value != ph_set<Pairs...>::value);
    FORCEINLINE decltype(auto) operator()(make_fun_impl<Expr, Is...> const &f, Pairs const &...pairs) const {
      return make_function(e_t()(f.ex, pairs...), placeholder<Is>()...);
    }
  };

  template <int... N>
  struct ph_list {};
  template <int... N>
  ph_list<N...> var(placeholder<N>...) {
    return {};
  }

  template <typename Expr, int... N>
  auto operator>>(ph_list<N...> &&, Expr const &ex) -> decltype(make_function(ex, placeholder<N>()...)) {
    return make_function(ex, placeholder<N>()...);
  }
  // add trailing as a workaround around a clang bug here on xcode 5.1.1 (?)

  /* --------------------------------------------------------------------------------------------------
  *  make_function
  *  x_ >> expression  is the same as make_function(expression,x)
  * --------------------------------------------------------------------------------------------------- */

  /*  template <int N, typename Expr>*/
  //auto operator>>(placeholder<N>, Expr &&ex) {
  //return make_function(ex, placeholder<N>{});
  //}

  /* ---------------------------------------------------------------------------------------------------
  * Auto assign for ()
  *  --------------------------------------------------------------------------------------------------- */

  // by default it is deleted = not implemented : every class has to define it...
  //template<typename T, typename F> void clef_auto_assign (T,F) = delete;

  // remove the ref_wrapper, terminal ...
  template <typename T, typename F>
  FORCEINLINE void clef_auto_assign(std::reference_wrapper<T> R, F &&f) {
    clef_auto_assign(R.get(), std::forward<F>(f));
  }
  template <typename T, typename F>
  FORCEINLINE void clef_auto_assign(expr<tags::terminal, T> const &t, F &&f) {
    clef_auto_assign(std::get<0>(t.childs), std::forward<F>(f));
  }

  // auto assign of an expr ? (for chain calls) : just reuse the same operator
  template <typename Tag, typename... Childs, typename RHS>
  FORCEINLINE void clef_auto_assign(expr<Tag, Childs...> &&ex, RHS const &rhs) {
    ex << rhs;
  }

  template <typename Tag, typename... Childs, typename RHS>
  FORCEINLINE void clef_auto_assign(expr<Tag, Childs...> const &ex, RHS const &rhs) {
    ex << rhs;
  }

  // a erroneous diagnostics in gcc : i0 is indeed used. We silence it.
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#endif
  template <typename... Is>
  constexpr bool _all_different(int i0, Is... is) {
    return (((is - i0) * ... * 1) != 0);
  }
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif

  // The case A(x_,y_) = RHS : we form the function (make_function) and call auto_assign (by ADL)
  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::function, F, placeholder<Is>...> &&ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholders on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  }
  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::function, F, placeholder<Is>...> const &ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholders on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  }
  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::function, F, placeholder<Is>...> &ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholders on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  }

  // The case A[x_,y_] = RHS : we form the function (make_function) and call auto_assign (by ADL)
  // template <typename F, typename RHS, int... Is> FORCEINLINE void operator<<(expr<tags::subscript, F, _tuple<placeholder<Is>...>>&& ex, RHS&& rhs) {
  //  clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  // }
  /*template <typename F, typename RHS, int... Is>
 FORCEINLINE void operator<<(expr<tags::subscript, F, placeholder<Is>...> const& ex, RHS&& rhs) {
  clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
 }
 template <typename F, typename RHS, int... Is> FORCEINLINE void operator<<(expr<tags::subscript, F, placeholder<Is>...>& ex, RHS&& rhs) {
  clef_auto_assign(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
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
  FORCEINLINE void clef_auto_assign_subscript(T, F) = delete;

  // remove the ref_wrapper, terminal ...
  template <typename T, typename F>
  FORCEINLINE void clef_auto_assign_subscript(std::reference_wrapper<T> R, F &&f) {
    clef_auto_assign_subscript(R.get(), std::forward<F>(f));
  }
  template <typename T, typename F>
  FORCEINLINE void clef_auto_assign_subscript(expr<tags::terminal, T> const &t, F &&f) {
    clef_auto_assign_subscript(std::get<0>(t.childs), std::forward<F>(f));
  }

  // auto assign of an expr ? (for chain calls) : just reuse the same operator
  template <typename Tag, typename... Childs, typename RHS>
  FORCEINLINE void clef_auto_assign_subscript(expr<Tag, Childs...> &&ex, RHS const &rhs) {
    ex << rhs;
  }

  template <typename Tag, typename... Childs, typename RHS>
  FORCEINLINE void clef_auto_assign_subscript(expr<Tag, Childs...> const &ex, RHS const &rhs) {
    ex << rhs;
  }

  // Same thing for the  [ ]
  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::subscript, F, placeholder<Is>...> const &ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholdes on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign_subscript(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  }
  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::subscript, F, placeholder<Is>...> &&ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholdes on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign_subscript(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
  }

  template <typename F, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<tags::subscript, F, placeholder<Is>...> &ex, RHS &&rhs) {
    static_assert(_all_different(Is...),
                  "Illegal expression : two of the placeholdes on the LHS are the same. This expression is only valid for loops on the full mesh");
    clef_auto_assign_subscript(std::get<0>(ex.childs), make_function(std::forward<RHS>(rhs), placeholder<Is>()...));
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
  expr<tags::function, expr_storage_t<Obj>, expr_storage_t<Args>...> make_expr_call(Obj &&obj, Args &&...args) CLEF_requires(is_any_lazy<Args...>) {
    //static_assert(((arity<Obj> == -1) || (arity<Obj> == sizeof...(Args))), "Object called with a wrong number of arguments");
    return {tags::function{}, std::forward<Obj>(obj), std::forward<Args>(args)...};
  }

  /* --------------------------------------------------------------------------------------------------
  * Create a [] call (subscript) node of an object
  * The object can be kept as a : a ref, a copy, a view
  * --------------------------------------------------------------------------------------------------- */

  template <typename Obj, typename Args>
  expr<tags::subscript, expr_storage_t<Obj>, expr_storage_t<Args>> make_expr_subscript(Obj &&obj, Args &&args) CLEF_requires(is_any_lazy<Args>) {
    return {tags::subscript{}, std::forward<Obj>(obj), std::forward<Args>(args)};
  }

  /* --------------------------------------------------------------------------------------------------
  *  The macro to make any function lazy
  *  CLEF_MAKE_FNT_LAZY (Arity,FunctionName ) : creates a new function in the triqs::lazy namespace
  *  taking expressions (at least one argument has to be an expression)
  *  The lookup happens by ADL, so IT MUST BE USED IN THE clef namespace
  * --------------------------------------------------------------------------------------------------- */
#define CLEF_MAKE_FNT_LAZY(name)                                                                                                                     \
  template <typename... A>                                                                                                                           \
  auto name(A &&...__a) CLEF_requires(nda::clef::is_any_lazy<A...>) {                                                                                \
    return make_expr_call([](auto &&...__b) -> decltype(auto) { return name(std::forward<decltype(__b)>(__b)...); }, std::forward<A>(__a)...);       \
  }

#define CLEF_IMPLEMENT_LAZY_METHOD(TY, name)                                                                                                         \
  template <typename... A>                                                                                                                           \
  auto name(A &&...__a) CLEF_requires(nda::clef::is_any_lazy<A...>) {                                                                                \
    return make_expr_call(                                                                                                                           \
       [](auto &&__obj, auto &&...__b) -> decltype(auto) { return std::forward<decltype(__obj)>(__obj).name(std::forward<decltype(__b)>(__b)...); }, \
       *this, std::forward<A>(__a)...);                                                                                                              \
  }

#define CLEF_IMPLEMENT_LAZY_CALL(...)                                                                                                                \
  template <typename... Args>                                                                                                                        \
  auto operator()(Args &&...args) const &CLEF_requires(nda::clef::is_any_lazy<Args...>) {                                                            \
    return make_expr_call(*this, std::forward<Args>(args)...);                                                                                       \
  }                                                                                                                                                  \
                                                                                                                                                     \
  template <typename... Args>                                                                                                                        \
     auto operator()(Args &&...args) & CLEF_requires(nda::clef::is_any_lazy<Args...>) {                                                              \
    return make_expr_call(*this, std::forward<Args>(args)...);                                                                                       \
  }                                                                                                                                                  \
                                                                                                                                                     \
  template <typename... Args>                                                                                                                        \
     auto operator()(Args &&...args) && CLEF_requires(nda::clef::is_any_lazy<Args...>) {                                                             \
    return make_expr_call(std::move(*this), std::forward<Args>(args)...);                                                                            \
  }

} // namespace nda::clef
