// Copyright (c) 2019-2020 Simons Foundation
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

#pragma once
#include <array>
#include <tuple>
#include <type_traits>
#include <complex>
#include "macros.hpp"

namespace nda::clef {

#define FORCEINLINE __inline__ __attribute__((always_inline))

  // Are all the is different ? We use it at compile time.
  template <typename... Is>
  constexpr bool all_different(Is... is) {
    std::array<int, sizeof...(Is)> arr{is...};
    auto pos = std::adjacent_find(std::begin(arr), std::end(arr));
    return (pos == std::end(arr));
  }

  // Tags of the expression nodes
  namespace tags {
    struct function {};
    struct subscript {};
    struct terminal {};
    struct if_else {};
    struct plus {};
    struct minus {};
    struct multiplies {};
    struct divides {};
    struct greater {};
    struct less {};
    struct leq {};
    struct geq {};
    struct unaryplus {};
    struct negate {};
    struct loginot {};
  } // namespace tags

  /* ---------------------------------------------------------------------------------------------------
  * Compute the type to put in the expression tree 
  *  --------------------------------------------------------------------------------------------------- */

  // If T is a lvalue reference, pack it into a reference_wrapper, unless force_copy_in_expr<T>::value == true
  // If T is an rvalue reference, we store it as the type (using move semantics).
  template <typename T>
  constexpr bool force_copy_in_expr = false;

  template <class T>
  using expr_storage_t = std::conditional_t<std::is_reference_v<T> and !force_copy_in_expr<std::decay_t<T>>, //
                                            std::reference_wrapper<std::remove_reference_t<T>>,              //
                                            std::decay_t<T>>;

  /* ---------------------------------------------------------------------------------------------------
  * placeholder-value-pair : a couple (placeholder index at compile time, runtime value)
  *  --------------------------------------------------------------------------------------------------- */

  template <int N, typename U>
  struct phvp {
    U rhs;
    static constexpr int p = N;
    using value_type       = std::decay_t<U>;
  };

  /* ---------------------------------------------------------------------------------------------------
  * comma_tuple is a specific tuple to accumulate objects in subscript call
  *  g[ x,x,x,x]. It is a tuple, in a struct to allow proper overload and void conflict with the ordinary tuple
  *  NB : why in 2020 does C++ not allow g[x,y,z] ?????
  *  NB : not really related to clef, it could be put aside
  *  --------------------------------------------------------------------------------------------------- */

  // the comma_tuple structure
  template <typename... T>
  struct comma_tuple {
    std::tuple<T...> _t;
  };

  // make_comma_tuple
  template <typename... T>
  comma_tuple<std::decay_t<T>...> make_comma_tuple(T &&...x) {
    return {std::make_tuple(std::forward<T>(x)...)};
  }

  // the tuple absorbs anything
  template <typename X, typename... T>
  FORCEINLINE comma_tuple<T..., std::decay_t<X>> operator,(comma_tuple<T...> &&t, X &&x) {
    return {std::tuple_cat(std::move(t._t), std::make_tuple(std::forward<X>(x)))};
  }

  template <typename X, typename... T>
  FORCEINLINE comma_tuple<T..., std::decay_t<X>> operator,(X &&x, comma_tuple<T...> &&t) {
    return {std::tuple_cat(std::forward<X>(x)), std::make_tuple(std::move(t._t))};
  }

  // any type tagged with this will overload , operator and make a comma tuple
  template <typename T>
  constexpr bool is_comma_tuple_seed_v = false;

  // NB : we will need to copy this in other namespaces because of ADL, see e.g. triqs::mesh
  template <typename T, typename U>
  requires(is_comma_tuple_seed_v<std::decay_t<T>> or is_comma_tuple_seed_v<std::decay_t<U>>) //
     comma_tuple<std::decay_t<T>, std::decay_t<U>>
  operator,(T &&x, U &&u) { return {std::make_tuple(x, u)}; }

  /* ---------------------------------------------------------------------------------------------------
  * Lazy and is_lazy_v 
  *  --------------------------------------------------------------------------------------------------- */
  template <typename T>
  constexpr bool is_lazy_v = false;

  template <typename T>
  concept Lazy = is_lazy_v<std::decay_t<T>>;

  // true iif T contains OTHER ph than Is
  template <typename T, int... Is>
  static constexpr bool contains_other_ph = false;

  /* ---------------------------------------------------------------------------------------------------
  * General node of the expression tree
  *  --------------------------------------------------------------------------------------------------- */

  template <typename Tag, typename... T>
  struct expr {
    // T can be U, U & (a reference or a value).
    std::tuple<T...> children;

    // constructor with Tag to disambiguate with other constructors because of the && (e.g. expr(expr &) )
    template <typename... Args>
    expr(Tag, Args &&...args) : children(std::forward<Args>(args)...) {}

    // rule of 4 : delete the =
    expr(expr const &x) = default;
    expr(expr &&x)      = default;
    expr &operator=(expr const &) = delete; // no ordinary assignment
    expr &operator=(expr &&) = default;     // move assign ok

    // calling an expression just build a larger expression
    // FIXME : we could avoid a copy in && case and move ...
    template <typename... Args>
    auto operator()(Args &&...args) const {
      return expr<tags::function, expr, expr_storage_t<Args>...>{tags::function{}, *this, std::forward<Args>(args)...};
    }

    template <typename... Args>
    auto operator[](Args &&...args) const {
      return expr<tags::subscript, expr, expr_storage_t<Args>...>{tags::subscript{}, *this, std::forward<Args>(args)...};
    }
  };

  // --- is_lazy_v ----
  template <typename Tag, typename... T>
  constexpr bool is_lazy_v<expr<Tag, T...>> = true;

  // --- force_copy ----
  template <typename Tag, typename... T>
  constexpr bool force_copy_in_expr<expr<Tag, T...>> = true;

  // --- contains_other_ph ----
  template <typename Tag, typename... T, int... Is>
  static constexpr bool contains_other_ph<expr<Tag, T...>, Is...> = (contains_other_ph<T, Is...> or ...);

  // --- Makers ----

  // Make a general node
  template <typename Tag, typename... Args>
  auto make_expr(Args &&...args) {
    return expr<Tag, expr_storage_t<Args>...>{Tag{}, std::forward<Args>(args)...};
  }

  // Make a terminal node
  template <typename T>
  auto make_expr_terminal(T &&x) {
    return make_expr<tags::terminal>(std::forward<T>(x));
  }

  // Make a call node
  template <typename Obj, typename... Args>
  requires(Lazy<Args> or ... or Lazy<Obj>) //
     auto make_expr_call(Obj &&obj, Args &&...args) {
    return make_expr<tags::function>(std::forward<Obj>(obj), std::forward<Args>(args)...);
  }

  // Make a subscript node
  template <typename Obj, typename... Args>
  requires(Lazy<Args> or ... or Lazy<Obj>) //
     auto make_expr_subscript(Obj &&obj, Args &&...args) {
    return make_expr<tags::subscript>(std::forward<Obj>(obj), std::forward<Args>(args)...);
  }

  // Make a subscript node : splat the comma_tuple when building teh expression
  // the expression allows for multiple arguments in [] (they are gathered back to a tuple at evaluation)
  template <typename Obj, typename... T>
  requires(Lazy<T> or ... or Lazy<Obj>) //
     auto make_expr_subscript(Obj &&obj, comma_tuple<T...> const &tu) {
    return [&tu, &obj ]<auto... Is>(std::index_sequence<Is...>) { return make_expr_subscript(std::forward<Obj>(obj), std::get<Is>(tu._t)...); }
    (std::make_index_sequence<sizeof...(T)>());
  }

  /* ---------------------------------------------------------------------------------------------------
  *  Placeholder 
  *  --------------------------------------------------------------------------------------------------- */

  // a placeholder is an empty struct, labelled by an int.
  template <int N>
  struct placeholder {
    static constexpr int index = N;
    template <typename RHS>
    phvp<N, RHS> operator=(RHS &&rhs) const { // NOLINT It is correct here to return a phvp, not the placeholder& as suggested by clang-tidy
      return {std::forward<RHS>(rhs)};
    }
    template <typename... T>
    auto operator()(T &&...x) const {
      return make_expr_call(*this, std::forward<T>(x)...);
    }
    template <typename T>
    auto operator[](T &&x) const {
      return make_expr_subscript(*this, std::forward<T>(x));
    }
  };

  // --- is_lazy_v ----
  template <int N>
  constexpr bool is_lazy_v<placeholder<N>> = true;

  // --- force_copy ----
  // placeholder will always be copied (they are empty anyway).
  template <int N>
  constexpr bool force_copy_in_expr<placeholder<N>> = true;

  // --- contains_other_ph ----
  template <int N, int... Is>
  static constexpr bool contains_other_ph<placeholder<N>, Is...> = (not((N == Is) or ...));

  // ----- comma_tuple_seed_v is true ------
  template <int N>
  constexpr bool is_comma_tuple_seed_v<placeholder<N>> = true;

  /* ---------------------------------------------------------------------------------------------------
  *  Placeholder list at compile time
  *  --------------------------------------------------------------------------------------------------- */

  template <int... Is>
  struct phlist {
    static constexpr int size = sizeof...(Is);
  };

  /* ---------------------------------------------------------------------------------------------------
  * expr_as_function : an expression transformed to a function given a ordered list of placeholders
  * make_function : transform an expression to a function
  *  --------------------------------------------------------------------------------------------------- */

  template <typename Expr, int... Is>
  struct expr_as_function {
    Expr ex; // keep a copy of the expression

    template <typename... Args>
    FORCEINLINE decltype(auto) operator()(Args &&...args) const {
      return eval(ex, phvp<Is, Args>{std::forward<Args>(args)}...);
    }
  };

  // Use CTAD
  template <typename Expr, typename... Phs>
  FORCEINLINE auto make_function(Expr ex, Phs...) {
    return expr_as_function<Expr, Phs::index...>{std::move(ex)};
  }

  // same function, from a phlist
  template <typename Expr, int... Is>
  FORCEINLINE auto make_function(Expr ex, phlist<Is...>) {
    return expr_as_function<Expr, Is...>{std::move(ex)};
  }

  // --- is_lazy_v ----
  template <typename Expr, int... Is>
  constexpr bool is_lazy_v<expr_as_function<Expr, Is...>> = contains_other_ph<Expr, Is...>;

  // --- force_copy ----
  template <typename Expr, int... Is>
  constexpr bool force_copy_in_expr<expr_as_function<Expr, Is...>> = true;

  // --- contains_other_ph ----
  template <typename Expr, int... Js, int... Is>
  static constexpr bool contains_other_ph<expr_as_function<Expr, Js...>, Is...> = contains_other_ph<Expr, Is..., Js...>;

  /* ---------------------------------------------------------------------------------------------------
  * is_expr_as_function_v : detects expr_as_function
  *  --------------------------------------------------------------------------------------------------- */

  // A trait to easily detect such an object in generic code.
  template <typename T>
  constexpr bool is_expr_as_function_v = false;

  template <typename Expr, int... Is>
  constexpr bool is_expr_as_function_v<expr_as_function<Expr, Is...>> = true;

  /* ---------------------------------------------------------------------------------------------------
  * generated code : all basic operators and invoke_operation 
  *  --------------------------------------------------------------------------------------------------- */
} // namespace nda::clef

#include "clef_ops.hxx"

namespace nda::clef {
  /* ---------------------------------------------------------------------------------------------------
  * Evaluation
  *  --------------------------------------------------------------------------------------------------- */

  // eval (x, phvp...) is the evaluation function.
  // it evaluate x in the context of the phvp, which are a "placeholder-value-pair"
  // replacing recursively the placeholder by the value.
  // if some placeholder remain unevaluated, we get a new expr
  // otherwise, we get whatever the expression evaluates to.

  // Evaluate default : do nothing. See other overloads below
  template <typename T, typename... Phvps>
  FORCEINLINE decltype(auto) eval(T const &x, Phvps const &...) {
    return x;
  }

  // Evaluate a placeholder
  template <int N, int... Is, typename... T>
  FORCEINLINE decltype(auto) eval(placeholder<N> x, phvp<Is, T> const &...phvps) {

    static_assert(all_different(Is...), "Evaluation context : the placeholder indices must be all different");

    // the position of N in the Is... : -1 if not present
    // NB works before of the assert above.
    constexpr int position_of_N = []<auto... Ps>(std::index_sequence<Ps...>) { return ((Is == N ? int(Ps) + 1 : 0) + ...) - 1; }
    (std::make_index_sequence<sizeof...(Is)>{});

    if constexpr (position_of_N != -1) { // N is one of the Is
      return std::get<position_of_N>(std::tie(phvps...)).rhs;
    } else // N is not one of the Is
      return x;
  }

  // Evaluate a reference_wrapper. Just remove the wrapper.
  template <typename T, typename... Phvps>
  FORCEINLINE decltype(auto) eval(std::reference_wrapper<T> x, Phvps const &...phvps) {
    return eval(x.get(), phvps...);
  }

  // Evaluate a general expression node a context of phvps
  template <typename Tag, typename... Childs, typename... Phvps>
  FORCEINLINE decltype(auto) eval(expr<Tag, Childs...> const &ex, Phvps const &...phvps) {

    // we evaluate the children for all Is : eval(std::get<Is>(ex.children), phvps...)
    // if one of them is lazy ( Lazy <decltype(...)> , we build another expression
    // if not we invoke the operation and return whatever the operation has returned...
    // The lambda at the top is just here to splat the children tuple into a pack
    //
    return [&phvps..., &ex ]<auto... Is>(std::index_sequence<Is...>) {
      if constexpr ((Lazy<decltype(eval(std::get<Is>(ex.children), phvps...))> or ... or false))
        return make_expr<Tag>(eval(std::get<Is>(ex.children), phvps...)...);
      else {
        return invoke_operation(Tag{}, eval(std::get<Is>(ex.children), phvps...)...);
      }
    }
    (std::make_index_sequence<sizeof...(Childs)>{});
  }

  // Evaluate a expr_as_function
  // we just evaluate the expression ... and make sure it is still lazy (?).
  template <typename Expr, int... Is, typename... Phvps>
  FORCEINLINE decltype(auto) eval(expr_as_function<Expr, Is...> const &f, Phvps const &...phvps) {
    static_assert(Lazy<decltype(eval(f.ex, phvps...))>, "Evaluation of a expr_as_function is not a function");
    return make_function(eval(f.ex, phvps...), placeholder<Is>{}...);
  }

  /* ---------------------------------------------------------------------------------------------------
  * Auto assign for ()
  *  --------------------------------------------------------------------------------------------------- */

  // Principle. For any object X
  // X (ph1a, ph1b, ph1c, ..)[ph2a, ph2b, ph2c] << RHS
  // is replaced by
  // clef_auto_assign (X, RHS, Tag1, phlist1, Tag2, phlist2, ....)
  // where Tagi is tags::function for () or tags::subscript for []
  // Each object has to write a clef_auto_assign function and do whatever it needs with this information
  // cf basic_functions for the nda::array
  // RHS has to be a clef lazy expression

  // a little convenient concept
  template <typename Tag>
  concept CallOrSubscriptTag = std::is_same_v<Tag, tags::function> or std::is_same_v<Tag, tags::subscript>;

  // ---------- operator << ------

  // we match  X(placeholders...) << expr and X[placeholders...] << expr
  // other operator << are not defined.
  template <CallOrSubscriptTag Tag, typename X, typename RHS, int... Is>
  FORCEINLINE void operator<<(expr<Tag, X, placeholder<Is>...> const &ex, RHS &&rhs) {
    static_assert(all_different(Is...), "Illegal expression : two of the placeholders on the LHS of << are the same.");
    clef_auto_assign_prepare(ex, std::forward<RHS>(rhs));
  }

  // ---------- clef_auto_assign_prepare : internal  ------

  // X (ph1a, ph1b, ph1c, ..)[ph2a, ph2b, ph2c] << RHS
  // we recursively accumulate the Tag and the phlist as arguments.
  // CTargs are compile time args : phlist or Tag, which have no state.
  template <CallOrSubscriptTag Tag, typename X, typename RHS, int... Is, typename... CTArgs>
  FORCEINLINE void clef_auto_assign_prepare(expr<Tag, X, placeholder<Is>...> const &ex, RHS &&rhs, CTArgs... ctargs) {
    clef_auto_assign_prepare(std::get<0>(ex.children), std::forward<RHS>(rhs), Tag{}, phlist<Is...>{}, ctargs...);
  }

  // If ex is not an expression of the right form, we stop accumulating the phlist and tags and call clef_auto_assign
  template <typename T, typename RHS, typename... CTArgs>
  FORCEINLINE void clef_auto_assign_prepare(T &&x, RHS &&rhs, CTArgs... ctargs) {
    clef_auto_assign(std::forward<T>(x), std::forward<RHS>(rhs), ctargs...);
  }

  // ---------- clef_auto_assign for two basic types ------
  // we clean the reference wrapper and the terminal

  template <typename T, typename RHS, typename... CTArgs>
  FORCEINLINE void clef_auto_assign(std::reference_wrapper<T> x, RHS &&rhs, CTArgs... ctargs) {
    clef_auto_assign(x.get(), std::forward<RHS>(rhs), ctargs...);
  }

  template <typename T, typename RHS, typename... CTArgs>
  FORCEINLINE void clef_auto_assign(expr<tags::terminal, T> const &ex, RHS &&rhs, CTArgs... ctargs) {
    clef_auto_assign(std::get<0>(ex.children), std::forward<RHS>(rhs), ctargs...);
  }

  /* --------------------------------------------------------------------------------------------------
  *  The macro to make any function lazy
  *  CLEF_MAKE_FNT_LAZY (Arity,FunctionName ) : creates a new function in the triqs::lazy namespace
  *  taking expressions (at least one argument has to be an expression)
  *  The lookup happens by ADL, so IT MUST BE USED IN THE clef namespace
  * --------------------------------------------------------------------------------------------------- */

#define CLEF_MAKE_FNT_LAZY(name)                                                                                                                     \
  template <typename... A>                                                                                                                           \
  requires((nda::clef::Lazy<A> or ...)) auto name(A &&...a) {                                                                                        \
    return make_expr_call(                                                                                                                           \
       []<typename... T>(T && ...x)->decltype(auto) { return name(std::forward<T>(x)...); }, std::forward<A>(a)...);                                 \
  }

#define CLEF_IMPLEMENT_LAZY_METHOD(TY, name)                                                                                                         \
  template <typename... A>                                                                                                                           \
  requires((nda::clef::Lazy<A> or ...)) auto name(A &&...a) {                                                                                        \
    return make_expr_call(                                                                                                                           \
       []<typename Self, typename... T>(Self && self, T && ...x)->decltype(auto) { return std::forward<Self>(self).name(std::forward<T>(x)...); },   \
       *this, std::forward<A>(a)...);                                                                                                                \
  }

#define CLEF_IMPLEMENT_LAZY_CALL(...)                                                                                                                \
  template <typename... Args>                                                                                                                        \
  requires((nda::clef::Lazy<Args> or ...)) auto operator()(Args &&...args) const & { return make_expr_call(*this, std::forward<Args>(args)...); }    \
                                                                                                                                                     \
  template <typename... Args>                                                                                                                        \
  requires((nda::clef::Lazy<Args> or ...)) auto operator()(Args &&...args) & { return make_expr_call(*this, std::forward<Args>(args)...); }          \
                                                                                                                                                     \
  template <typename... Args>                                                                                                                        \
  requires((nda::clef::Lazy<Args> or ...)) auto operator()(Args &&...args) && {                                                                      \
    return make_expr_call(std::move(*this), std::forward<Args>(args)...);                                                                            \
  }

} // namespace nda::clef
