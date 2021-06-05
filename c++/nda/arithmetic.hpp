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
#include "linalg/matmul.hpp"
#include "linalg/det_and_inverse.hpp"

namespace nda {

  // Wrap a scalar as a constant array, with Algebra (A, M, ...)
  template <typename S, int Rank, char Algebra>
  struct scalar_array;

  // alias : S as a scalar with the Rank and Algebra of A
  template <typename S, typename A>
  using scalar_array_similar_to = scalar_array<std::decay_t<S>, get_rank<std::decay_t<A>>, get_algebra<std::decay_t<A>>>;

  // binary expression
  template <char OP, typename L, typename R>
  struct expr;

  // unary expression
  template <char OP, typename L>
  struct expr_unary;

  // algebra
  template <typename S, int R, char A>
  inline constexpr char get_algebra<scalar_array<S, R, A>> = A;

  template <char OP, typename L, typename R>
  inline constexpr char get_algebra<expr<OP, L, R>> = expr<OP, L, R>::algebra;

  template <char OP, typename L>
  inline constexpr char get_algebra<expr_unary<OP, L>> = expr_unary<OP, L>::algebra;

  // Both model NdArray concept
  template <char OP, typename L, typename R>
  inline constexpr bool is_ndarray_v<expr<OP, L, R>> = true;

  template <char OP, typename L>
  inline constexpr bool is_ndarray_v<expr_unary<OP, L>> = true;

  template <typename S, int Rank, char Algebra>
  inline constexpr bool is_ndarray_v<scalar_array<S, Rank, Algebra>> = true;

  // Get the layout info recursively
  template <char OP, typename L, typename R>
  inline constexpr layout_info_t get_layout_info<expr<OP, L, R>> = expr<OP, L, R>::layout_info;

  template <char OP, typename L>
  inline constexpr layout_info_t get_layout_info<expr_unary<OP, L>> = get_layout_info<std::decay_t<L>>;

  template <typename S, int Rank, char Algebra>
  inline constexpr layout_info_t get_layout_info<scalar_array<S, Rank, Algebra>> =
     layout_info_t{}; // NOT contiguous, we disable the linear optimisation of assign

  // -------------------------------------------------------------------------------------------
  //         A constant array
  // -------------------------------------------------------------------------------------------

  template <typename S, int Rank, char Algebra>
  struct scalar_array {
    S const s;
    std::array<long, Rank> _shape;

    [[nodiscard]] std::array<long, Rank> const &shape() const { return _shape; }

    S operator[](long) const requires(Rank == 1) { return s; }

    template <std::integral... Is>
    S operator()(Is...) const requires(Algebra == 'A') {
      static_assert(sizeof...(Is) == Rank, "Logic error");
      return s;
    }

    S operator()(long i1, long i2) const requires(Algebra == 'M') { return (i1 == i2 ? S{s} : S{}); }
  };

  // -------------------------------------------------------------------------------------------
  //                             binary expressions
  // -------------------------------------------------------------------------------------------
  // OP : '+', '-', ...
  template <char OP, typename L, typename R>
  struct expr {

    L l;
    R r;

    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    static constexpr bool l_is_scalar = nda::is_scalar_v<L_t>;
    static constexpr bool r_is_scalar = nda::is_scalar_v<R_t>;
    static constexpr char algebra     = (l_is_scalar ? get_algebra<R_t> : get_algebra<L_t>);
    static constexpr layout_info_t layout_info =
       (l_is_scalar ? get_layout_info<R_t> : (r_is_scalar ? get_layout_info<L_t> : get_layout_info<R_t> & get_layout_info<L_t>));

    //  --- shape ---
    [[nodiscard]] constexpr auto shape() const {
      if constexpr (l_is_scalar) {
        return r.shape();
      } else if constexpr (r_is_scalar) {
        return l.shape();
      } else {
        EXPECTS(l.shape() == r.shape());
        return l.shape();
      }
    }

    //  --- extent ---
    [[nodiscard]] constexpr long extent(int i) const noexcept {
      if constexpr (l_is_scalar) {
        return r.extent(i);
      } else if constexpr (r_is_scalar) {
        return l.extent(i);
      } else {
        //EXPECTS(l.extent(i) == r.extent(i));
        return l.extent(i);
      }
    }

    public:
    // FIXME Clef
    template <typename... Args>
    auto operator()(Args const &...args) const { //  REQUIRES(not(clef::is_lazy<A> and ...)) {

      // We simply implement all cases
      if constexpr (OP == '+') { // we KNOW that l and r are NOT scalar (cf operator +,- impl).
        return l(args...) + r(args...);
      }

      if constexpr (OP == '-') { return l(args...) - r(args...); }

      if constexpr (OP == '*') {
        if constexpr (l_is_scalar)
          return l * r(args...);
        else if constexpr (r_is_scalar)
          return l(args...) * r;
        else {
          static_assert(algebra != 'M', "Logic Error");
          return l(args...) * r(args...);
        }
      }

      if constexpr (OP == '/') {
        if constexpr (l_is_scalar)
          return l / r(args...);
        else if constexpr (r_is_scalar)
          return l(args...) / r;
        else {
          static_assert(algebra != 'M', "Logic Error");
          return l(args...) / r(args...);
        }
      }
    }

    // FIXME clef
    //TRIQS_CLEF_IMPLEMENT_LAZY_CALL(); // can not simply capture in () and dispatch becuase of && case. Cf macro def.

    private: // detail for operator [] below. get_rank does not work for a scalar, so I can not simply make a || list in the static_assert below
    static constexpr bool rank_is_one() {
      if constexpr (nda::is_scalar_v<L_t> || nda::is_scalar_v<R_t>)
        return true;
      else
        return ((get_rank<L_t> == 1) and (get_rank<R_t> == 1));
    }

    public:
    // FIXME
    // [long] ? 1d only ? strided only ?
    // Overload with _long ? long ? lazy ?
    /// [ ] is the same as (). Enable for Vectors only
    template <typename Arg>
    auto operator[](Arg const &arg) const {
      static_assert(rank_is_one(), "operator[] only available for array of rank 1");
      return operator()(std::forward<Arg>(arg));
    }

    // just for better error messages
    template <typename T>
    void operator=(T &&x) = delete; // expressions are immutable
    template <typename T>
    void operator+=(T &&x) = delete; // expressions are immutable
    template <typename T>
    void operator-=(T &&x) = delete; // expressions are immutable
  };

  // -------------------------------------------------------------------------------------------
  //                             unary expressions
  // -------------------------------------------------------------------------------------------

  template <char OP, typename L>
  struct expr_unary {
    using L_t = std::decay_t<L>;
    L l;
    static constexpr char algebra = get_algebra<L_t>;
    static_assert(OP == '-', "Internal error");

    template <typename LL>
    expr_unary(LL &&l_) : l(std::forward<LL>(l_)) {}

    // FIXME clef
    template <typename... Args>
    auto operator()(Args &&...args) const { // REQUIRES(not(clef::is_lazy<L>))
      return -l(std::forward<Args>(args)...);
    }

    //    TRIQS_CLEF_IMPLEMENT_LAZY_CALL();

    [[nodiscard]] constexpr auto shape() const { return l.shape(); }

    // just for better error messages
    template <typename T>
    void operator=(T &&x) = delete; // expressions are immutable
    template <typename T>
    void operator+=(T &&x) = delete; // expressions are immutable
    template <typename T>
    void operator-=(T &&x) = delete; // expressions are immutable
  };

  // -------------------------------------------------------------------------------------------
  //                                 Operator overload
  // -------------------------------------------------------------------------------------------

  // ===== unary - ========
  template <Array A>
  expr_unary<'-', A> operator-(A &&a) {
    return {std::forward<A>(a)};
  }

  // ===== operator + ========
  template <Array L, Array R>
  Array auto operator+(L &&l, R &&r) {
    static_assert(get_rank<L> == get_rank<R>, "Rank mismatch in array addition");
    return expr<'+', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  // --- scalar ---
  template <Array A, typename S>
  Array auto operator+(A &&a, S &&s) { // S&& is MANDATORY for proper concept  Array <: typename to work
    return expr<'+', A, scalar_array_similar_to<S, A>>{std::forward<A>(a), {s, a.shape()}};
  }

  template <typename S, Array A>
  Array auto operator+(S &&s, A &&a) {
    return expr<'+', scalar_array_similar_to<S, A>, A>{{s, a.shape()}, std::forward<A>(a)};
  }

  // ===== operator - ========

  template <Array L, Array R>
  Array auto operator-(L &&l, R &&r) {
    static_assert(get_rank<L> == get_rank<R>, "Rank mismatch in array substract");
    return expr<'-', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  // --- scalar ---
  template <Array A, typename S>
  Array auto operator-(A &&a, S &&s) { // S&& is MANDATORY for proper concept  Array <: typename to work
    return expr<'-', A, scalar_array_similar_to<S, A>>{std::forward<A>(a), {s, a.shape()}};
  }

  template <typename S, Array A>
  Array auto operator-(S &&s, A &&a) {
    return expr<'-', scalar_array_similar_to<S, A>, A>{{s, a.shape()}, std::forward<A>(a)};
  }

  // ===== operator * ========

  template <Array L, Array R>
  auto operator*(L &&l, R &&r) {
    static constexpr char l_algebra = get_algebra<std::decay_t<L>>;
    static constexpr char r_algebra = get_algebra<std::decay_t<R>>;
    static_assert(l_algebra != 'V', "Error Can not multiply vector by an array or a matrix");

    // three cases (with algebras...) :  A * A or M * M or M * V
    if constexpr (l_algebra == 'A') {
      static_assert(r_algebra == 'A', "Error Try to multiply array * matrix or vector");
      static_assert(get_rank<L> == get_rank<R>, "Rank mismatch in array multiply");
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.shape() != r.shape()) NDA_RUNTIME_ERROR << "Matrix product : dimension mismatch in matrix product " << l.shape() << " " << r.shape();
#endif
      return expr<'*', L, R>{std::forward<L>(l), std::forward<R>(r)};
    } else { // l_algebra is M
      static_assert(r_algebra != 'A', "Error Can not multiply matrix by array");
      if constexpr (r_algebra == 'M')
        // matrix * matrix
        return matmul(std::forward<L>(l), std::forward<R>(r));
      else
        // matrix * vector
        return matvecmul(std::forward<L>(l), std::forward<R>(r));
    }
  }

  // --- scalar ---

  template <Array A, typename S>
  Array auto operator*(A &&a, S &&s) { // S&& is MANDATORY for proper concept  Array <: typename to work
    // copy the scalar. Not strictly necessary, but it is a good protection, e.g. s = 3; return s* A;
    return expr<'*', A, std::decay_t<S>>{a, s};
  }

  template <typename S, Array A>
  Array auto operator*(S &&s, A &&a) {
    return expr<'*', std::decay_t<S>, A>{s, a};
  }

  // ===== operator / ========

  template <Array L, Array R>
  Array auto operator/(L &&l, R &&r) {
    static constexpr char l_algebra = get_algebra<std::decay_t<L>>;
    static constexpr char r_algebra = get_algebra<std::decay_t<R>>;
    static_assert(l_algebra != 'V', "Error Can not divide vector by an array or a matrix");

    // two cases (with algebras...) :  A / A or M / M
    if constexpr (l_algebra == 'A') {
      static_assert(r_algebra == 'A', "Error Try to multiply array * matrix or vector");
      static_assert(get_rank<L> == get_rank<R>, "Rank mismatch in array multiply");
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.shape() != r.shape()) NDA_RUNTIME_ERROR << "Matrix product : dimension mismatch in matrix product " << l.shape() << " " << r.shape();
#endif
      return expr<'/', L, R>{std::forward<L>(l), std::forward<R>(r)};
    } else { // l_algebra is M
      static_assert(r_algebra == 'M', "Error Can only divide a matrix by a matrix (or scalar)");
      using R_t = std::decay_t<R>;
      return std::forward<L>(l) * inverse(matrix<get_value_t<R_t>>{std::forward<R>(r)});
    }
  }

  // --- scalar ---

  template <Array A, typename S>
  Array auto operator/(A &&a, S &&s) { // S&& is MANDATORY for proper concept  Array <: typename to work
    return expr<'/', A, std::decay_t<S>>{a, s};
  }

  template <typename S, Array A>
  Array auto operator/(S &&s, A &&a) {
    using A_t                     = std::decay_t<A>;
    static constexpr char algebra = get_algebra<A_t>;
    if constexpr (algebra == 'M')
      return s * inverse(matrix<get_value_t<A_t>>{std::forward<A>(a)});
    else
      return expr<'/', std::decay_t<S>, A>{s, a};
  }

} // namespace nda
