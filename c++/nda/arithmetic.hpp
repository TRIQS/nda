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
#include "concepts.hpp"
#include "linalg/matmul.hpp"
#include "linalg/det_and_inverse.hpp"

// arithmetic expression with object of Array concept
// expr and expr_unary are the expression template
// then overload operators.
namespace nda {

  // -------------------------------------------------------------------------------------------
  //                             unary expressions
  // -------------------------------------------------------------------------------------------

  template <char OP, typename L>
  struct expr_unary {
    static_assert(OP == '-', "Internal error");
    L l;

    template <typename... Args>
    auto operator()(Args &&...args) const {
      return -l(std::forward<Args>(args)...);
    }

    [[nodiscard]] constexpr auto shape() const { return l.shape(); }
    [[nodiscard]] constexpr long size() const { return l.size(); }
  }; // end expr_unary class

  // get_algebra
  template <char OP, typename L>
  inline constexpr char get_algebra<expr_unary<OP, L>> = get_algebra<L>;

  // get_layout_info
  template <char OP, typename L>
  inline constexpr layout_info_t get_layout_info<expr_unary<OP, L>> = get_layout_info<L>;

  // -------------------------------------------------------------------------------------------
  //                             binary expressions
  // -------------------------------------------------------------------------------------------
  // OP : '+', '-', ...
  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  struct expr {

    L l;
    R r;

    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    // FIXME : we should use is_scalar_for_v but the trait needs work to accomodate scalar L or R
    static constexpr bool l_is_scalar = nda::is_scalar_v<L>;
    static constexpr bool r_is_scalar = nda::is_scalar_v<R>;
    static constexpr char algebra     = (l_is_scalar ? get_algebra<R> : get_algebra<L>);

    static constexpr layout_info_t compute_layout_info() {
      if (l_is_scalar) return (algebra == 'A' ? get_layout_info<R> : layout_info_t{}); // 1 as an array has all flags, it is just 1
      if (r_is_scalar) return (algebra == 'A' ? get_layout_info<L> : layout_info_t{}); // 1 as a matrix does not, as it is diagonal only.
      return get_layout_info<R> & get_layout_info<L>;                                  // default case. Take the logical and of all flags
    }

    //  --- shape ---
    [[nodiscard]] constexpr decltype(auto) shape() const {
      if constexpr (l_is_scalar) {
        return r.shape();
      } else if constexpr (r_is_scalar) {
        return l.shape();
      } else {
        EXPECTS(l.shape() == r.shape());
        return l.shape();
      }
    }

    //  --- size ---
    [[nodiscard]] constexpr long size() const {
      if constexpr (l_is_scalar) {
        return r.size();
      } else if constexpr (r_is_scalar) {
        return l.size();
      } else {
        EXPECTS(l.size() == r.size());
        return l.size();
      }
    }

    template <typename... Args>
    auto operator()(Args const &...args) const {

      if constexpr (OP == '+') {
        if constexpr (l_is_scalar) {
          if constexpr (algebra == 'M')
            // args... is of size 2,
            return (std::equal_to{}(args...) ? l + r(args...) : r(args...));
          else
            return l + r(args...);
        } else if constexpr (r_is_scalar) {
          if constexpr (algebra == 'M')
            return (std::equal_to{}(args...) ? l(args...) + r : l(args...));
          else
            return l(args...) + r;
        } else
          return l(args...) + r(args...);
      }

      if constexpr (OP == '-') {
        if constexpr (l_is_scalar) {
          if constexpr (algebra == 'M') // ... of size 2
            return (std::equal_to{}(args...) ? l - r(args...) : -r(args...));
          else
            return l - r(args...);
        } else if constexpr (r_is_scalar) {
          if constexpr (algebra == 'M')
            return (std::equal_to{}(args...) ? l(args...) - r : l(args...));
          else
            return l(args...) - r;
        } else
          return l(args...) - r(args...);
      }

      if constexpr (OP == '*') {
        if constexpr (l_is_scalar)
          return l * r(args...);
        else if constexpr (r_is_scalar)
          return l(args...) * r;
        else {
          static_assert(algebra != 'M', "We should not be here");
          return l(args...) * r(args...);
        }
      }

      if constexpr (OP == '/') {
        if constexpr (l_is_scalar) {
          static_assert(algebra != 'M', "We should not be here");
          return l / r(args...);
        } else if constexpr (r_is_scalar)
          return l(args...) / r;
        else {
          static_assert(algebra != 'M', "We should not be here");
          return l(args...) / r(args...);
        }
      }
    }

    template <typename Arg>
    auto operator[](Arg const &arg) const {
      static_assert(get_rank<expr> == 1, "operator[] only available for expression of rank 1");
      return operator()(std::forward<Arg>(arg));
    }
  }; // end expr class

  // get_algebra
  template <char OP, typename L, typename R>
  inline constexpr char get_algebra<expr<OP, L, R>> = expr<OP, L, R>::algebra;

  // get_layout_info
  template <char OP, typename L, typename R>
  inline constexpr layout_info_t get_layout_info<expr<OP, L, R>> = expr<OP, L, R>::compute_layout_info();

  // -------------------------------------------------------------------------------------------
  //                                 Operator overload
  // -------------------------------------------------------------------------------------------

  // ===== unary - ========
  template <Array A>
  expr_unary<'-', A> operator-(A &&a) {
    return {std::forward<A>(a)};
  }

  // ===== operator + ========
  // first case : array + array
  // then array + scalar and scalar + array
  // FIXME we copy the scalar. Should we ?
  // FIXME we can regroup in one with requires(Array<L> or Array<R>) but then it is harder to assert rank as scalar have no rank.
  // choice : keep it simple like this

  template <Array L, Array R>
  Array auto operator+(L &&l, R &&r) {
    static_assert(get_rank<L> == get_rank<R>, "Rank mismatch in array addition");
    return expr<'+', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  // --- scalar ---
  // S is not an array, so it is treated as a scalar.
  template <Array A, Scalar S>
  Array auto operator+(A &&a, S &&s) { // S&& is MANDATORY for proper concept  Array <: typename to work
    return expr<'+', A, std::decay_t<S>>{std::forward<A>(a), s};
  }

  template <Scalar S, Array A>
  Array auto operator+(S &&s, A &&a) {
    return expr<'+', std::decay_t<S>, A>{s, std::forward<A>(a)};
  }

  // ===== operator - ========

  template <Array L, Array R>
  Array auto operator-(L &&l, R &&r) {
    static_assert(get_rank<L> == get_rank<R>, "Rank mismatch in array substract");
    return expr<'-', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  // --- scalar ---
  template <Array A, Scalar S>
  Array auto operator-(A &&a, S &&s) { // S&& is MANDATORY for proper concept  Array <: typename to work
    return expr<'-', A, std::decay_t<S>>{std::forward<A>(a), s};
  }

  template <Scalar S, Array A>
  Array auto operator-(S &&s, A &&a) {
    return expr<'-', std::decay_t<S>, A>{s, std::forward<A>(a)};
  }

  // ===== operator * ========

  template <Array L, Array R>
  auto operator*(L &&l, R &&r) {
    static constexpr char l_algebra = get_algebra<L>;
    static constexpr char r_algebra = get_algebra<R>;
    static_assert(l_algebra != 'V', "Error Can not multiply vector by an array or a matrix");

    // three cases (with algebras...) :  A * A or M * M or M * V
    // A * ?
    if constexpr (l_algebra == 'A') {
      static_assert(r_algebra == 'A', "Error Try to multiply array * matrix or vector");
      static_assert(get_rank<L> == get_rank<R>, "Rank mismatch in array multiply");
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.shape() != r.shape()) NDA_RUNTIME_ERROR << "Matrix product : dimension mismatch in matrix product " << l.shape() << " " << r.shape();
#endif
      return expr<'*', L, R>{std::forward<L>(l), std::forward<R>(r)};
    }
    // M * ?
    if constexpr (l_algebra == 'M') {
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

  template <Array A, Scalar S>
  Array auto operator*(A &&a, S &&s) { // S&& is MANDATORY for proper concept  Array <: typename to work
    // copy the scalar. Not strictly necessary, but it is a good protection, e.g. s = 3; return s* A;
    return expr<'*', A, std::decay_t<S>>{std::forward<A>(a), s};
  }

  template <Scalar S, Array A>
  Array auto operator*(S &&s, A &&a) {
    return expr<'*', std::decay_t<S>, A>{s, std::forward<A>(a)};
  }

  // ===== operator / ========

  template <Array L, Array R>
  Array auto operator/(L &&l, R &&r) {
    static constexpr char l_algebra = get_algebra<L>;
    static constexpr char r_algebra = get_algebra<R>;
    static_assert(l_algebra != 'V', "Error Can not divide vector by an array or a matrix");

    // two cases (with algebras...) :  A / A or M / M
    // A / A
    if constexpr (l_algebra == 'A') {
      static_assert(r_algebra == 'A', "Error Try to multiply array * matrix or vector");
      static_assert(get_rank<L> == get_rank<R>, "Rank mismatch in array multiply");
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.shape() != r.shape()) NDA_RUNTIME_ERROR << "Matrix product : dimension mismatch in matrix product " << l.shape() << " " << r.shape();
#endif
      return expr<'/', L, R>{std::forward<L>(l), std::forward<R>(r)};
    }
    // M / M
    if constexpr (l_algebra == 'M') {
      static_assert(r_algebra == 'M', "Error Can only divide a matrix by a matrix (or scalar)");
      return std::forward<L>(l) * inverse(matrix<get_value_t<R>>{std::forward<R>(r)});
    }
  }

  // --- scalar ---

  template <Array A, Scalar S>
  Array auto operator/(A &&a, S &&s) { // S&& is MANDATORY for proper concept  Array <: typename to work
    return expr<'/', A, std::decay_t<S>>{std::forward<A>(a), s};
  }

  template <Scalar S, Array A>
  Array auto operator/(S &&s, A &&a) {
    static constexpr char algebra = get_algebra<A>;
    if constexpr (algebra == 'M')
      return s * inverse(matrix<get_value_t<A>>{std::forward<A>(a)});
    else
      return expr<'/', std::decay_t<S>, A>{s, std::forward<A>(a)};
  }

} // namespace nda
