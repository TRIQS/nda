// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/nda.hpp>

#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>

using dcomplex = std::complex<double>;
using view_t   = nda::matrix_view<dcomplex>;
using mat_t    = nda::matrix<dcomplex>;

// Storage type of the single operand of a lazy nda::expr_call.
template <typename E>
using call_operand_t = std::tuple_element_t<0, decltype(E::a)>;

// Pointer type that nda::blas_lapack::get_array hands out for an expression of type E.
template <typename E>
using get_array_ptr_t = decltype(nda::blas_lapack::get_array(std::declval<E>()).data());

// Lazy expressions built from lvalue operands.
using conj_lv_t = decltype(nda::conj(std::declval<view_t &>()));
using neg_lv_t  = decltype(-std::declval<view_t &>());
using plus_lv_t = decltype(std::declval<view_t &>() + std::declval<view_t &>());
using scal_lv_t = decltype(std::declval<double &>() * std::declval<view_t &>());

// Lazy expressions built from rvalue operands.
using conj_rv_t = decltype(nda::conj(mat_t{2, 2}));
using neg_rv_t  = decltype(-mat_t{2, 2});

TEST(NDA, ExprLvalueOperandsAreBoundByRef) {
  // lvalue operands are bound by reference, so building a lazy expression never copies array data
  static_assert(std::is_same_v<call_operand_t<conj_lv_t>, view_t &>);
  static_assert(std::is_same_v<decltype(neg_lv_t::a), view_t &>);
  static_assert(std::is_same_v<decltype(plus_lv_t::l), view_t &>);
  static_assert(std::is_same_v<decltype(plus_lv_t::r), view_t &>);
  static_assert(std::is_same_v<decltype(scal_lv_t::r), view_t &>);
}

TEST(NDA, ExprRvalueAndScalarOperandsAreStoredByValue) {
  // rvalue operands are moved into the expression, and scalars are stored by value even when passed as
  // lvalues (see DanglingScalarIssue)
  static_assert(std::is_same_v<call_operand_t<conj_rv_t>, mat_t>);
  static_assert(std::is_same_v<decltype(neg_rv_t::a), mat_t>);
  static_assert(std::is_same_v<decltype(scal_lv_t::l), double>);
}

TEST(NDA, GetArrayPropagatesExprConstness) {
  // nda::blas_lapack::get_array applies the constness of the expression object to the operand, so a
  // const conj expression can only ever be a read-only BLAS/tensor operand, while a non-const one is
  // still usable as a destination
  static_assert(std::is_same_v<get_array_ptr_t<conj_lv_t &>, dcomplex *>);
  static_assert(std::is_same_v<get_array_ptr_t<conj_lv_t>, dcomplex *>);
  static_assert(std::is_same_v<get_array_ptr_t<conj_lv_t const &>, dcomplex const *>);

  // the conjugate of a const array stays const, regardless of the constness of the expression itself
  using conj_cv_t = decltype(nda::conj(std::declval<mat_t const &>()));
  static_assert(std::is_same_v<get_array_ptr_t<conj_cv_t &>, dcomplex const *>);
}

TEST(NDA, GetArrayDoesNotCopy) {
  // propagating the constness neither copies nor repoints the operand
  auto mat      = mat_t{2, 2};
  auto v        = mat();
  auto e        = nda::conj(v);
  auto const ce = nda::conj(v);

  EXPECT_EQ(nda::blas_lapack::get_array(e).data(), mat.data());
  EXPECT_EQ(nda::blas_lapack::get_array(ce).data(), mat.data());
}
