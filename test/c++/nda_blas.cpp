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

#include "test_common.hpp"

#include <nda/blas.hpp>
#include <nda/clef/literals.hpp>

using nda::F_layout;
using namespace clef::literals;

//----------------------------

template <typename value_t, typename Layout>
void test_gemm() {
  nda::matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3{{1, 0}, {0, 1}};
  nda::blas::gemm(1.0, M1, M2, 1.0, M3);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<value_t>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<value_t>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});

  // batch strided
  nda::array<value_t, 3> A1{{{0, 1}, {1, 2}}, {{0, 1}, {1, 2}}}, A2{{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}}, A3{{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}};
  nda::blas::gemm_batch_strided(1.0, A1, A2, 1.0, A3);

  EXPECT_ARRAY_NEAR(A1, nda::array<value_t, 3>{{{0, 1}, {1, 2}}, {{0, 1}, {1, 2}}});
  EXPECT_ARRAY_NEAR(A2, nda::array<value_t, 3>{{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}});
  EXPECT_ARRAY_NEAR(A3, nda::array<value_t, 3>{{{2, 1}, {3, 4}}, {{2, 1}, {3, 4}}});
}

TEST(BLAS, gemm) { test_gemm<double, C_layout>(); }     //NOLINT
TEST(BLAS, gemmF) { test_gemm<double, F_layout>(); }    //NOLINT
TEST(BLAS, zgemm) { test_gemm<dcomplex, C_layout>(); }  //NOLINT
TEST(BLAS, zgemmF) { test_gemm<dcomplex, F_layout>(); } //NOLINT

// ==============================================================

template <typename value_t, typename Layout>
void test_gemv() {

  nda::matrix<value_t, Layout> A(5, 5);
  A(i_, j_) << i_ + 2 * j_ + 1;

  nda::vector<value_t> MC(5), MB(5);
  MC() = 1;
  MB() = 0;

  nda::range R(1, 3);
  nda::blas::gemv(1, A(R, R), MC(R), 0, MB(R));
  EXPECT_ARRAY_NEAR(MB, nda::vector<value_t>{0, 10, 12, 0, 0});

  auto AT = make_regular(transpose(A));
  nda::blas::gemv(1, AT(R, R), MC(R), 0, MB(R));
  EXPECT_ARRAY_NEAR(MB, nda::vector<value_t>{0, 9, 13, 0, 0});

  // test operator*
  MB()  = -8;
  MB(R) = AT(R, R) * MC(R);
  EXPECT_ARRAY_NEAR(MB, nda::vector<value_t>{-8, 9, 13, -8, -8});
}

TEST(BLAS, gemv) { test_gemv<double, C_layout>(); }     //NOLINT
TEST(BLAS, gemvF) { test_gemv<double, F_layout>(); }    //NOLINT
TEST(BLAS, zgemv) { test_gemv<dcomplex, C_layout>(); }  //NOLINT
TEST(BLAS, zgemvF) { test_gemv<dcomplex, F_layout>(); } //NOLINT

//----------------------------

template <typename value_t, typename Layout>
void test_ger() {

  nda::matrix<value_t, Layout> M(2, 2);
  M = 0;
  nda::array<value_t, 1> V{1, 2};

  nda::blas::ger(1.0, V, V, M);
  EXPECT_ARRAY_NEAR(M, nda::matrix<value_t>{{1, 2}, {2, 4}});
}

TEST(BLAS, dger) { test_ger<double, C_layout>(); }    //NOLINT
TEST(BLAS, dgerF) { test_ger<double, F_layout>(); }   //NOLINT
TEST(BLAS, zger) { test_ger<dcomplex, C_layout>(); }  //NOLINT
TEST(BLAS, zgerF) { test_ger<dcomplex, C_layout>(); } //NOLINT

//----------------------------

TEST(BLAS, outer_product) { //NOLINT

  auto N = nda::rand<double>(2, 3);
  auto M = nda::rand<double>(4, 5);

  nda::array<double, 4> P(2, 3, 4, 5);

  for (auto [i, j] : N.indices())
    for (auto [k, l] : M.indices()) P(i, j, k, l) = N(i, j) * M(k, l);

  EXPECT_ARRAY_NEAR(P, (nda::blas::outer_product(N, M)));
}

//----------------------------

template <typename value_t>
void test_dot() { //NOLINT

  nda::vector<value_t> a{1, 2, 3, 4, 5};
  nda::vector<value_t> b{10, 20, 30, 40, 50};
  if constexpr (nda::is_complex_v<value_t>) {
    a *= 1 + 1i;
    b *= 1 + 2i;
  }

  EXPECT_COMPLEX_NEAR((nda::blas::dot(a, b)), (nda::blas::dot_generic(a, b)), 1.e-14);
}

TEST(BLAS, ddot) { test_dot<double>(); }   //NOLINT
TEST(BLAS, zdot) { test_dot<dcomplex>(); } //NOLINT

//----------------------------

template <typename value_t>
void test_dotc() { //NOLINT

  nda::vector<value_t> a{1, 2, 3, 4, 5};
  nda::vector<value_t> b{10, 20, 30, 40, 50};
  if constexpr (nda::is_complex_v<value_t>) {
    a *= 1 + 1i;
    b *= 1 + 2i;
  }

  EXPECT_COMPLEX_NEAR((nda::blas::dotc(a, b)), (nda::blas::dotc_generic(a, b)), 1.e-14);
}

TEST(BLAS, ddotc) { test_dotc<double>(); }   //NOLINT
TEST(BLAS, zdotc) { test_dotc<dcomplex>(); } //NOLINT
