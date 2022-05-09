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

static_assert(not std::is_constructible_v<long, nda::range>, "ioio");

//----------------------------

template <typename value_t, typename Layout>
void test_gemm() {
  nda::matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3{{1, 0}, {0, 1}};
  nda::cumatrix<value_t, Layout> M1_d{M1}, M2_d{M2}, M3_d{M3};

  nda::blas::gemm(1.0, M1_d, M2_d, 1.0, M3_d);
  M3 = M3_d;

  EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});
}

TEST(CUBLAS, gemm) { test_gemm<double, C_layout>(); }     //NOLINT
TEST(CUBLAS, gemmF) { test_gemm<double, F_layout>(); }    //NOLINT
TEST(CUBLAS, zgemm) { test_gemm<dcomplex, C_layout>(); }  //NOLINT
TEST(CUBLAS, zgemmF) { test_gemm<dcomplex, F_layout>(); } //NOLINT

// ==============================================================

template <typename value_t, typename Layout>
void test_gemv() {

  nda::matrix<value_t, Layout> A(5, 5);
  A(i_, j_) << i_ + 2 * j_ + 1;

  nda::vector<value_t> MC(5), MB(5);
  MC() = 1;
  MB() = 0;

  nda::cumatrix<value_t, Layout> A_d{A};
  nda::cuvector<value_t> MC_d{MC}, MB_d{MB};

  nda::range R(1, 3);
  nda::blas::gemv(1, A_d(R, R), MC_d(R), 0, MB_d(R));
  MB = MB_d;
  EXPECT_ARRAY_NEAR(MB, nda::vector<value_t>{0, 10, 12, 0, 0});

  auto AT_d = transpose(A_d);
  nda::blas::gemv(1, AT_d(R, R), MC_d(R), 0, MB_d(R));
  MB = MB_d;
  EXPECT_ARRAY_NEAR(MB, nda::vector<value_t>{0, 9, 13, 0, 0});

  // test operator*
  MB_d(R) = AT_d(R, R) * MC_d(R);
  MB()    = -8;
  MB(R)   = MB_d(R);
  EXPECT_ARRAY_NEAR(MB, nda::vector<value_t>{-8, 9, 13, -8, -8});
}

TEST(CUBLAS, gemv) { test_gemv<double, C_layout>(); }     //NOLINT
TEST(CUBLAS, gemvF) { test_gemv<double, F_layout>(); }    //NOLINT
TEST(CUBLAS, zgemv) { test_gemv<dcomplex, C_layout>(); }  //NOLINT
TEST(CUBLAS, zgemvF) { test_gemv<dcomplex, F_layout>(); } //NOLINT

//----------------------------

template <typename value_t, typename Layout>
void test_ger() {

  nda::matrix<value_t, Layout> M(2, 2);
  M = 0;
  nda::array<value_t, 1> V{1, 2};

  nda::cumatrix<value_t, Layout> M_d{M};
  nda::cuvector<value_t> V_d{V};

  nda::blas::ger(1.0, V_d, V_d, M_d);

  M = M_d;
  EXPECT_ARRAY_NEAR(M, nda::matrix<value_t>{{1, 2}, {2, 4}});
}

TEST(CUBLAS, dger) { test_ger<double, C_layout>(); }    //NOLINT
TEST(CUBLAS, dgerF) { test_ger<double, F_layout>(); }   //NOLINT
TEST(CUBLAS, zger) { test_ger<dcomplex, C_layout>(); }  //NOLINT
TEST(CUBLAS, zgerF) { test_ger<dcomplex, C_layout>(); } //NOLINT

//----------------------------

TEST(CUBLAS, outer_product) { //NOLINT

  auto N = nda::rand<double>(2, 3);
  auto M = nda::rand<double>(4, 5);

  nda::array<double, 4> P(2, 3, 4, 5);

  for (auto [i, j] : N.indices())
    for (auto [k, l] : M.indices()) P(i, j, k, l) = N(i, j) * M(k, l);

  nda::cumatrix<double> M_d{M}, N_d{N};
  auto Res_d = nda::blas::outer_product(N_d, M_d);
  auto Res   = nda::array<double, 4>{Res_d};
  EXPECT_ARRAY_NEAR(P, Res);
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

  nda::cuvector<value_t> a_d{a}, b_d{b};
  EXPECT_COMPLEX_NEAR((nda::blas::dot(a_d, b_d)), (nda::blas::dot_generic(a, b)), 1.e-14);
}

TEST(CUBLAS, ddot) { test_dot<double>(); }   //NOLINT
TEST(CUBLAS, zdot) { test_dot<dcomplex>(); } //NOLINT

//----------------------------

template <typename value_t>
void test_dotc() { //NOLINT

  nda::vector<value_t> a{1, 2, 3, 4, 5};
  nda::vector<value_t> b{10, 20, 30, 40, 50};
  if constexpr (nda::is_complex_v<value_t>) {
    a *= 1 + 1i;
    b *= 1 + 2i;
  }

  nda::cuvector<value_t> a_d{a}, b_d{b};
  EXPECT_COMPLEX_NEAR((nda::blas::dotc(a_d, b_d)), (nda::blas::dotc_generic(a, b)), 1.e-14);
}

TEST(CUBLAS, ddotc) { test_dotc<double>(); }   //NOLINT
TEST(CUBLAS, zdotc) { test_dotc<dcomplex>(); } //NOLINT
