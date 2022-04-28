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

TEST(CUBLAS, gemv) { //NOLINT

  nda::matrix<double, F_layout> A(5, 5);
  A(i_, j_) << i_ + 2 * j_ + 1;

  nda::vector<double> MC(5), MB(5);
  MC() = 1;
  MB() = 0;

  nda::cumatrix<double, F_layout> A_d{A};
  nda::cuvector<double> MC_d{MC}, MB_d{MB};

  nda::range R(1, 3);
  nda::blas::gemv(1, A_d(R, R), MC_d(R), 0, MB_d(R));
  MB = MB_d;
  EXPECT_ARRAY_NEAR(MB, nda::array<double, 1>{0, 10, 12, 0, 0});

  nda::cumatrix_view<double> AT_d = transpose(A_d);
  nda::blas::gemv(1, AT_d(R, R), MC_d(R), 0, MB_d(R));
  MB = MB_d;
  EXPECT_ARRAY_NEAR(MB, nda::vector<double>{0, 9, 13, 0, 0});

  // test operator*
  MB_d(R) = AT_d(R, R) * MC_d(R);
  MB()    = -8;
  MB(R)   = MB_d(R);
  EXPECT_ARRAY_NEAR(MB, nda::vector<double>{-8, 9, 13, -8, -8});
}

//----------------------------
TEST(CUBLAS, ger) { //NOLINT

  nda::matrix<double, F_layout> M(2, 2);
  M = 0;
  nda::array<double, 1> V{1, 2};

  nda::cumatrix<double, F_layout> M_d{M};
  nda::cuvector<double> V_d{V};

  nda::blas::ger(1.0, V_d, V_d, M_d);

  M = M_d;
  EXPECT_ARRAY_NEAR(M, nda::matrix<double>{{1, 2}, {2, 4}});
}

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
TEST(CUBLAS, dot) { //NOLINT

  nda::vector<double> a{1, 2, 3, 4, 5};
  nda::vector<double> b{10, 20, 30, 40, 50};

  nda::cuvector<double> a_d{a}, b_d{b};
  EXPECT_NEAR((nda::blas::dot(a_d, b_d)), (10 + 2 * 20 + 3 * 30 + 4 * 40 + 5 * 50), 1.e-14);
  EXPECT_COMPLEX_NEAR((nda::blas::dotc(a_d, b_d)), (10 + 2 * 20 + 3 * 30 + 4 * 40 + 5 * 50), 1.e-14);
}

//----------------------------
TEST(CUBLAS, dotc1) { //NOLINT

  nda::vector<dcomplex> a{1, 2, 3};
  nda::vector<dcomplex> b{10, 20, 30};
  a *= 1 + 1i;
  b *= 1 + 2i;

  nda::cuvector<dcomplex> a_d{a}, b_d{b};
  EXPECT_COMPLEX_NEAR((nda::blas::dotc(a_d, b_d)), (1 - 1i) * (1 + 2i) * (10 + 2 * 20 + 3 * 30), 1.e-14);
}
