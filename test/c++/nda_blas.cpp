// Copyright (c) 2019-2022 Simons Foundation
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

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>
#include <vector>

// Test the BLAS gemm function and its generic implementation.
template <typename value_t, typename Layout>
void test_gemm() {
  nda::matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3{{1, 0}, {0, 1}}, M3_gen;
  M3_gen = M3;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);
  EXPECT_ARRAY_NEAR(M1, nda::matrix<value_t>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<value_t>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});

  nda::blas::gemm_generic(1.0, M1, M2, 1.0, M3_gen);
  EXPECT_ARRAY_NEAR(M1, nda::matrix<value_t>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<value_t>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3_gen, nda::matrix<value_t>{{2, 1}, {3, 4}});
}

TEST(NDA, BLASGemm) {
  test_gemm<double, nda::C_layout>();
  test_gemm<double, nda::F_layout>();
  test_gemm<std::complex<double>, nda::C_layout>();
  test_gemm<std::complex<double>, nda::F_layout>();
}

// Test the BLAS gemm_batch function.
template <typename value_t, typename Layout>
void test_gemm_batch() {
  int batch_count = 10;
  long size       = 64;

  auto vec_A = std::vector(batch_count, nda::matrix<value_t, Layout>::rand({size, size}));
  auto vec_B = std::vector(batch_count, nda::matrix<value_t, Layout>::rand({size, size}));
  auto vec_C = std::vector(batch_count, nda::matrix<value_t, Layout>::zeros({size, size}));
  nda::blas::gemm_batch(1.0, vec_A, vec_B, 0.0, vec_C);

  for (auto i : nda::range(batch_count)) EXPECT_ARRAY_NEAR(make_regular(vec_A[i] * vec_B[i]), vec_C[i]);
}

TEST(NDA, BLASGemmBatch) {
  test_gemm_batch<double, nda::C_layout>();
  test_gemm_batch<double, nda::F_layout>();
  test_gemm_batch<std::complex<double>, nda::C_layout>();
  test_gemm_batch<std::complex<double>, nda::F_layout>();
}

// Test the BLAS gemm_vbatch function.
template <typename value_t, typename Layout>
void test_gemm_vbatch() {
  int batch_count = 10;
  long size       = 64;

  auto vec_A = std::vector(batch_count, nda::matrix<value_t, Layout>::rand({size, size}));
  auto vec_B = std::vector(batch_count, nda::matrix<value_t, Layout>::rand({size, size}));
  auto vec_C = std::vector(batch_count, nda::matrix<value_t, Layout>::zeros({size, size}));
  nda::blas::gemm_vbatch(1.0, vec_A, vec_B, 0.0, vec_C);

  for (auto i : nda::range(batch_count)) EXPECT_ARRAY_NEAR(make_regular(vec_A[i] * vec_B[i]), vec_C[i]);
}

TEST(NDA, BLASGemmVbatch) {
  test_gemm_vbatch<double, nda::C_layout>();
  test_gemm_vbatch<double, nda::F_layout>();
  test_gemm_vbatch<std::complex<double>, nda::C_layout>();
  test_gemm_vbatch<std::complex<double>, nda::F_layout>();
}

// Test the BLAS gemv function and its generic implementation.
template <typename value_t, typename Layout>
void test_gemv() {
  using namespace nda::clef::literals;

  nda::matrix<value_t, Layout> A(5, 5);
  A(i_, j_) << i_ + 2 * j_ + 1;

  nda::vector<value_t> v(5), w(5);
  v() = 1;
  w() = 0;

  nda::range rg(1, 3);
  nda::blas::gemv(1, A(rg, rg), v(rg), 0, w(rg));
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{0, 10, 12, 0, 0});

  nda::vector<value_t> w_gen(5);
  w_gen() = 0;
  nda::blas::gemv_generic(1, A(rg, rg), v(rg), 0, w_gen(rg));
  EXPECT_ARRAY_NEAR(w_gen, nda::vector<value_t>{0, 10, 12, 0, 0});

  auto AT = nda::make_regular(transpose(A));
  nda::blas::gemv(1, AT(rg, rg), v(rg), 0, w(rg));
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{0, 9, 13, 0, 0});

  nda::blas::gemv_generic(1, AT(rg, rg), v(rg), 0, w_gen(rg));
  EXPECT_ARRAY_NEAR(w_gen, nda::vector<value_t>{0, 9, 13, 0, 0});

  // test operator*
  w()   = -8;
  w(rg) = AT(rg, rg) * v(rg);
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{-8, 9, 13, -8, -8});
}

TEST(NDA, BLASGemv) {
  test_gemv<double, nda::C_layout>();
  test_gemv<double, nda::F_layout>();
  test_gemv<std::complex<double>, nda::C_layout>();
  test_gemv<std::complex<double>, nda::F_layout>();
}

// Test the BLAS ger function.
template <typename value_t, typename Layout>
void test_ger() {
  nda::matrix<value_t, Layout> M(2, 2);
  M = 0;
  nda::array<value_t, 1> v{1, 2};

  nda::blas::ger(1.0, v, v, M);
  EXPECT_ARRAY_NEAR(M, nda::matrix<value_t>{{1, 2}, {2, 4}});
}

TEST(NDA, BLASGer) {
  test_ger<double, nda::C_layout>();
  test_ger<double, nda::F_layout>();
  test_ger<std::complex<double>, nda::C_layout>();
  test_ger<std::complex<double>, nda::C_layout>();
}

TEST(NDA, BLASOuterProduct) {
  auto N = nda::rand<double>(2, 3);
  auto M = nda::rand<double>(4, 5);

  nda::array<double, 4> P(2, 3, 4, 5);
  for (auto [i, j] : N.indices())
    for (auto [k, l] : M.indices()) P(i, j, k, l) = N(i, j) * M(k, l);

  EXPECT_ARRAY_NEAR(P, nda::blas::outer_product(N, M));
}

// Test the BLAS dot function and its generic implementation.
template <typename value_t>
void test_dot() {
  nda::vector<value_t> a{1, 2, 3, 4, 5};
  nda::vector<value_t> b{10, 20, 30, 40, 50};
  if constexpr (nda::is_complex_v<value_t>) {
    a *= 1 + 1i;
    b *= 1 + 2i;
  }

  EXPECT_COMPLEX_NEAR(nda::blas::dot(a, b), nda::blas::dot_generic(a, b), 1.e-14);
}

TEST(NDA, BLASDot) {
  test_dot<double>();
  test_dot<std::complex<double>>();
}

// Test the BLAS dotc function and its generic implementation.
template <typename value_t>
void test_dotc() {
  nda::vector<value_t> a{1, 2, 3, 4, 5};
  nda::vector<value_t> b{10, 20, 30, 40, 50};
  if constexpr (nda::is_complex_v<value_t>) {
    a *= 1 + 1i;
    b *= 1 + 2i;
  }

  EXPECT_COMPLEX_NEAR(nda::blas::dotc(a, b), nda::blas::dotc_generic(a, b), 1.e-14);
}

TEST(NDA, BLASDotc) {
  test_dotc<double>();
  test_dotc<std::complex<double>>();
}

// Test the BLAS scal function.
template <typename value_t>
void test_scal() {
  nda::vector<value_t> a{1, 2, 3, 4, 5};
  value_t x = 3.0;
  if constexpr (nda::is_complex_v<value_t>) {
    a *= 1 + 1i;
    x = 3.0 + 2.0i;
  }

  auto exp = nda::make_regular(x * a);
  nda::blas::scal(x, a);
  EXPECT_ARRAY_NEAR(a, exp);
}

TEST(NDA, BLASScal) {
  test_scal<double>();
  test_scal<std::complex<double>>();
}
