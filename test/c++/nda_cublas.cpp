// Copyright (c) 2022 Simons Foundation
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
// Authors: Miguel Morales, Nils Wentzell

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>
#include <vector>

// Test the CUBLAS gemm function.
template <typename value_t, typename Layout>
void test_gemm() {
  nda::matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3{{1, 0}, {0, 1}};
  nda::cumatrix<value_t, Layout> M1_d{M1}, M2_d{M2}, M3_d{M3};

  nda::blas::gemm(1.0, M1_d, M2_d, 1.0, M3_d);
  M3 = M3_d;

  EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});
}

TEST(NDA, CUBLASGemm) {
  test_gemm<double, nda::C_layout>();
  test_gemm<double, nda::F_layout>();
  test_gemm<std::complex<double>, nda::C_layout>();
  test_gemm<std::complex<double>, nda::F_layout>();
}

// Test the CUBLAS gemm_batch function.
template <typename value_t, typename Layout>
void test_gemm_batch() {
  int batch_count = 10;
  long size       = 64;

  auto vec_A_d = std::vector(batch_count, nda::to_device(nda::matrix<value_t, Layout>::rand({size, size})));
  auto vec_B_d = std::vector(batch_count, nda::to_device(nda::matrix<value_t, Layout>::rand({size, size})));
  auto vec_C_d = std::vector(batch_count, nda::to_device(nda::matrix<value_t, Layout>::zeros({size, size})));
  nda::blas::gemm_batch(1.0, vec_A_d, vec_B_d, 0.0, vec_C_d);

  for (auto i : nda::range(batch_count))
    EXPECT_ARRAY_NEAR(nda::make_regular(nda::to_host(vec_A_d[i]) * nda::to_host(vec_B_d[i])), nda::to_host(vec_C_d[i]));
}

TEST(NDA, CUBLASGemmBatch) {
  test_gemm_batch<double, nda::C_layout>();
  test_gemm_batch<double, nda::F_layout>();
  test_gemm_batch<std::complex<double>, nda::C_layout>();
  test_gemm_batch<std::complex<double>, nda::F_layout>();
}

#ifdef NDA_HAVE_MAGMA
template <typename value_t, typename Layout>
void test_gemm_vbatch() {
  int batch_count = 10;
  long size       = 64;

  auto vec_A_d = std::vector(batch_count, nda::to_device(nda::matrix<value_t, Layout>::rand({size, size})));
  auto vec_B_d = std::vector(batch_count, nda::to_device(nda::matrix<value_t, Layout>::rand({size, size})));
  auto vec_C_d = std::vector(batch_count, nda::to_device(nda::matrix<value_t, Layout>::zeros({size, size})));
  nda::blas::gemm_vbatch(1.0, vec_A_d, vec_B_d, 0.0, vec_C_d);

  for (auto i : nda::range(batch_count))
    EXPECT_ARRAY_NEAR(nda::make_regular(nda::to_host(vec_A_d[i]) * nda::to_host(vec_B_d[i])), nda::to_host(vec_C_d[i]));
}

TEST(NDA, CUBLASGemmVbatch) {
  test_gemm_vbatch<double, nda::C_layout>();
  test_gemm_vbatch<double, nda::F_layout>();
  test_gemm_vbatch<std::complex<double>, nda::C_layout>();
  test_gemm_vbatch<std::complex<double>, nda::F_layout>();
}
#endif

// Test the CUBLAS gemv function.
template <typename value_t, typename Layout>
void test_gemv() {
  using namespace nda::clef::literals;

  nda::matrix<value_t, Layout> A(5, 5);
  A(i_, j_) << i_ + 2 * j_ + 1;

  nda::vector<value_t> v(5), w(5);
  v() = 1;
  w() = 0;

  nda::cumatrix<value_t, Layout> A_d{A};
  nda::cuvector<value_t> v_d{v}, w_d{w};

  nda::range rg(1, 3);
  nda::blas::gemv(1, A_d(rg, rg), v_d(rg), 0, w_d(rg));
  w = w_d;
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{0, 10, 12, 0, 0});

  auto AT_d = nda::transpose(A_d);
  nda::blas::gemv(1, AT_d(rg, rg), v_d(rg), 0, w_d(rg));
  w = w_d;
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{0, 9, 13, 0, 0});

  // test operator*
  w_d(rg) = AT_d(rg, rg) * v_d(rg);
  w()     = -8;
  w(rg)   = w_d(rg);
  EXPECT_ARRAY_NEAR(w, nda::vector<value_t>{-8, 9, 13, -8, -8});
}

TEST(NDA, CUBLASGemv) {
  test_gemv<double, nda::C_layout>();
  test_gemv<double, nda::F_layout>();
  test_gemv<std::complex<double>, nda::C_layout>();
  test_gemv<std::complex<double>, nda::F_layout>();
}

// Test the CUBLAS ger function.
template <typename value_t, typename Layout>
void test_ger() {
  nda::matrix<value_t, Layout> M(2, 2);
  M = 0;
  nda::array<value_t, 1> v{1, 2};

  nda::cumatrix<value_t, Layout> M_d{M};
  nda::cuvector<value_t> v_d{v};

  nda::blas::ger(1.0, v_d, v_d, M_d);

  M = M_d;
  EXPECT_ARRAY_NEAR(M, nda::matrix<value_t>{{1, 2}, {2, 4}});
}

TEST(NDA, CUBLASGer) {
  test_ger<double, nda::C_layout>();
  test_ger<double, nda::F_layout>();
  test_ger<std::complex<double>, nda::C_layout>();
  test_ger<std::complex<double>, nda::C_layout>();
}

TEST(NDA, CUBLASOuterProduct) {
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

// Test the CUBLAS dot function.
template <typename value_t>
void test_dot() {
  nda::vector<value_t> a{1, 2, 3, 4, 5};
  nda::vector<value_t> b{10, 20, 30, 40, 50};
  if constexpr (nda::is_complex_v<value_t>) {
    a *= 1 + 1i;
    b *= 1 + 2i;
  }

  nda::cuvector<value_t> a_d{a}, b_d{b};
  EXPECT_COMPLEX_NEAR((nda::blas::dot(a_d, b_d)), (nda::blas::dot_generic(a, b)), 1.e-14);
}

TEST(NDA, CUBLASDot) {
  test_dot<double>();
  test_dot<std::complex<double>>();
}

// Test the CUBLAS dotc function.
template <typename value_t>
void test_dotc() {
  nda::vector<value_t> a{1, 2, 3, 4, 5};
  nda::vector<value_t> b{10, 20, 30, 40, 50};
  if constexpr (nda::is_complex_v<value_t>) {
    a *= 1 + 1i;
    b *= 1 + 2i;
  }

  nda::cuvector<value_t> a_d{a}, b_d{b};
  EXPECT_COMPLEX_NEAR((nda::blas::dotc(a_d, b_d)), (nda::blas::dotc_generic(a, b)), 1.e-14);
}

TEST(NDA, CUBLASDotc) {
  test_dotc<double>();
  test_dotc<std::complex<double>>();
}
