// Copyright (c) 2019-2023 Simons Foundation
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
// Authors: Dominik Kiese, Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <array>
#include <utility>

TEST(NDA, InverseOfTensorProductOfMatrices) {
  using namespace nda::clef::literals;

  // calculate the inverse of the tensor product of two matrices B and C
  // note that the inverse of the tensor product is the tensor product of the inverses
  nda::matrix<double> B(2, 2), C(3, 3), Binv, Cinv;
  C(i_, j_) << 1.7 / (3.4 * i_ - 2.3 * j_ + 1);
  B(i_, j_) << 2 * i_ + j_;
  Binv = inverse(B);
  Cinv = inverse(C);

  {
    nda::array<double, 4> A(2, 3, 2, 3);
    A(i_, j_, k_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, nda::idx_group<0, 1>, nda::idx_group<2, 3>));
    M      = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, j_, k_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }

  {
    nda::array<double, 4> A(2, 3, 2, 3);
    A(i_, j_, k_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, nda::idx_group<2, 3>, nda::idx_group<0, 1>));
    M      = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, j_, k_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }

  {
    nda::array<double, 4, nda::basic_layout<0, nda::encode(std::array{1, 0, 3, 2}), nda::layout_prop_e::contiguous>> A(2, 3, 2, 3);
    A(i_, j_, k_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, nda::idx_group<0, 1>, nda::idx_group<2, 3>));
    M      = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, j_, k_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }

  {
    nda::array<double, 4, nda::basic_layout<0, nda::encode(std::array{1, 0, 3, 2}), nda::layout_prop_e::contiguous>> A(2, 3, 2, 3);
    A(i_, j_, k_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, nda::idx_group<2, 3>, nda::idx_group<0, 1>));
    M      = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, j_, k_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }

  {
    nda::array<double, 4, nda::basic_layout<0, nda::encode(std::array{0, 2, 1, 3}), nda::layout_prop_e::contiguous>> A(2, 2, 3, 3);
    A(i_, k_, j_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, nda::idx_group<0, 2>, nda::idx_group<1, 3>));
    M      = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, k_, j_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }
}

TEST(NDA, MatrixVectorMultiplicationWithPermutedViews) {
  // generate some dummy data
  nda::array<double, 3> A = nda::rand(3, 3, 3);
  nda::vector<double> v   = nda::rand(3);

  // build permuted arrays
  auto B = nda::array<double, 3>{nda::permuted_indices_view<nda::encode(std::array<int, 3>{1, 2, 0})>(A)};
  auto C = make_regular(nda::permuted_indices_view<nda::encode(std::array<int, 3>{1, 2, 0})>(A));

  for (auto k : nda::range(3)) {
    auto Amat = nda::matrix<double>{A(nda::ellipsis{}, k)};
    auto Bmat = nda::matrix_view<double>{B(k, nda::ellipsis{})};
    auto Cmat = nda::matrix_view<double>{C(k, nda::ellipsis{})};
    EXPECT_EQ_ARRAY(Amat, Bmat);
    EXPECT_EQ_ARRAY(Amat * v, Bmat * v);
    EXPECT_DEBUG_DEATH(Cmat * v, "gemv");
  }
}

// Rotate the lengths and strides of the dimensions lower than N.
template <int N, typename A>
auto rotate_index_view(A &&a) {
  return permuted_indices_view<nda::encode(nda::permutations::cycle<nda::get_rank<A>>(-1, N + 1))>(std::forward<A>(a));
}

TEST(NDA, RotateIndexView) {
  // test special permuted_indices_view by rotating some of the indicies
  nda::array<long, 4> A(3, 4, 5, 6);
  for (int i = 0; i < A.extent(0); ++i)
    for (int j = 0; j < A.extent(1); ++j)
      for (int k = 0; k < A.extent(2); ++k)
        for (int l = 0; l < A.extent(3); ++l) A(i, j, k, l) = 1 + i + 10 * j + 100 * k + 1000 * l;

  auto A_r = rotate_index_view<2>(A);
  EXPECT_EQ(A_r.shape(), (std::array<long, 4>{5, 3, 4, 6}));

  auto it   = A.begin();
  auto it_r = A_r.begin();
  for (int i = 0; i < A.extent(0); ++i)
    for (int j = 0; j < A.extent(1); ++j)
      for (int k = 0; k < A.extent(2); ++k)
        for (int l = 0; l < A.extent(3); ++l) {
          EXPECT_EQ(A(i, j, k, l), *it++);
          EXPECT_EQ(A(i, j, k, l), A_r(k, i, j, l));
          EXPECT_EQ(A_r(k, i, j, l), *it_r++);
        }

  for (int i = 0; i < A_r.extent(0); ++i)
    for (int j = 0; j < A_r.extent(1); ++j)
      for (int k = 0; k < A_r.extent(2); ++k)
        for (int l = 0; l < A_r.extent(3); ++l) { EXPECT_EQ(A_r(i, j, k, l), A(j, k, i, l)); }
}

TEST(NDA, ArithmeticOperationsWithRand) {
  const auto size = 100;

  auto A = nda::rand<>(size, size);
  EXPECT_EQ(A.shape(), (std::array<long, 2>{size, size}));
  EXPECT_GE(nda::min_element(A), 0);
  EXPECT_LT(nda::max_element(A), 1);

  auto B = nda::make_regular(A + 3.0 * nda::rand<>(size, size) + 1.0);
  EXPECT_EQ(B.shape(), (std::array<long, 2>{size, size}));
  EXPECT_GE(nda::max_element(B), 1.0);
  EXPECT_LT(nda::max_element(B), 5.0);

  auto r = 2.0 + 3.0 * nda::rand<>();
  EXPECT_GE(r, 2.0);
  EXPECT_LT(r, 5.0);
}
