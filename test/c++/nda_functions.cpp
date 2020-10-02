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

#include "test_common.hpp"
#include <nda/linalg/det_and_inverse.hpp>

clef::placeholder<0> i_;
clef::placeholder<1> j_;
clef::placeholder<2> k_;
clef::placeholder<3> l_;
using nda::encode;
using nda::idx_group;

// ------------------------------------

TEST(matrix, vstack) { //NOLINT

  matrix<long> A1 = {{1, 2}, {3, 4}};
  matrix<long> A2 = {{5, 6}, {7, 8}};

  auto B = nda::vstack(A1, A2);

  EXPECT_EQ(B(2, 0), 5);
  EXPECT_EQ(B(3, 1), 8);

  EXPECT_EQ(B(range(0, 2), range(0, 2)), A1);
  EXPECT_EQ(B(range(2, 4), range(0, 2)), A2);

  EXPECT_EQ(B.shape(), (std::array<long, 2>{4, 2}));
}
//================================================

TEST(reshape, array) { //NOLINT

  nda::array<long, 1> a{1, 2, 3, 4, 5, 6};
  nda::array<long, 2> check{{1, 2}, {3, 4}, {5, 6}};

  // should not compile
  // auto b = reshape(a, std::array{2,3});
  auto b = reshape(std::move(a), std::array{3, 2});

  EXPECT_EQ(b, check);
}

//================================================

TEST(reshaped_view, array) { //NOLINT

  nda::array<long, 1> a{1, 2, 3, 4, 5, 6};
  nda::array<long, 2> check{{1, 2}, {3, 4}, {5, 6}};

  auto v = reshaped_view(a, std::array{3, 2});
  EXPECT_EQ(v, check);

  auto v2 = reshaped_view(a(), std::array{3, 2});
  EXPECT_EQ(v2, check);
}
//================================================

TEST(reshaped_view, checkView) { //NOLINT

  nda::array<long, 1> a{1, 2, 3, 4, 5, 6};     // 1d array
  auto v = reshaped_view(a, std::array{2, 3}); // v is an array_view<long,2> of size 2 x 3
  v(0, nda::range::all) *= 10;                 // a is now {10, 20, 30, 4, 5, 6}

  EXPECT_EQ_ARRAY(a, (nda::array<long, 1>{10, 20, 30, 4, 5, 6}));
}
//================================================

TEST(GroupIndices, check) { //NOLINT
  nda::check_grouping(nda::permutations::identity<4>(), std::array{0, 1}, std::array{2, 3});
}

TEST(GroupIndices, v1) { //NOLINT
  nda::array<int, 4> A(2, 2, 2, 2);
  A(i_, j_, k_, l_) << i_ + 10 * j_ + 100 * k_ + 1000 * l_;
  nda::group_indices_view(A(), nda::idx_group<0, 1>, nda::idx_group<2, 3>);
}

// ------------------------------------

TEST(GroupIndices, ProdInverse) { //NOLINT

  // more complex : inversing a tensor product of matrices...
  nda::matrix<double> B(2, 2), C(3, 3), Binv, Cinv;
  C(i_, j_) << 1.7 / (3.4 * i_ - 2.3 * j_ + 1);
  B(i_, j_) << 2 * i_ + j_;
  Binv = inverse(B);
  Cinv = inverse(C);

  {
    nda::array<double, 4> A(2, 3, 2, 3);
    A(i_, j_, k_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, idx_group<0, 1>, idx_group<2, 3>));
    NDA_PRINT(M.indexmap());
    M = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, j_, k_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }

  {
    nda::array<double, 4> A(2, 3, 2, 3);
    A(i_, j_, k_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, idx_group<2, 3>, idx_group<0, 1>));
    NDA_PRINT(M.indexmap());
    M = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, j_, k_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }

  {
    nda::array<double, 4, nda::basic_layout<0, encode(std::array{1, 0, 3, 2}), nda::layout_prop_e::contiguous>> A(2, 3, 2, 3);
    A(i_, j_, k_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, idx_group<0, 1>, idx_group<2, 3>));
    NDA_PRINT(M.indexmap());
    M = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, j_, k_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }

  {
    nda::array<double, 4, nda::basic_layout<0, encode(std::array{1, 0, 3, 2}), nda::layout_prop_e::contiguous>> A(2, 3, 2, 3);
    A(i_, j_, k_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, idx_group<2, 3>, idx_group<0, 1>));
    NDA_PRINT(M.indexmap());
    M = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, j_, k_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }

  {
    nda::array<double, 4, nda::basic_layout<0, encode(std::array{0, 2, 1, 3}), nda::layout_prop_e::contiguous>> A(2, 2, 3, 3);
    A(i_, k_, j_, l_) << B(i_, k_) * C(j_, l_);
    auto M = make_matrix_view(group_indices_view(A, idx_group<0, 2>, idx_group<1, 3>));
    NDA_PRINT(M.indexmap());
    M = inverse(M);
    nda::array<double, 4> R(A.shape());
    R(i_, k_, j_, l_) << Binv(i_, k_) * Cinv(j_, l_);
    EXPECT_ARRAY_NEAR(R, A, 5.e-15);
  }
}

//================================================

TEST(Array, SwapIndex) { //NOLINT

  nda::array<long, 4> A(1, 2, 3, 4);

  A(i_, j_, k_, l_) << i_ + 10 * j_ + 100 * k_ + 1000 * l_;

  auto S = nda::transposed_view<0, 2>(A);

  nda::array<long, 4> B(3, 2, 1, 4);
  B(k_, j_, i_, l_) << i_ + 10 * j_ + 100 * k_ + 1000 * l_;

  EXPECT_EQ(S, B());
  EXPECT_EQ(S.shape(), B.shape());
}
