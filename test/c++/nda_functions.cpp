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

#include "test_common.hpp"
#include <nda/linalg/det_and_inverse.hpp>

clef::placeholder<0> i_;
clef::placeholder<1> j_;
clef::placeholder<2> k_;
clef::placeholder<3> l_;
using nda::encode;
using nda::idx_group;
using nda::matrix_view;

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

TEST(reshape, array_rval) { //NOLINT

  nda::array<long, 1> a{1, 2, 3, 4, 5, 6};
  nda::array<long, 2> check{{1, 2}, {3, 4}, {5, 6}};

  auto b = reshape(nda::basic_array{a}, std::array{3, 2});
  EXPECT_EQ(b, check);
  static_assert(nda::is_regular_v<decltype(b)>);

  auto c = reshape(nda::basic_array{a}, 3, 2);
  EXPECT_EQ(c, check);
  static_assert(nda::is_regular_v<decltype(c)>);

  EXPECT_EQ(a, flatten(a));
  EXPECT_EQ(a, flatten(b));
  EXPECT_EQ(a, flatten(c));
}

//================================================

TEST(reshape, array) { //NOLINT

  nda::array<long, 1> a{1, 2, 3, 4, 5, 6};
  nda::array<long, 2> check{{1, 2}, {3, 4}, {5, 6}};

  auto v = reshape(a, std::array{3, 2});
  EXPECT_EQ(v, check);
  static_assert(nda::is_view_v<decltype(v)>);

  auto v2 = reshape(a(), 3, 2);
  EXPECT_EQ(v2, check);
  static_assert(nda::is_view_v<decltype(v2)>);
}
//================================================

TEST(reshape, checkView) { //NOLINT

  nda::array<long, 1> a{1, 2, 3, 4, 5, 6}; // 1d array
  auto v = reshape(a, std::array{2, 3});   // v is an array_view<long,2> of size 2 x 3
  v(0, nda::range::all) *= 10;             // a is now {10, 20, 30, 4, 5, 6}

  EXPECT_EQ_ARRAY(a, (nda::array<long, 1>{10, 20, 30, 4, 5, 6}));
}
//================================================

TEST(GroupIndices, check) { //NOLINT
  nda::details::is_partition_of_indices<4>(std::array{0, 1}, std::array{2, 3});
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

// ----------------------------------------------

TEST(Array, Permuted_view1) { //NOLINT

  // first check with all same length
  nda::array<long, 4> A(3, 3, 3, 3);

  A(i_, j_, k_, l_) << (1 + i_) + 10 * (1 + j_) + 100 * (1 + k_) + 1000 * (1 + l_);

  // permutation (1 2 0 3)
  auto S = nda::permuted_indices_view<nda::encode(std::array<int, 4>{1, 2, 0, 3})>(A);

  for (int i0 = 0; i0 < 3; ++i0)
    for (int i1 = 0; i1 < 3; ++i1)
      for (int i2 = 0; i2 < 3; ++i2)
        for (int i3 = 0; i3 < 3; ++i3) { EXPECT_EQ((S(i0, i1, i2, i3)), (A(i1, i2, i0, i3))); }
}

// ----------------------------------------------

TEST(Array, Permuted_view_matmul) { //NOLINT

  // generate some dummy data
  nda::array<double, 3> A = nda::rand(3, 3, 3);
  nda::vector<double> v   = nda::rand(3);

  // build permuted array
  auto B = nda::array<double, 3>{nda::permuted_indices_view<nda::encode(std::array<int, 3>{1, 2, 0})>(A)};
  auto C = make_regular(nda::permuted_indices_view<nda::encode(std::array<int, 3>{1, 2, 0})>(A));

  for (auto k : range(3)) {
    auto Amat = matrix<double>{A(_, _, k)};
    auto Bmat = matrix_view<double>{B(k, _, _)};
    auto Cmat = matrix_view<double>{C(k, _, _)};
    EXPECT_EQ_ARRAY(Amat, Bmat);
    EXPECT_EQ_ARRAY(Amat * v, Bmat * v);
    EXPECT_DEBUG_DEATH(Cmat * v, "gemv");
  }
}

// ----------------------------------------------

TEST(Array, Permuted_view) { //NOLINT

  nda::array<long, 4> A(1, 2, 3, 4);

  A(i_, j_, k_, l_) << i_ + 10 * j_ + 100 * k_ + 1000 * l_;

  // permutation (1 2 0 3)
  auto S = nda::permuted_indices_view<nda::encode(std::array<int, 4>{1, 2, 0, 3})>(A);

  nda::array<long, 4> B(3, 1, 2, 4);
  B(i_, j_, k_, l_) << j_ + 10 * k_ + 100 * i_ + 1000 * l_;

  EXPECT_EQ(S.shape(), B.shape());
  EXPECT_EQ(S, B());
  EXPECT_TRUE(S.indexmap().is_stride_order_valid());

  // reversing the permutation
  {
    auto Sinv = nda::permuted_indices_view<nda::encode(std::array<int, 4>{2, 0, 1, 3})>(B);

    EXPECT_EQ(Sinv.shape(), A.shape());
    EXPECT_EQ(Sinv, A());
    EXPECT_TRUE(Sinv.indexmap().is_stride_order_valid());
  }
  // permutation composition
  //  (0 3 1 2)  (1 2 0 3)   = (3 1 0 2)
  {
    EXPECT_EQ((nda::permutations::compose(std::array<int, 4>{0, 3, 1, 2}, std::array<int, 4>{1, 2, 0, 3})), (std::array<int, 4>{3, 1, 0, 2}));
    auto S2    = nda::permuted_indices_view<nda::encode(std::array<int, 4>{0, 3, 1, 2})>(S);
    auto Scomp = nda::permuted_indices_view<nda::encode(std::array<int, 4>{3, 1, 0, 2})>(A);

    EXPECT_EQ(S2.shape(), Scomp.shape());
    EXPECT_EQ(S2, Scomp);
    EXPECT_TRUE(S2.indexmap().is_stride_order_valid());
    EXPECT_TRUE(Scomp.indexmap().is_stride_order_valid());
  }
}

// ------------------------------------------
// another composition
// Ideally, we would need to test all permutations ??
// chosen so that the 2 perm. do not commute

TEST(Array, Permuted_view2) { //NOLINT

  nda::array<long, 4> A(1, 2, 3, 4);

  A(i_, j_, k_, l_) << i_ + 10 * j_ + 100 * k_ + 1000 * l_;

  // (1 2 0 3)
  auto S = nda::permuted_indices_view<nda::encode(std::array<int, 4>{1, 2, 0, 3})>(A);

  nda::array<long, 4> B(3, 1, 2, 4);
  B(i_, j_, k_, l_) << j_ + 10 * k_ + 100 * i_ + 1000 * l_;

  EXPECT_EQ(S.shape(), B.shape());
  EXPECT_EQ(S, B());
  EXPECT_TRUE(S.indexmap().is_stride_order_valid());

  EXPECT_EQ((nda::permutations::compose(std::array<int, 4>{0, 1, 3, 2}, std::array<int, 4>{1, 2, 0, 3})), (std::array<int, 4>{1, 3, 0, 2}));
  EXPECT_EQ((nda::permutations::compose(std::array<int, 4>{1, 2, 0, 3}, std::array<int, 4>{0, 1, 3, 2})), (std::array<int, 4>{1, 2, 3, 0}));

  auto S2    = nda::permuted_indices_view<nda::encode(std::array<int, 4>{0, 1, 3, 2})>(S);
  auto Scomp = nda::permuted_indices_view<nda::encode(std::array<int, 4>{1, 3, 0, 2})>(A);

  EXPECT_EQ(S2.shape(), Scomp.shape());
  EXPECT_EQ(S2, Scomp);
  EXPECT_TRUE(S2.indexmap().is_stride_order_valid());
  EXPECT_TRUE(Scomp.indexmap().is_stride_order_valid());
}
