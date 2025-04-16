// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <array>
#include <complex>
#include <utility>

TEST(NDA, ConstructLongMatrices) {
  nda::matrix<long> M1(3, 3);
  EXPECT_EQ(M1.shape(), (std::array<long, 2>{3, 3}));

  // no ambiguity with initializer list
  nda::matrix<long> M2{3, 3};
  EXPECT_EQ(M2.shape(), (std::array<long, 2>{3, 3}));

  // uses initializer list constructor
  nda::matrix<long> M3{{3, 3}};
  EXPECT_EQ(M3.shape(), (std::array<long, 2>{1, 2}));
}

TEST(NDA, ConstructDoubleMatrices) {
  auto M1 = nda::matrix<double>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};
  EXPECT_EQ(M1.shape(), (std::array<long, 2>{5, 2}));

  auto M2 = nda::matrix<double>{{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}}};
  EXPECT_EQ(M2.shape(), (std::array<long, 2>{5, 2}));
}

TEST(NDA, ConstructComplexMatrices) {
  auto M1 = nda::matrix<std::complex<double>>{{-10, -3i}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};
  EXPECT_EQ(M1.shape(), (std::array<long, 2>{5, 2}));

  auto M2 = nda::matrix<std::complex<double>>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};
  EXPECT_EQ(M2.shape(), (std::array<long, 2>{5, 2}));

  auto M3 = nda::matrix<std::complex<double>>{{{-10, -3i}, {12, 14}, {14, 12}, {16, 16}, {18, 16}}};
  EXPECT_EQ(M3.shape(), (std::array<long, 2>{5, 2}));

  // constructs a 1x5 matrix
  auto M4 = nda::matrix<std::complex<double>>{{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}}};
  EXPECT_EQ(M4.shape(), (std::array<long, 2>{1, 5}));
}

TEST(NDA, MoveMatrixIntoArray) {
  auto M      = nda::matrix<double>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};
  auto M_copy = M;
  auto A      = nda::array<double, 2>{std::move(M)};
  EXPECT_EQ_ARRAY(M_copy, nda::matrix_view<double>{A});

  // only on heap, it would not be correct for the other cases
#if !defined(NDA_TEST_DEFAULT_ALLOC_SSO) and !defined(NDA_TEST_DEFAULT_ALLOC_MBUCKET)
  EXPECT_EQ(M.storage().size(), 0);
  EXPECT_TRUE(M.storage().is_null());
#endif
}

TEST(NDA, MovaArrayIntoMatrix) {
  auto A      = nda::array<double, 2>{{-10, -3}, {12, 14}, {14, 12}, {16, 16}, {18, 16}};
  auto A_copy = A;
  auto M      = nda::matrix<double>{std::move(A)};
  EXPECT_EQ_ARRAY(M, nda::matrix_view<double>{A_copy});

  // only on heap, it would not be correct for the other cases
#if !defined(NDA_TEST_DEFAULT_ALLOC_SSO) and !defined(NDA_TEST_DEFAULT_ALLOC_MBUCKET)
  EXPECT_EQ(A.storage().size(), 0);
  EXPECT_TRUE(A.storage().is_null());
#endif
}

TEST(NDA, MatrixTranspose) {
  const int size = 5;
  nda::matrix<double, nda::F_layout> M_d(size, size);
  nda::matrix<std::complex<double>> M_c(size, size);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      M_d(i, j) = i + 2 * j + 1;
      M_c(i, j) = i + 2.5 * j + (i - 0.8 * j) * 1i;
    }
  }

  auto M_d_T = nda::transpose(M_d);
  auto M_c_T = nda::transpose(M_c);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      EXPECT_COMPLEX_NEAR(M_d_T(i, j), M_d(j, i));
      EXPECT_COMPLEX_NEAR(M_c_T(i, j), M_c(j, i));
    }
  }
}

TEST(NDA, MatrixDagger) {
  const int size = 5;
  nda::matrix<double, nda::F_layout> M_d(size, size);
  nda::matrix<std::complex<double>> M_c(size, size);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      M_d(i, j) = i + 2 * j + 1;
      M_c(i, j) = i + 2.5 * j + (i - 0.8 * j) * 1i;
    }
  }

  auto M_d_dag = nda::dagger(M_d);
  auto M_c_dag = nda::dagger(M_c);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      EXPECT_COMPLEX_NEAR(M_d_dag(i, j), M_d(j, i));
      EXPECT_COMPLEX_NEAR(M_c_dag(i, j), std::conj(M_c(j, i)));
    }
  }
}

TEST(NDA, MatrixSliceDagger) {
  const int size = 5, slice_size = 2;

  nda::matrix<std::complex<double>> M(size, size);
  nda::matrix<std::complex<double>> exp_slice(slice_size, slice_size);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) { M(i, j) = i + 2.5 * j + (i - 0.8 * j) * 1i; }
  }

  for (int i = 0; i < slice_size; ++i) {
    for (int j = 0; j < slice_size; ++j) { exp_slice(j, i) = std::conj(M(i, j)); }
  }

  auto M_conj = nda::conj(nda::matrix_view<std::complex<double>>{M});
  EXPECT_EQ(M_conj.shape(), (std::array<long, 2>{size, size}));

  auto M_slice_dag = nda::dagger(M)(nda::range(0, 2), nda::range(0, 2));
  EXPECT_EQ(M_slice_dag.shape(), (std::array<long, 2>{slice_size, slice_size}));

  EXPECT_ARRAY_NEAR(M_slice_dag, exp_slice, 1.e-14);
}

TEST(NDA, IdentityMatrix) { EXPECT_EQ_ARRAY(nda::eye<long>(3), (nda::matrix<long>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})); }

TEST(NDA, DiagonalMatrix) {
  auto v = nda::vector<int>{1, 2, 3};
  auto M = nda::diag(v);
  EXPECT_EQ_ARRAY(M, (nda::matrix<int>{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}));
  EXPECT_EQ_ARRAY(nda::diagonal(M), v);

  nda::diagonal(M) += v;
  EXPECT_EQ_ARRAY(nda::diagonal(M), 2 * v);
}

TEST(NDA, MatrixSlice) {
  nda::matrix<double> M(10, 10);
  auto v = M(nda::range(2, 4), 7);
  static_assert(decltype(v)::layout_t::layout_prop == nda::layout_prop_e::strided_1d);
  static_assert(nda::has_contiguous(decltype(v)::layout_t::layout_prop) == false);
}

TEST(NDA, MatrixAlgebra) {
  auto M1   = nda::matrix<double>{{1, 2}, {3, 4}};
  auto M2   = nda::matrix<double>{{1, 2}, {2, 1}};
  auto prod = nda::matrix<double>{{5, 4}, {11, 10}};
  EXPECT_EQ(prod, nda::make_regular(M1 * M2));
  EXPECT_EQ(nda::make_regular(M1 / M2), nda::make_regular(M1 * nda::inverse(M2)));
}
