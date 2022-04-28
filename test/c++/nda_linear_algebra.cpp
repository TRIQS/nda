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

#include "test_common.hpp"
#include "nda/blas/gemm.hpp"
#include "nda/linalg/dot.hpp"
#include <nda/lapack.hpp>

#include <nda/linalg/det_and_inverse.hpp>
#include <nda/linalg/eigenelements.hpp>

using nda::C_layout;
using nda::F_layout;
using nda::matrix;
using nda::matrix_view;
using nda::range;
namespace blas = nda::blas;
using nda::dot;

// ==============================================================

TEST(Vector, Dot) { //NOLINT
  nda::array<double, 1> a(2), aa(2), c(2);
  a() = 2.0;
  c() = 1;
  nda::array<int, 1> b(2);
  b() = 3;
  aa  = 2 * a;

  EXPECT_DOUBLE_EQ(dot(a, b), 12);
  EXPECT_DOUBLE_EQ(dot(aa, a), 16);
  EXPECT_DOUBLE_EQ(dot(aa, b), 24);
  EXPECT_DOUBLE_EQ(dot(aa - a, b), 12);
}

// ==============================================================

TEST(Vector, Dot2) { //NOLINT

  /// Added by I. Krivenko, #122
  /// test the complex version, specially with the zdotu workaround on Os X.
  nda::array<std::complex<double>, 1> v(2);
  v(0) = 0;
  v(1) = {0, 1};

  EXPECT_COMPLEX_NEAR(nda::blas::dot(v, v), -1);
  EXPECT_COMPLEX_NEAR(nda::blas::dotc(v, v), 1);
}

// ==============================================================

template <typename T, typename L1, typename L2, typename L3>
void test_matmul() {
  matrix<T, L1> M1(2, 3);
  matrix<T, L2> M2(3, 4);
  matrix<T, L2> M3(2, 4), M3b(2, 4);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) { M1(i, j) = i + j; }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) { M2(i, j) = 1 + i - j; }

  M3  = 0;
  M3b = 0; // for MSAN
  if constexpr (nda::is_blas_lapack_v<T>) { blas::gemm(1, M1, M2, 0, M3b); }
  M3 = M1 * M2;

  auto M4 = M3;
  M4      = 0;
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 4; ++j) M4(i, j) += M1(i, k) * M2(k, j);

  EXPECT_ARRAY_NEAR(M4, M3, 1.e-13);
  if constexpr (nda::is_blas_lapack_v<T>) { EXPECT_ARRAY_NEAR(M4, M3b, 1.e-13); }
  // recheck gemm_generic
  blas::gemm_generic(1, M1, M2, 0, M4);
  EXPECT_ARRAY_NEAR(M4, M3, 1.e-13);
}

template <typename T>
void all_test_matmul() {
  test_matmul<T, C_layout, C_layout, C_layout>();
  test_matmul<T, C_layout, C_layout, F_layout>();
  test_matmul<T, C_layout, F_layout, F_layout>();
  test_matmul<T, C_layout, F_layout, C_layout>();
  test_matmul<T, F_layout, F_layout, F_layout>();
  test_matmul<T, F_layout, C_layout, F_layout>();
  test_matmul<T, F_layout, F_layout, C_layout>();
  test_matmul<T, F_layout, C_layout, C_layout>();
}

TEST(Matmul, Double) { // NOLINT
  all_test_matmul<double>();
}
TEST(Matmul, Complex) { // NOLINT
  all_test_matmul<std::complex<double>>();
}
TEST(Matmul, Int) { // NOLINT
  all_test_matmul<long>();
}

//-------------------------------------------------------------

TEST(Matmul, Promotion) { //NOLINT
  matrix<double> C, D, A = {{1.0, 2.3}, {3.1, 4.3}};
  matrix<int> B     = {{1, 2}, {3, 4}};
  matrix<double> Bd = {{1, 2}, {3, 4}};

  C = A * B;
  D = A * Bd;
  EXPECT_ARRAY_NEAR(A * B, A * Bd, 1.e-13);
}

//-------------------------------------------------------------

TEST(Matmul, Cache) { //NOLINT
  // testing with view for possible cache issue

  nda::array<std::complex<double>, 3> TMPALL(2, 2, 5);
  TMPALL() = -1;
  matrix_view<std::complex<double>> TMP(TMPALL(range(), range(), 2));
  matrix<std::complex<double>> M1(2, 2), Res(2, 2);
  M1()      = 0;
  M1(0, 0)  = 2;
  M1(1, 1)  = 3.2;
  Res()     = 0;
  Res(0, 0) = 8;
  Res(1, 1) = 16.64;
  TMP()     = 0;
  TMP()     = matrix<std::complex<double>>{M1 * (M1 + 2.0)};
  EXPECT_ARRAY_NEAR(TMP(), Res, 1.e-13);

  // not matmul, just recheck diagonal unity
  Res()     = 0;
  Res(0, 0) = 4;
  Res(1, 1) = 5.2;
  TMP()     = 0;
  TMP()     = matrix<std::complex<double>>{(M1 + 2.0)};
  EXPECT_ARRAY_NEAR(TMP(), Res, 1.e-13);
}

//-------------------------------------------------------------

TEST(Matmul, Alias) { //NOLINT

  nda::array<dcomplex, 3> A(10, 2, 2);
  A() = -1;

  A(4, _, _) = 1;
  A(5, _, _) = 2;

  matrix_view<dcomplex> M1 = A(4, _, _);
  matrix_view<dcomplex> M2 = A(5, _, _);

  M1 = M1 * M2;

  EXPECT_ARRAY_NEAR(M1, matrix<dcomplex>{{4, 4}, {4, 4}});
  EXPECT_ARRAY_NEAR(M2, matrix<dcomplex>{{2, 2}, {2, 2}});

  matrix<double> B1(2, 2), B2(2, 2);
  B1() = 2;
  B2() = 3;

  B1 = make_regular(B1) * B2;
  EXPECT_ARRAY_NEAR(B1, matrix<double>{{6, 0}, {0, 6}});
}

//-------------------------------------------------------------

TEST(Determinant, Fortran) { //NOLINT

  matrix<double, F_layout> W(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  EXPECT_NEAR(determinant(W), -7.8, 1.e-12);
}

//-------------------------------------------------------------

TEST(Determinant, C) { //NOLINT

  matrix<double> W(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  EXPECT_NEAR(determinant(W), -7.8, 1.e-12);

  auto W_sso = matrix<double, nda::C_layout, nda::sso<100>>{W};
  EXPECT_NEAR(determinant(W_sso), -7.8, 1.e-12);
}

//-------------------------------------------------------------

TEST(Inverse, F) { //NOLINT

  using matrix_t = matrix<double, F_layout>;

  matrix_t W(3, 3), Wi(3, 3), A;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  auto Wkeep = W;

  Wi = inverse(W);
  EXPECT_NEAR(determinant(Wi), -1 / 7.8, 1.e-12);

  matrix<double, F_layout> should_be_one(W * Wi);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(std::abs(should_be_one(i, j)), (i == j ? 1 : 0), 1.e-13);

  // FIXME MOVE THIS IN LAPACK TEST //NOLINT
  // testing against "manual" call of bindings
  nda::array<int, 1> ipiv2(3);
  ipiv2    = 0;
  int info = nda::lapack::getrf(Wi(), ipiv2);
  EXPECT_EQ(info, 0);
  info = nda::lapack::getri(Wi(), ipiv2);
  EXPECT_EQ(info, 0);
  EXPECT_ARRAY_NEAR(Wi, Wkeep, 1.e-12);
}

//-------------------------------------------------------------

TEST(Inverse, C) { //NOLINT

  using matrix_t = matrix<double, C_layout>;

  matrix_t W(3, 3), Wi(3, 3), A;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  auto Wkeep = W;

  Wi = inverse(W);
  EXPECT_NEAR(determinant(Wi), -1 / 7.8, 1.e-12);

  matrix<double, F_layout> should_be_one(W * Wi);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(std::abs(should_be_one(i, j)), (i == j ? 1 : 0), 1.e-13);

  // FIXME MOVE THIS IN LAPACK TEST //NOLINT
  // testing against "manual" call of bindings
  nda::array<int, 1> ipiv2(3);
  ipiv2    = 0;
  int info = nda::lapack::getrf(Wi(), ipiv2);
  EXPECT_EQ(info, 0);
  info = nda::lapack::getri(Wi, ipiv2);
  EXPECT_EQ(info, 0);
  EXPECT_ARRAY_NEAR(Wi, Wkeep, 1.e-12);
}

//-------------------------------------------------------------

TEST(Inverse, Involution) { //NOLINT

  using matrix_t = matrix<double, C_layout>;

  matrix_t W(3, 3), Wi(3, 3), A;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  auto Wkeep = W;

  W = inverse(W);
  W = inverse(W);
  EXPECT_ARRAY_NEAR(W, Wkeep, 1.e-12);
}

//-------------------------------------------------------------

TEST(Inverse, slice) { //NOLINT

  using matrix_t = matrix<double, C_layout>;

  matrix_t W(3, 3), Wi(3, 3), A;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) W(i, j) = (i > j ? i + 2.5 * j : i * 0.8 - j);

  {
    auto V      = W(range(0, 3, 2), range(0, 3, 2));
    matrix_t Vi = inverse(V);
    matrix_t Viref{{-0.1, 0.5}, {-0.5, 0.0}};
    EXPECT_ARRAY_NEAR(Vi, Viref, 1.e-12);
  }
  W = inverse(W);

  {
    auto V      = W(range(0, 3, 2), range(0, 3, 2));
    matrix_t Vi = inverse(V);
    matrix_t Viref{{-5.0, 4.0}, {24.5, -27.4}};
    EXPECT_ARRAY_NEAR(Vi, Viref, 1.e-12);
  }
}

// ==============================================================

TEST(Matvecmul, Promotion) { //NOLINT

  matrix<int> Ai   = {{1, 2}, {3, 4}};
  matrix<double> A = {{1, 2}, {3, 4}};
  nda::array<int, 1> Ci, B     = {1, 1};
  nda::array<double, 1> Cd, Bd = {1, 1};

  Cd = matvecmul(A, B);
  Ci = matvecmul(Ai, B);

  EXPECT_ARRAY_NEAR(Cd, Ci, 1.e-13);
}

// ================================================================================

template <typename M, typename V1, typename V2>
void check_eig(M const &m, V1 const &vectors, V2 const &values) {
  for (auto i : range(0, m.extent(0))) { EXPECT_ARRAY_NEAR(matvecmul(m, vectors(_, i)), values(i) * vectors(_, i), 1.e-13); }
}

//----------------------------------

TEST(eigenelements, test1) { //NOLINT

  auto test = [](auto &&M) {
    auto [ev, vecs] = nda::linalg::eigenelements(M);
    check_eig(M, vecs, ev);
  };

  {
    nda::matrix<double> A(3, 3);

    for (int i = 0; i < 3; ++i)
      for (int j = 0; j <= i; ++j) {
        A(i, j) = (i > j ? i + 2 * j : i - j);
        A(j, i) = A(i, j);
      }
    test(A);

    A()     = 0;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(2, 2) = 8;
    A(0, 2) = 2;
    A(2, 0) = 2;

    test(A);

    A()     = 0;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(2, 2) = 8;

    test(A);
  }

  { // the real case with fortran layout

    matrix<double, F_layout> D(2, 2);

    D(0, 0) = 1.3;
    D(0, 1) = 1.2;
    D(1, 0) = 1.2;
    D(1, 1) = 2.2;

    test(D);
  }

  { // the complex case

    matrix<dcomplex> B(2, 2);

    B(0, 0) = 1;
    B(0, 1) = 1.0i;
    B(1, 0) = -1.0i;
    B(1, 1) = 2;

    test(B);
  }

  { // the complex case with fortran layout

    matrix<dcomplex, F_layout> C(2, 2);

    C(0, 0) = 1.3;
    C(0, 1) = 1.1i;
    C(1, 0) = -1.1i;
    C(1, 1) = 2.4;

    test(C);
  }
}
