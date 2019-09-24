#include "test_common.hpp"

#include <nda/blas/gemm.hpp>
#include <nda/blas/gemv.hpp>
#include <nda/blas/ger.hpp>
#include <nda/blas/dot.hpp>

using nda::F_layout;

//----------------------------

TEST(BLAS, gemm) {

  nda::matrix<double> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3(2, 2);
  M3 = 0;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<double>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<double>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<double>{{1, 1}, {3, 3}});
}

//----------------------------
TEST(BLAS, gemmF) {

  nda::matrix<double, F_layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3(2, 2);
  M3 = 0;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);

  nda::array<double, 2> M3copy{M3};

  EXPECT_ARRAY_NEAR(M1, nda::matrix<double>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<double>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<double>{{1, 1}, {3, 3}});
}

//----------------------------
TEST(BLAS, zgemm) {

  nda::matrix<dcomplex> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3(2, 2);
  M3 = 0;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<dcomplex>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<dcomplex>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<dcomplex>{{1, 1}, {3, 3}});
}

//----------------------------
TEST(BLAS, gemmCF) {
  nda::matrix<dcomplex, F_layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3(2, 2);
  M3 = 0;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<dcomplex>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<dcomplex>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<dcomplex>{{1, 1}, {3, 3}});
}

// ==============================================================

TEST(BLAS, gemv) {

  nda::matrix<double, F_layout> A(5, 5), Ac(5, 5);
  nda::array<double, 1> MC(5), MB(5);

  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j) A(i, j) = i + 2 * j + 1;

  Ac = A;

  MC() = 1;
  MB() = 0;
  nda::range R(1, 3);

  // FIXME IMPLEMENT
  //matrix_view<double> Acw = A.transpose();

  auto MB_w = MB(R); // view !

  nda::blas::gemv(1, A(R, R), MC(R), 0, MB_w);
  EXPECT_ARRAY_NEAR(MB, nda::array<double, 1>{0, 10, 12, 0, 0});

  nda::blas::gemv(1, Ac(R, R), MC(R), 0, MB_w);
  EXPECT_ARRAY_NEAR(MB, nda::array<double, 1>{0, 10, 12, 0, 0});

  //blas::gemv(1, Acw(R, R), MC(R), 0, MB_w);
  //EXPECT_ARRAY_NEAR(MB, vector<double>{0, 9, 13, 0, 0});
}

//----------------------------
TEST(BLAS, ger) {

  nda::matrix<double, F_layout> M(2, 2);
  M = 0;
  nda::array<double, 1> V{1, 2};

  nda::blas::ger(1.0, V, V, M);
  EXPECT_ARRAY_NEAR(M, nda::matrix<double>{{1, 2}, {2, 4}});
}

//----------------------------
TEST(BLAS, dot) {

  nda::array<double, 1> a{1, 2, 3, 4, 5};
  nda::array<double, 1> b{10, 20, 30, 40, 50};

  EXPECT_NEAR((nda::blas::dot(a, b)), (10 + 2 * 20 + 3 * 30 + 4 * 40 + 5 * 50), 1.e-14);
}

//----------------------------
TEST(BLAS, dotc1) {

  nda::array<dcomplex, 1> a{1, 2, 3};
  nda::array<dcomplex, 1> b{10, 20, 30};
  a *= 1 + 1i;
  b *= 1 + 2i;

  EXPECT_COMPLEX_NEAR((nda::blas::dotc(a, b)), (1 - 1i) * (1 + 2i) * (10 + 2 * 20 + 3 * 30), 1.e-14);
}

//----------------------------
TEST(BLAS, dotc2) {

  nda::array<double, 1> a{1, 2, 3, 4, 5};
  nda::array<double, 1> b{10, 20, 30, 40, 50};

  EXPECT_COMPLEX_NEAR((nda::blas::dotc(a, b)), (10 + 2 * 20 + 3 * 30 + 4 * 40 + 5 * 50), 1.e-14);
}

/*
//----------------------------

TEST(BLAS, Blas3InvMat) {
  placeholder<0> i_;
  placeholder<1> j_;

  nda::matrix<double> M(2, 2);
  M(i_, j_) << i_ + j_;
  vector<int> ipiv(2);
  lapack::getrf(M, ipiv);
  lapack::getri(M, ipiv);
  EXPECT_ARRAY_NEAR(M, nda::matrix<double>{{-2, 1}, {1, 0}});
}

// ==================DEEP SWAP for vectors============================================

// FIXME Rename as BLAS_SWAP (swap of blas). Only for vector of same size
TEST(BLAS, DeepSwap) {
  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4};

  deep_swap(V, W);

  // V , W are swapped
  EXPECT_EQ(V, (vector<double>{4, 4, 4}));
  EXPECT_EQ(W, (vector<double>{3, 3, 3}));
}
// ----------------------------------

TEST(NDA, DeepSwapView) {
  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4, 4};

  auto VV = V(range(0, 2));
  auto WW = W(range(0, 2));

  deep_swap(VV, WW);

  // VV, WW swapped
  EXPECT_EQ(WW, (vector<double>{3, 3}));
  EXPECT_EQ(VV, (vector<double>{4, 4}));
}
*/
