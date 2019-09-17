#include "test_common.hpp"

#include <nda/blas/gemm.hpp>
//#include <nda/blas/gemv.hpp>
//#include <nda/blas/ger.hpp>

//----------------------------

TEST(NDA, Blas3R) {

  nda::matrix<double> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3(2, 2);
  M3 = 0;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<double>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<double>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<double>{{1, 1}, {3, 3}});
}

//----------------------------
TEST(NDA, Blas3R_f) {

  nda::matrix<double, nda::F_layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3(2, 2);
  M3 = 0;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);
  NDA_PRINT(M3);

  auto mcheck = nda::matrix<double>{{1, 1}, {3, 3}};
  NDA_PRINT(mcheck);
  NDA_PRINT(mcheck(1, 0));
  NDA_PRINT((M3 - mcheck)(1, 0));
  auto max_diff = max_element(abs(M3 - mcheck));
  NDA_PRINT(max_diff);
  EXPECT_ARRAY_NEAR(M3, mcheck);

  nda::array<double, 2> M3copy{M3};
  NDA_PRINT(M3copy);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<double>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<double>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<double>{{1, 1}, {3, 3}});
}

//----------------------------
TEST(NDA, Blas3C) {

  nda::matrix<dcomplex> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3(2, 2);
  M3 = 0;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<dcomplex>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<dcomplex>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<dcomplex>{{1, 1}, {3, 3}});
}

//----------------------------
TEST(NDA, Blas3Cf) {
  nda::matrix<dcomplex, nda::F_layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3(2, 2);
  M3 = 0;

  nda::blas::gemm(1.0, M1, M2, 1.0, M3);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<dcomplex>{{0, 1}, {1, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<dcomplex>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M3, nda::matrix<dcomplex>{{1, 1}, {3, 3}});
}

/*
//----------------------------
TEST(NDA, Blas2) {
  placeholder<0> i_;
  nda::matrix<double> M(2, 2, FORTRAN_LAYOUT);
  M() = 0;
  vector<double> V(2);
  V[i_] << i_ + 1;

  nda::blas::ger(1.0, V, V, M);
  EXPECT_ARRAY_NEAR(M, nda::matrix<double>{{1, 2}, {2, 4}});
}

//----------------------------

TEST(NDA, Blas3InvMat) {
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
TEST(NDA, DeepSwap) {
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
