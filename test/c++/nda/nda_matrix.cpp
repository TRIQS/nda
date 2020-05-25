#include "./test_common.hpp"

TEST(Matrix, Create1) { //NOLINT
  nda::matrix<long> A(3, 3);
  EXPECT_EQ(A.shape(), (nda::shape_t<2>{3, 3}));

  std::cerr << A.indexmap() << std::endl;
}

// ===============================================================

TEST(Matrix, Transpose) { //NOLINT

  const int N = 5;

  nda::matrix<double, F_layout> A(N, N);
  nda::matrix<std::complex<double>> B(N, N);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      A(i, j) = i + 2 * j + 1;
      B(i, j) = i + 2.5 * j + (i - 0.8 * j) * 1i;
    }

  auto at = transpose(A);
  auto bt = transpose(B);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      EXPECT_COMPLEX_NEAR(at(i, j), A(j, i));
      EXPECT_COMPLEX_NEAR(bt(i, j), B(j, i));
    }
}
// ===============================================================

TEST(Matrix, Dagger) { //NOLINT

  const int N = 5;

  nda::matrix<double, F_layout> A(N, N);
  nda::matrix<std::complex<double>> B(N, N);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      A(i, j) = i + 2 * j + 1;
      B(i, j) = i + 2.5 * j + (i - 0.8 * j) * 1i;
    }

  auto ad = dagger(A);
  auto bd = dagger(B);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      EXPECT_COMPLEX_NEAR(ad(i, j), A(j, i));
      EXPECT_COMPLEX_NEAR(bd(i, j), std::conj(B(j, i)));
    }
}

// ===============================================================

TEST(Matrix, Eye) { //NOLINT

  EXPECT_EQ_ARRAY( nda::eye<long>(3), (nda::matrix<long>{{1,0,0},  { 0,1,0}, {0,0,1}}));
};
