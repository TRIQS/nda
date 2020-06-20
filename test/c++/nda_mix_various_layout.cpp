#include "./test_common.hpp"

// ==============================================================

TEST(FortranC, Assign) { //NOLINT
  nda::array<long, 2, F_layout> Af(2, 3);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) Af(i, j) = 10 * i + j;

  // copy to C layout
  nda::array<long, 2> B(Af);

  // assign
  nda::array<long, 2> A;
  A = Af;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(A(i, j), 10 * i + j);
      EXPECT_EQ(B(i, j), 10 * i + j);
    }
}

// ===============================================================

TEST(Fortran, ScalarAssign) { //NOLINT

  int N = 5;
  matrix<int, F_layout> a(N, N);
  a() = 2;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) EXPECT_EQ(a(i, j), (i == j ? 2 : 0));

  nda::array_view<int, 2, F_layout> aa(a);
  aa = 2;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) EXPECT_EQ(a(i, j), 2);
}
