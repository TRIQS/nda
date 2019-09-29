#include "./test_common.hpp"

TEST(NDA, compound_ops) {

  nda::array<long, 2> A(2, 3);

  for (long i = 0; i < 2; ++i)
    for (long j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  auto A2 = A;

  A *= 2.0;
  EXPECT_EQ(A, (nda::array<long, 2>{{0, 2, 4}, {20, 22, 24}}));

  A2 /= 2.0;
  EXPECT_EQ(A2, (nda::array<long, 2>{{0, 0, 1}, {5, 5, 6}}));

  nda::array<double, 2> B(A);
  B /= 4;
  EXPECT_ARRAY_NEAR(B, (nda::array<double, 2>{{0.0, 0.5, 1.0}, {5.0, 5.5, 6.0}}));
}

// ==============================================================

TEST(Vector, Ops) {

  nda::array<double, 1> V{1, 2, 3, 4, 5};

  V *= 2;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{2, 4, 6, 8, 10}));

  V[range(2, 4)] /= 2.0;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{2, 4, 3, 4, 10}));

  V[range(0, 5, 2)] *= 10;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{20, 4, 30, 4, 100}));
}

// ==============================================================

TEST(Vector, Ops2) {

  nda::array<double, 1> V{1, 2, 3, 4, 5};
  auto W = V;

  W += V;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{1, 2, 3, 4, 5}));
  EXPECT_ARRAY_NEAR(W, (nda::array<double, 1>{2, 4, 6, 8, 10}));

  W -= V;
  EXPECT_ARRAY_NEAR(V, (nda::array<double, 1>{1, 2, 3, 4, 5}));
  EXPECT_ARRAY_NEAR(W, (nda::array<double, 1>{1, 2, 3, 4, 5}));
}
