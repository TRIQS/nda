#include "./test_common.hpp"

TEST(NDA, compound_ops) {

  nda::array<long, 2> A(2, 3);

  for (long i = 0; i < 2; ++i)
    for (long j = 0; j < 3; ++j) A(i, j) =  10 * i + j;

  auto A2 = A;

  A *= 2.0;
  EXPECT_ARRAY_NEAR(A, nda::array<long, 2>{{0, 2, 4}, {20, 22, 24}});

  A2 /= 2.0;
  EXPECT_ARRAY_NEAR(A2, nda::array<long, 2>{{0, 0, 1}, {5, 5, 6}});

  nda::array<double, 2> B(A);
  B /= 4;
  EXPECT_ARRAY_NEAR(B, nda::array<double, 2>{{0.0, 0.5, 1.0}, {5.0, 5.5, 6.0}});
}
MAKE_MAIN;
