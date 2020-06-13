#include "./test_common.hpp"

// ==============================================================

TEST(NDA, Ellipsis) { //NOLINT
  nda::array<long, 3> A(2, 3, 4);
  A() = 7;

  EXPECT_ARRAY_NEAR(A(0, ___), A(0, _, _), 1.e-15);

  nda::array<long, 4> B(2, 3, 4, 5);
  B() = 8;

  EXPECT_ARRAY_NEAR(B(0, ___, 3), B(0, _, _, 3), 1.e-15);
  EXPECT_ARRAY_NEAR(B(0, ___, 2, 3), B(0, _, 2, 3), 1.e-15);
  EXPECT_ARRAY_NEAR(B(___, 2, 3), B(_, _, 2, 3), 1.e-15);
}

// ==============================================================

TEST(NDA, NullEllipsis) { //NOLINT
  nda::array<long, 3> A(2, 3, 4);
  A() = 7;

  EXPECT_ARRAY_NEAR(A(0, ___), A(0, _, _), 1.e-15);

  nda::array<long, 4> B(2, 3, 4, 5);
  B() = 8;

  EXPECT_ARRAY_NEAR(B(1, 2, 3, _), B(1, 2, 3, _, ___), 1.e-15);
  EXPECT_ARRAY_NEAR(B(1, 2, 3, _), B(1, 2, ___, 3, _), 1.e-15);
  EXPECT_ARRAY_NEAR(B(1, 2, 3, _), B(1, ___, 2, 3, _), 1.e-15);
  EXPECT_ARRAY_NEAR(B(1, 2, 3, _), B(___, 1, 2, 3, _), 1.e-15);

  EXPECT_NEAR(B(1, 2, 3, 4), B(1, 2, 3, 4, ___), 1.e-15);
  EXPECT_NEAR(B(1, 2, 3, 4), B(___, 1, 2, 3, 4), 1.e-15);
  EXPECT_NEAR(B(1, 2, 3, 4), B(1, ___, 2, 3, 4), 1.e-15);
}

// ==============================================================

template <typename ArrayType>
auto sum0(ArrayType const &A) {
  nda::array<typename ArrayType::value_type, ArrayType::rank - 1> res = A(0, ___);
  for (size_t u = 1; u < A.shape()[0]; ++u) res += A(u, ___);
  return res;
}

TEST(NDA, Ellipsis2) { //NOLINT
  nda::array<double, 2> A(5, 2);
  A() = 2;
  nda::array<double, 3> B(5, 2, 3);
  B() = 3;
  EXPECT_ARRAY_NEAR(sum0(A), nda::array<double, 1>{10, 10}, 1.e-15);
  EXPECT_ARRAY_NEAR(sum0(B), nda::array<double, 2>{{15, 15, 15}, {15, 15, 15}}, 1.e-15);
}


