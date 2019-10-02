#include "./test_common.hpp"

using clef::placeholder;

// ==============================================================

TEST(NDA, LazyFill) { //NOLINT

  placeholder<0> i_;
  placeholder<1> j_;

  nda::array<double, 2> A(2, 2);
  A(i_, j_) << i_ * 8.1 + 2.31 * j_;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(A(i, j), i * 8.1 + 2.31 * j);
}
