#include "./test_common.hpp"

// Check the calculation of layout_prop in a 5d : only some  slices
TEST(Slice, d3to2) { //NOLINT

  nda::array<dcomplex, 3, nda::contiguous_layout_with_stride_order<nda::encode(std::array{1, 0, 2})>> a(5, 2, 2);

  for (int w = 0; w < 5; ++w)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) a(w, i, j) = 100 * w + 10 * i + j;

  nda::array<dcomplex, 3> b;
  nda::array<dcomplex, 3> c(2,5,2);

  b = a;

  for (int w = 0; w < 5; ++w)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) {
        EXPECT_COMPLEX_NEAR((a(w, _, _)(i, j)), (b(w, i, j)));
        EXPECT_COMPLEX_NEAR((a(w, i, j)), (b(w, i, j)));
        EXPECT_COMPLEX_NEAR((a - b)(w, i, j), 0);
      }

  nda::array<dcomplex, 3> r = a - b;

  for (int w = 0; w < 5; ++w) {
    EXPECT_ARRAY_NEAR((a(w, _, _)), (b(w, _, _)));
  }
}
