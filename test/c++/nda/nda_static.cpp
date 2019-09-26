#include "./test_common.hpp"

// ==============================================================

TEST(StackArray, create) { //NOLINT

  //  nda::basic_array<long, 2, nda::basic_layout<nda::encode(std::array{3, 3}), nda::C_stride_order<3>, nda::layout_prop_e::contiguous>, 'A', nda::stack>
  nda::stack_array<long, 2, nda::static_extents(3, 3)> a;
  nda::array<long, 2> d(3, 3);

  a = 3;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      a(i, j) = i + 10 * j;
      d(i, j) = i + 10 * j;
    }

  auto ac = a;

  ac = a + d;

  NDA_PRINT(a.indexmap());
  NDA_PRINT(ac);
  NDA_PRINT(ac);
  NDA_PRINT(ac.indexmap());

  EXPECT_ARRAY_NEAR(a, d);
}

// ==============================================================

TEST(Loop, Static) { //NOLINT
  nda::array<long, 2> a(3, 3);

  nda::for_each_static<nda::encode(std::array{3, 3})>(a.shape(), [&a](auto x0, auto x1) { a(x0, x1) = x0 + 10 * x1; });

  std::cout << a << std::endl;
}
