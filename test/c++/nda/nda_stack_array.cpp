#include "./test_common.hpp"

// ==============================================================

TEST(StackArray, create) { //NOLINT

  nda::stack_array<long, 2, nda::static_extents(3, 3)> a;
  nda::array<long, 2> d(3, 3);

  a = 3;
  d = 3;
  EXPECT_ARRAY_NEAR(a, d);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      a(i, j) = i + 10 * j;
      d(i, j) = i + 10 * j;
    }

  auto ac = a;

  ac = a + d;

  NDA_PRINT(a.indexmap());
  //NDA_PRINT(ac);
  NDA_PRINT(ac.indexmap());

  EXPECT_ARRAY_NEAR(a, d);
}

// ==============================================================

TEST(StackArray, slice) { //NOLINT

  nda::stack_array<long, 2, nda::static_extents(3, 3)> a;

  a = 3;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) { a(i, j) = i + 10 * j; }

  {
    auto v = a(_, 1);

    nda::array<long, 2> ad{a};
    nda::array<long, 1> vd{v};

    NDA_PRINT(v.indexmap());
    NDA_PRINT(a);
    NDA_PRINT(v);

    EXPECT_ARRAY_NEAR(a, ad);
    EXPECT_ARRAY_NEAR(v, vd);
  }

  {
    auto v = a(1, _);

    nda::array<long, 2> ad{a};
    nda::array<long, 1> vd{v};

    NDA_PRINT(v.indexmap());
    NDA_PRINT(a);
    NDA_PRINT(v);

    EXPECT_ARRAY_NEAR(a, ad);
    EXPECT_ARRAY_NEAR(v, vd);
  }
}

// ==============================================================

TEST(Loop, Static) { //NOLINT
  nda::array<long, 2> a(3, 3);

  nda::for_each_static<nda::encode(std::array{3, 3}), 0>(a.shape(), [&a](auto x0, auto x1) { a(x0, x1) = x0 + 10 * x1; });

  std::cout << a << std::endl;
}
