#include "./test_common.hpp"

// -------------------------------------

TEST(View, ChangeData) {
  nda::array<long, 3> a(3, 3, 4);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) a(i, j, k) = i + 10 * j + 100 * k;

  auto v = a(_, 1, 2);

  EXPECT_EQ( (nda::slice_static::slice_layout_info(true, std::array<bool, 3>{1,0,0}, 2, std::array<int, 3>{0,1,2}, nda::layout_info_e::contiguous)),   nda::layout_info_e::strided_1d);

  EXPECT_EQ(v.shape(), (nda::shape_t<1>{3}));

  EXPECT_EQ(a(1, 1, 2), 1 + 10 * 1 + 100 * 2);

  a(1, 1, 2) = -28;
  EXPECT_EQ(v(1), a(1, 1, 2));
}

// -------------------------------------

TEST(View, OnRawPointers) {

  std::vector<long> _data(10, 3);

  nda::array_view<long const, 2> a({3, 3}, _data.data());

  EXPECT_EQ(a(1, 1), 3);

  // a(1,1) = 19; // DOES NOT COMPILE
}

// -------------------------------------

TEST(RawPointers, add) {

  std::vector<long> v1(10), v2(10), vr(10, -1);
  for (int i = 0; i < 10; ++i) {
    v1[i] = i;
    v2[i] = 10 * i;
  }

  nda::array_view<long const, 2> a({3, 3}, v1.data());
  nda::array_view<long const, 2> b({3, 3}, v2.data());
  nda::array_view<long, 2> c({3, 3}, vr.data());

  c = a + b;

  for (int i = 0; i < 9; ++i) EXPECT_EQ(vr[i], 11 * i);
  EXPECT_EQ(vr[9], -1);

  // a(1,1) = 19; // DOES NOT COMPILE
}
