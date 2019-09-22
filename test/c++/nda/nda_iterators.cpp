#include "./test_common.hpp"

// ==============================================================

TEST(iterator, empty) {
  nda::array<int, 1> arr(0);
  int s = 0;
  for (auto i : arr) s += i;
  EXPECT_EQ(s, 0);
}

//-----------------------------

TEST(iterator, Contiguous1d) { //NOLINT
  nda::array<long, 1> a;
  for (int i = 0; i < a.extent(0); ++i) a(i) = 1 + i;

  long c = 1;
  for (auto x : a) { EXPECT_EQ(x, c++); }
}

//-----------------------------

TEST(iterator, Contiguous2d) { //NOLINT
  nda::array<long, 2> a(2, 3);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j) a(i, j) = 1 + i + 10 * j;

  auto it = a.begin();

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j) {
      EXPECT_EQ(*it, a(i, j));
      EXPECT_FALSE(it == a.end());
      ++it;
    }
}

//-----------------------------

TEST(iterator, Contiguous3d) { //NOLINT
  nda::array<long, 3> a(3, 5, 9);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k) a(i, j, k) = 1 + i + 10 * j + 100 * k;

  auto it = a.begin();

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k) {
        EXPECT_EQ(*it, a(i, j, k));
        EXPECT_TRUE(it != a.end());
        ++it;
      }
}

//-----------------------------

TEST(iterator, Strided3d) { //NOLINT
  nda::array<long, 3> a(3, 5, 9);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k) a(i, j, k) = 1 + i + 10 * j + 100 * k;

  auto v = a(range(0, a.extent(0), 2), range(0, a.extent(1), 2), range(0, a.extent(2), 2));

  auto it = v.begin();

  for (int i = 0; i < v.extent(0); ++i)
    for (int j = 0; j < v.extent(1); ++j)
      for (int k = 0; k < v.extent(2); ++k) {
        EXPECT_EQ(*it, v(i, j, k));
        EXPECT_TRUE(it != v.end());
        ++it;
      }
  EXPECT_TRUE(it == v.end());

}
//-----------------------------

TEST(iterator, bug) { //NOLINT
  const int N1 = 1000, N2 = 1000;
  nda::array<double, 2> a(2 * N1, 2 * N2);
  auto v = a(range(0, -1, 2), range(0, -1, 2));
  for (auto &x : v) { x = 10; }
}
