#include "test_common.hpp"
#include <nda/array_adapter.hpp>

// ------------------------------------

TEST(array_adapter, base) { //NOLINT

  auto l = [](long i, long j) { return i + 2 * j; };

  auto al = nda::array_adapter{std::array{2, 2}, l};

#if (__cplusplus > 201703L)
  static_assert(nda::ArrayOfRank<decltype(al), 2>, "Opps");
#endif

  auto a = nda::array<long, 2>{al};

  EXPECT_EQ_ARRAY(a, (nda::array<long, 2>{{0, 2}, {1, 3}}));
}

// ------------------------------------

struct A {
  long i = 0;
};

struct B {
  int j = 0;
  B(A &&a) : j(a.i) { a.i = 0; }
};

TEST(array_adapter, move1) { //NOLINT

  nda::array<A, 2> arr1(2, 2);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) arr1(i, j) = A{1 + i + 10 * j};

  // -------
  auto l2 = [&arr1](long i, long j) { return B{std::move(arr1(i, j))}; };

  nda::array<B, 2> arr_b = nda::array_adapter{std::array{2, 2}, l2};
  // -------

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(arr1(i, j).i, 0);
      EXPECT_EQ(arr_b(i, j).j, (1 + i + 10 * j));
    }
}
// ----------------

TEST(array_adapter, move2) { //NOLINT

  nda::array<A, 2> arr1(2, 2);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) arr1(i, j) = A{1 + i + 10 * j};

  // -------
  auto l2 = [arr2 = std::move(arr1)](long i, long j) { return B{std::move(arr2(i, j))}; };
  nda::array<B, 2> arr_b = nda::array_adapter{std::array{2, 2}, l2};

  // -------
  EXPECT_TRUE(arr1.empty());

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) { EXPECT_EQ(arr_b(i, j).j, (1 + i + 10 * j)); }
}

// ----------------

TEST(array_adapter, map_equivalent) { //NOLINT

  nda::array<A, 2> arr1(2, 2);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) arr1(i, j) = A{1 + i + 10 * j};

  // -------
  nda::array<B, 2> arr_b = nda::map([](auto &&a) { return B{std::move(a)}; })(arr1);
  // -------

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(arr1(i, j).i, 0);
      EXPECT_EQ(arr_b(i, j).j, (1 + i + 10 * j));
    }
}
