#include "./test_common.hpp"

// ====================== VIEW ========================================

using namespace nda::permutations;

TEST(Permutation, cycle) { //NOLINT

  EXPECT_EQ(identity<5>(), (std::array{0, 1, 2, 3, 4}));
  EXPECT_EQ(reverse_identity<5>(), (std::array{4, 3, 2, 1, 0}));

  EXPECT_EQ(cycle<5>(1), (std::array{4, 0, 1, 2, 3}));
  EXPECT_EQ(cycle<5>(1, 3), (std::array{2, 0, 1, 3, 4}));
  EXPECT_EQ(cycle<5>(-1, 3), (std::array{1, 2, 0, 3, 4}));

  EXPECT_EQ(cycle<5>(-1, 0), identity<5>());
}

namespace nda {
  // FIXME : MOVE UP
  // Rotate the index n to 0, preserving the relative order of the other indices
  template <int N, typename A> //[[deprecated]]
  auto rotate_index_view(A &&a) {
    return permuted_indices_view<encode(nda::permutations::cycle<get_rank<A>>(-1, N + 1))>(std::forward<A>(a));
  }
} // namespace nda

// ---------------------------------------------

TEST(Permutation, Rotate) { //NOLINT

  nda::array<long, 4> a(3, 4, 5, 6);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k)
        for (int l = 0; l < a.extent(3); ++l) a(i, j, k, l) = 1 + i + 10 * j + 100 * k + 1000 * l;

  auto v = nda::rotate_index_view<2>(a);

  PRINT(a.indexmap().lengths());
  PRINT(v.indexmap().lengths());

  PRINT(a.indexmap().strides());
  PRINT(v.indexmap().strides());

  EXPECT_EQ(v.shape(), (std::array<long, 4>{5, 3, 4, 6}));

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k)
        for (int l = 0; l < a.extent(3); ++l) EXPECT_EQ(a(i, j, k, l), v(k, i, j, l));

  for (int i = 0; i < v.extent(0); ++i)
    for (int j = 0; j < v.extent(1); ++j)
      for (int k = 0; k < v.extent(2); ++k)
        for (int l = 0; l < v.extent(3); ++l) EXPECT_EQ(v(i, j, k, l), a(j, k, i, l));
}

// ---------------------------------------------
TEST(Permutation, Iterator) { //NOLINT

  nda::array<long, 4> a(3, 4, 5, 6);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k)
        for (int l = 0; l < a.extent(3); ++l) a(i, j, k, l) = 1 + i + 10 * j + 100 * k + 1000 * l;

  {
    auto it = a.begin();

    for (int i = 0; i < a.extent(0); ++i)
      for (int j = 0; j < a.extent(1); ++j)
        for (int k = 0; k < a.extent(2); ++k)
          for (int l = 0; l < a.extent(3); ++l) { EXPECT_EQ(a(i, j, k, l), (*it++)); }
  }

  auto v = nda::rotate_index_view<2>(a);
PRINT(v.iterator_rank);
  {
    auto it = v.begin();

    for (int i = 0; i < v.extent(0); ++i)
      for (int j = 0; j < v.extent(1); ++j)
        for (int k = 0; k < v.extent(2); ++k)
          for (int l = 0; l < v.extent(3); ++l) { EXPECT_EQ(v(i, j, k, l), (*it++)); }
  }
}
