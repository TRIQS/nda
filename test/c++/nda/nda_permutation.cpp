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
