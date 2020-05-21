#include "test_common.hpp"
#include <nda/array_adapter.hpp>

// ------------------------------------

TEST(array_adapter, base) { //NOLINT

  auto l = [](long i, long j) { return i + 2 * j; };

  auto al = nda::array_adapter{std::array{2, 2}, l};

  static_assert(nda::ArrayOfRank<decltype(al), 2>, "Opps");

  auto a = nda::array<long, 2>{al};

  EXPECT_EQ_ARRAY(a, (nda::array<long, 2>{{0, 2}, {1, 3}}));
}
