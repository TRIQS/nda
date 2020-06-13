#include "./test_common.hpp"

static_assert(!std::is_pod<nda::array<long, 2>>::value, "POD pb");
static_assert(nda::is_scalar_for_v<int, matrix<std::complex<double>>> == 1, "oops");

// ==============================================================

TEST(Reinterpret, add_N_one) { //NOLINT
  nda::array<long, 2> a(3, 3);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j) a(i, j) = 1 + i + 10 * j;

  // view form
  {
    auto v = reinterpret_add_fast_dims_of_size_one<2>(a());

    EXPECT_EQ(v.shape(), (nda::shape_t<4>{3, 3, 1, 1}));

    for (int i = 0; i < a.extent(0); ++i)
      for (int j = 0; j < a.extent(1); ++j) EXPECT_EQ(v(i, j, 0, 0), 1 + i + 10 * j);
  }

  {
    // array form
    auto b = reinterpret_add_fast_dims_of_size_one<2>(std::move(a));

    EXPECT_EQ(b.shape(), (nda::shape_t<4>{3, 3, 1, 1}));

    for (int i = 0; i < a.extent(0); ++i)
      for (int j = 0; j < a.extent(1); ++j) EXPECT_EQ(b(i, j, 0, 0), 1 + i + 10 * j);
  }
}
