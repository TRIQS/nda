#include "./test_common.hpp"

// ==============================================================

TEST(Shared, Lifetime) { //NOLINT

  nda::array<double, 2> a(2, 3);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = i * 8.1 + 2.31 * j;

  using v_t = nda::basic_array_view<double, 2, C_layout, 'A', nda::default_accessor, nda::shared>;
  v_t v;

  {
    auto b = a;
    v.rebind(v_t{b});
  }

  EXPECT_EQ_ARRAY(v, a);
}

// ==============================================================
