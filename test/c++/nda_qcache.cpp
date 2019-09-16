#include "./test_common.hpp"

#include <nda/blas_lapack/qcache.hpp>

using nda::qcache;

// -------------------------------------

TEST(qcache, no_copy) {

  nda::array<long, 2> a(3, 3);

  auto c = qcache(a);

  EXPECT_FALSE(c.use_copy());
  EXPECT_ARRAY_EQ(c(), a);
  EXPECT_EQ(c().data_start(), a.data_start());
}

// --------------------

TEST(qcache, view_no_copy) {

  nda::array<long, 2> a(4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) a(i, j) = i + 10 * j;

  auto v = a(nda::range(1, 4, 2), nda::range(1, 4, 1));

  auto c = qcache(v);

  //NDA_PRINT(a);
  //NDA_PRINT(v);

  EXPECT_FALSE(c.use_copy());
  EXPECT_ARRAY_EQ(c(), v);
}
// --------------------

TEST(qcache, view_with_copy) {

  nda::array<long, 2> a(4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) a(i, j) = i + 10 * j;

  auto v = a(nda::range(1, 4, 2), nda::range(1, 4, 2));

  auto c = qcache(v);

  //NDA_PRINT(a);
  //NDA_PRINT(v);

  EXPECT_TRUE(c.use_copy());
  EXPECT_ARRAY_EQ(c(), v);
}

// --------------------

TEST(qcache, reflexive) {

  nda::array<long, 2> a(4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) a(i, j) = i + 10 * j;

  auto ref = a;
  ref(1, 1) *= -10;

  auto v = a(nda::range(1, 4, 2), nda::range(1, 4, 2));

  {
    auto c = reflexive_qcache(v);

    //NDA_PRINT(a);

    EXPECT_TRUE(c.use_copy());
    EXPECT_ARRAY_EQ(c(), v);

    c()(0, 0) *= -10;

    //NDA_PRINT(a);
  }

  //NDA_PRINT(a);

  EXPECT_ARRAY_EQ(ref, a); // changed copied back
}

// --------------------

TEST(qcache, no_reflexive_modif_ignored) {

  nda::array<long, 2> a(4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) a(i, j) = i + 10 * j;

  auto ref = a;
  auto v   = a(nda::range(1, 4, 2), nda::range(1, 4, 2));

  {
    auto c = qcache(v);

    EXPECT_TRUE(c.use_copy());
    EXPECT_ARRAY_EQ(c(), v);

    c()(0, 0) *= -10;
  }

  //NDA_PRINT(a);

  EXPECT_ARRAY_EQ(ref, a); // no change
}
