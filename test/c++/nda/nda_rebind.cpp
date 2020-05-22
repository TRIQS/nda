#include "./test_common.hpp"


// ==============================================================

TEST(Rebind, base) { //NOLINT
  
  nda::array<long, 2> a {{1,2}, {3,4}};
  nda::array<long, 2> b {{10,20}, {30,40}};
  
  auto v = a();
  v.rebind(b());

  EXPECT_EQ_ARRAY(v, b);
}

// ------------

TEST(Rebind, const) { //NOLINT
  
  nda::array<long, 2> a {{1,2}, {3,4}};
  nda::array<long, 2> b {{10,20}, {30,40}};
  
  auto const & aa = a;

  auto v = aa();
  v.rebind(b());

  // FIXME : const view should not compile
#if 0
  auto v2 = a();
  v2.rebind(v);
#endif

  EXPECT_EQ_ARRAY(v, b);
}


