#include "./test_common.hpp"

// ==============================================================

// #ifdef _OPENMP 

using alloc_t = nda::allocators::segregator<8 * 100, nda::allocators::multiple_bucket<8 * 100>, nda::allocators::mallocator>;

TEST(CustomAlloc, Create1) { //NOLINT
  nda::basic_array<long, 2, C_layout, 'A', nda::heap_custom_alloc<alloc_t>> A(3, 3);
  EXPECT_EQ(A.shape(), (nda::shape_t<2>{3, 3}));
}

// -------------------

TEST(SSO, Create1) { // NOLINT
  nda::basic_array<long, 2, C_layout, 'A', nda::sso<100>> A(3, 3);
  EXPECT_EQ(A.shape(), (nda::shape_t<2>{3, 3}));
}

// -------------------

TEST(SSO, is_at_right_place) { // NOLINT
  nda::basic_array<long, 2, C_layout, 'A', nda::sso<10>> a(3, 3);
  nda::basic_array<long, 2, C_layout, 'A', nda::sso<10>> b(3, 4);

  EXPECT_FALSE(a.storage().on_heap());
  EXPECT_TRUE(b.storage().on_heap());
}
