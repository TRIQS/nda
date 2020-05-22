#include "./test_common.hpp"

// ==============================================================

// #ifdef _OPENMP 

using alloc_t = nda::allocators::segregator<8*100, nda::allocators::multiple_bucket<8*100>, nda::allocators::mallocator>;

TEST(CustomAlloc, Create1) { //NOLINT
  nda::basic_array<long, 2, C_layout, 'A', nda::heap_custom_alloc<alloc_t>> A(3, 3);
  EXPECT_EQ(A.shape(), (nda::shape_t<2>{3, 3}));
}

