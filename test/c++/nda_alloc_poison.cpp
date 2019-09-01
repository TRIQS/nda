#include "./test_common.hpp"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#endif
#endif

TEST(Array, Poison) {

  long *p;
  {
    nda::array<long, 2> A(3, 3);
    A() = 3;
    p   = &(A(0, 0));
  }

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
  EXPECT_EQ(__asan_address_is_poisoned(p), 1);
#endif
#endif
}

MAKE_MAIN;
