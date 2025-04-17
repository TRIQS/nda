// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)

#include <nda/nda.hpp>
#include <sanitizer/asan_interface.h>

TEST(NDA, ASANPoison) {
  long *p;
  {
    nda::array<long, 2> A(3, 3);
    A() = 3;
    p   = &(A(0, 0));
  }

  EXPECT_EQ(__asan_address_is_poisoned(p), 1);
}

#endif
#endif
