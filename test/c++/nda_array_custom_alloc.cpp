// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/nda.hpp>

TEST(NDA, ArrayWithCustomAllocator) {
  using alloc_t = nda::mem::segregator<8ul * 100, nda::mem::multi_bucket<8 * 100>, nda::mem::mallocator<>>;
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<alloc_t>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));
}

TEST(NDA, SSOArray) {
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::sso<10>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));
  EXPECT_FALSE(A.storage().on_heap());

  nda::basic_array<long, 2, nda::C_layout, 'A', nda::sso<10>> B(3, 4);
  EXPECT_EQ(B.shape(), (shape_t<2>{3, 4}));
  EXPECT_TRUE(B.storage().on_heap());
}
