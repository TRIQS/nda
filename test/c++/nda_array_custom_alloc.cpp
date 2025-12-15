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

TEST(NDA, ArrayWithDynamicBucketAllocator) {
  using alloc_t = nda::mem::dynamic_bucket<nda::mem::Host>;
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<alloc_t>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));
}

TEST(NDA, ArrayWithFallbackAllocator) {
  using primary_alloc_t = nda::mem::dynamic_bucket<nda::mem::Host>;
  using alloc_t         = nda::mem::fallback<primary_alloc_t>;
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<alloc_t>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));

  {
    nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<alloc_t>> B(4, 5);
    EXPECT_EQ(B.shape(), (shape_t<2>{4, 5}));
  }

  {
    auto B(A);
    auto C = nda::make_regular(A + B);
    EXPECT_EQ(B.shape(), A.shape());
    EXPECT_EQ(C.shape(), A.shape());

    C.resize(shape_t<2>{4, 4});
    EXPECT_EQ(C.shape(), (shape_t<2>{4, 4}));
  }
}

TEST(NDA, ArrayWithStaticFallbackAllocator) {
  using primary_alloc_t = nda::mem::dynamic_bucket<nda::mem::Host>;
  using alloc_t         = nda::mem::static_fallback<primary_alloc_t>;
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<alloc_t>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));

  {
    nda::basic_array<long, 2, nda::C_layout, 'A', nda::heap_basic<alloc_t>> B(4, 5);
    EXPECT_EQ(B.shape(), (shape_t<2>{4, 5}));
  }

  {
    auto B(A);
    auto C = nda::make_regular(A + B);
    EXPECT_EQ(B.shape(), A.shape());
    EXPECT_EQ(C.shape(), A.shape());

    C.resize(shape_t<2>{4, 4});
    EXPECT_EQ(C.shape(), (shape_t<2>{4, 4}));
  }
}

TEST(NDA, SSOArray) {
  nda::basic_array<long, 2, nda::C_layout, 'A', nda::sso<10>> A(3, 3);
  EXPECT_EQ(A.shape(), (shape_t<2>{3, 3}));
  EXPECT_FALSE(A.storage().on_heap());

  nda::basic_array<long, 2, nda::C_layout, 'A', nda::sso<10>> B(3, 4);
  EXPECT_EQ(B.shape(), (shape_t<2>{3, 4}));
  EXPECT_TRUE(B.storage().on_heap());
}
