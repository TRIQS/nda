// Copyright (c) 2021--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <tblis/tblis.h>

// Test various typedefs of TBLIS.
TEST(NDA, TBLISTypes) {
  static_assert(std::same_as<::tblis::len_type, long>);
  static_assert(std::same_as<::tblis::stride_type, long>);
  static_assert(std::same_as<::tblis::label_type, char>);
}
