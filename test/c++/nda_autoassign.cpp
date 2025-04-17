// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/nda.hpp>

using namespace nda::clef::literals;

TEST(NDA, AutoAssign2DArray) {
  nda::array<double, 2> A(2, 2);
  A(i_, j_) << i_ * 8.1 + 2.31 * j_;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(A(i, j), i * 8.1 + 2.31 * j);
}

TEST(NDA, AutoAssignArrayofArray) {
  nda::array<nda::array<double, 1>, 2> A(2, 2);
  A = nda::array<double, 1>(3);
  A(i_, j_)(k_) << i_ + 8.1 * j_ + 100 * k_;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 3; ++k) EXPECT_EQ((A(i, j)(k)), i + 8.1 * j + 100 * k);
}
