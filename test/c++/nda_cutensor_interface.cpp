// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <cutensor.h>

#include <iostream>

// Test successful linking against cuTENSOR.
TEST(NDA, CUTENSORLinking) {
  auto version = cutensorGetVersion();
  std::cout << "cuTENSOR version: " << version << std::endl;
}
