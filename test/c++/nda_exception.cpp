// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/exceptions.hpp>

#include <exception>
#include <iostream>
#include <string>

TEST(NDA, AccumulateErrorMessageAndThrow) {
  std::string msg("Test error message");
  try {
    throw nda::runtime_error{} << msg;
  } catch (std::exception const &e) {
    std::cout << e.what() << std::endl;
    EXPECT_EQ(std::string(e.what()), msg);
  }
}

TEST(NDA, AssertMacro) {
  ASSERT_NO_THROW(NDA_ASSERT(true));
  ASSERT_THROW(NDA_ASSERT(false), nda::runtime_error);
}
