// Copyright (c) 2023 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn

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
