// Copyright (c) 2018-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#include <gtest/gtest.h> // NOLINT
#include <memory>

#define NDA_DEBUG_MEMORY

#include <nda/mem/handle.hpp>

using namespace nda::mem;

//---------------------------------------------

TEST(Storage, HR) { // NOLINT

  handle_heap<int> h{10};

  // make sure it is a copy
  h.data()[2] = 89;
  handle_heap<int> h3{h};
  h.data()[2] = 0;
  EXPECT_EQ(h3.data()[2], 89); //NOLINT
}

// ---- Construct R, S
TEST(Storage, HSR) { // NOLINT

  handle_heap<int> h{10};

  handle_shared<int> s{h};

  EXPECT_EQ(s.refcount(), 2); //NOLINT
}

// ---- More complex
TEST(Ref, HSRS) { // NOLINT

  handle_heap<int> h{10};

  handle_shared<int> s{h};
  EXPECT_EQ(s.refcount(), 2); //NOLINT

  s = handle_shared<int>{h};
  EXPECT_EQ(s.refcount(), 2); //NOLINT

  handle_shared<int> s2{h};
  s = s2;
  EXPECT_EQ(s.refcount(), 3); //NOLINT
}

// ---- check with something that is constructed/destructed.
struct Number {
  int u               = 9;
  static inline int c = 0;
  Number() {
    c++;
    std::cerr << "Constructing Number \n";
  };
  ~Number() {
    c--;
    std::cerr << "Destructing Number \n";
  };
};

TEST(Storage, HR_with_cd) { // NOLINT
  { handle_heap<Number> h{5}; }
  EXPECT_EQ(Number::c, 0); //NOLINT
}

// --- check with a shared_ptr

TEST(Storage, HR_with_sharedPtr) { // NOLINT
  { handle_shared<Number> s; }
  EXPECT_EQ(Number::c, 0); //NOLINT
}
