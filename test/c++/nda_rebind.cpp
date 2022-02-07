// Copyright (c) 2020 Simons Foundation
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

#include "./test_common.hpp"

// ==============================================================

TEST(Rebind, base) { //NOLINT

  nda::array<long, 2> a{{1, 2}, {3, 4}};
  nda::array<long, 2> b{{10, 20}, {30, 40}};

  auto v = a();
  v.rebind(b());

  EXPECT_EQ_ARRAY(v, b);
}

// ------------

TEST(Rebind, const) { //NOLINT

  nda::array<long, 2> a{{1, 2}, {3, 4}};
  nda::array<long, 2> b{{10, 20}, {30, 40}};

  auto const &aa = a;

  auto v = aa();
  v.rebind(b());

  // FIXME : const view should not compile
#if 0
  auto v2 = a();
  v2.rebind(v);
#endif

  EXPECT_EQ_ARRAY(v, b);
}
