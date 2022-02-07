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

TEST(Shared, Lifetime) { //NOLINT

  nda::array<double, 2> a(2, 3);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = i * 8.1 + 2.31 * j;

  using v_t = nda::basic_array_view<double, 2, C_layout, 'A', nda::default_accessor, nda::shared>;
  v_t v;

  {
    auto b = a;
    v.rebind(v_t{b});
  }

  EXPECT_EQ_ARRAY(v, a);
}

// ==============================================================
