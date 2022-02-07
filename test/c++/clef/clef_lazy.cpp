// Copyright (c) 2019-2020 Simons Foundation
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

#include "./common.hpp"
#include <nda/clef/adapters/vector.hpp>

using namespace clef;

clef::placeholder<10> i_;
clef::placeholder<11> j_;

TEST(Clef, Lazy) {
  std::vector<int> V{14, 2, 3}, W(3, 0);

  EXPECT_EQ(eval(make_expr(V)[i_], i_ = 0), 14);

  make_expr(V)[i_] << i_ + 2;
  EXPECT_EQ(V[0], 2);
  EXPECT_EQ(V[1], 3);
  EXPECT_EQ(V[2], 4);

  make_expr(W)[i_] << i_ + make_expr(V)[i_];

  EXPECT_EQ(W[0], 2);
  EXPECT_EQ(W[1], 4);
  EXPECT_EQ(W[2], 6);

  std::vector<std::vector<int>> v2(3, std::vector<int>(2));

  make_expr(v2)[i_][j_] << (i_ + j_ + 1);

  for (size_t u = 0; u < v2.size(); ++u)
    for (size_t up = 0; up < v2[0].size(); ++up) EXPECT_EQ(v2[u][up], u + up + 1);
}
