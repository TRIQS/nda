// Copyright (c) 2021 Simons Foundation
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
// Authors: Nils Wentzell

#include "./test_common.hpp"

// ==============================================================

TEST(NDA, arange) { //NOLINT

  for (auto first : range(-100, 100)) {
    for (auto last : range(-100, 100)) {
      for (auto step : range(-100, 100)) {
        if (step == 0) continue;
        auto a = nda::arange(first, last, step);
        int n  = 0;
        for (auto i = first; step > 0 ? i < last : i > last; i = i + step) EXPECT_EQ(a[n++], i);
      }
    }
  }

  for (auto N : range(100)) EXPECT_EQ(nda::sum(nda::arange(N)), N * (N - 1) / 2);
}
