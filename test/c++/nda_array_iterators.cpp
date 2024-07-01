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

#include <nda/iterators.hpp>
#include <nda/layout/idx_map.hpp>
#include <nda/layout/policies.hpp>

#include <array>
#include <numeric>
#include <vector>

TEST(NDA, ArrayIterator1D) {
  // test array iterators for a 1D array
  int constexpr size = 10;
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), 0);

  // loop over various strides
  for (int s = 1; s <= size; ++s) {
    auto shape   = std::array<long, 1>{size / s};
    auto strides = std::array<long, 1>{s};
    auto it      = nda::array_iterator<1, int, int *>(shape, strides, vec.data(), false);
    auto it_end  = nda::array_iterator<1, int, int *>(shape, strides, vec.data(), true);
    int exp_val  = 0;
    for (; it != it_end; ++it, exp_val += s) { EXPECT_EQ(*it, exp_val); }
  }
}

TEST(NDA, ArrayIterator3D) {
  // test array iterators for a 3D array
  long constexpr n1   = 2;
  long constexpr n2   = 5;
  long constexpr n3   = 3;
  auto constexpr size = n1 * n2 * n3;
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), 0);

  // iterate over various strides (h5 style strides)
  for (int s1 = 1; s1 <= n1; ++s1) {
    for (int s2 = 1; s2 <= n2; ++s2) {
      for (int s3 = 1; s3 <= n3; ++s3) {
        // compare with idx_map
        auto shape   = std::array<long, 3>{n1 / s1, n2 / s2, n3 / s3};
        auto strides = std::array<long, 3>{s1 * n2 * n3, s2 * n3, s3};
        auto idxm    = nda::C_stride_layout::mapping<3>(shape, strides);

        auto it     = nda::array_iterator<3, int, int *>(shape, strides, vec.data(), false);
        auto it_end = nda::array_iterator<3, int, int *>(shape, strides, vec.data(), true);
        for (int i = 0; i < shape[0]; ++i) {
          for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
              EXPECT_EQ(*it, vec[idxm(i, j, k)]);
              ++it;
            }
          }
        }
        EXPECT_EQ(it, it_end);
      }
    }
  }
}
