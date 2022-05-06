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

// ====================== VIEW ========================================

using namespace nda::permutations;

TEST(Permutation, cycle) { //NOLINT

  EXPECT_EQ(identity<5>(), (std::array{0, 1, 2, 3, 4}));
  EXPECT_EQ(reverse_identity<5>(), (std::array{4, 3, 2, 1, 0}));

  EXPECT_EQ(cycle<5>(1), (std::array{4, 0, 1, 2, 3}));
  EXPECT_EQ(cycle<5>(1, 3), (std::array{2, 0, 1, 3, 4}));
  EXPECT_EQ(cycle<5>(-1, 3), (std::array{1, 2, 0, 3, 4}));

  EXPECT_EQ(cycle<5>(-1, 0), identity<5>());
}

namespace nda {
  // FIXME : MOVE UP
  // Rotate the lengths / strides for indices lower than N cyclically forward by one
  template <int N, typename A> //[[deprecated]]
  auto rotate_index_view(A &&a) {
    return permuted_indices_view<encode(nda::permutations::cycle<get_rank<A>>(-1, N + 1))>(std::forward<A>(a));
  }
} // namespace nda

// ---------------------------------------------

TEST(Permutation, Rotate) { //NOLINT

  nda::array<long, 4> a(3, 4, 5, 6);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k)
        for (int l = 0; l < a.extent(3); ++l) a(i, j, k, l) = 1 + i + 10 * j + 100 * k + 1000 * l;

  auto v = nda::rotate_index_view<2>(a);

  PRINT(a.indexmap().lengths());
  PRINT(v.indexmap().lengths());

  PRINT(a.indexmap().strides());
  PRINT(v.indexmap().strides());

  EXPECT_EQ(v.shape(), (std::array<long, 4>{5, 3, 4, 6}));

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k)
        for (int l = 0; l < a.extent(3); ++l) EXPECT_EQ(a(i, j, k, l), v(k, i, j, l));

  for (int i = 0; i < v.extent(0); ++i)
    for (int j = 0; j < v.extent(1); ++j)
      for (int k = 0; k < v.extent(2); ++k)
        for (int l = 0; l < v.extent(3); ++l) EXPECT_EQ(v(i, j, k, l), a(j, k, i, l));
}

// ---------------------------------------------
TEST(Permutation, Iterator) { //NOLINT

  nda::array<long, 4> a(3, 4, 5, 6);

  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(1); ++j)
      for (int k = 0; k < a.extent(2); ++k)
        for (int l = 0; l < a.extent(3); ++l) a(i, j, k, l) = 1 + i + 10 * j + 100 * k + 1000 * l;

  {
    auto it = a.begin();

    for (int i = 0; i < a.extent(0); ++i)
      for (int j = 0; j < a.extent(1); ++j)
        for (int k = 0; k < a.extent(2); ++k)
          for (int l = 0; l < a.extent(3); ++l) { EXPECT_EQ(a(i, j, k, l), (*it++)); }
  }

  auto v = nda::rotate_index_view<2>(a);
  {
    auto it = v.begin();

    // We traverse the view in a memory-contiguous way
    for (int j = 0; j < v.extent(1); ++j)
      for (int k = 0; k < v.extent(2); ++k)
        for (int i = 0; i < v.extent(0); ++i)
          for (int l = 0; l < v.extent(3); ++l) { EXPECT_EQ(v(i, j, k, l), (*it++)); }
  }
}
