// Copyright (c) 2020-2022 Simons Foundation
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

#include <gtest/gtest.h>

#include <nda/layout/permutation.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <random>

// Make identity or random permutation of size N.
template <int N>
auto make_permutation(bool random = true) {
  std::array<int, N> arr{};
  std::iota(arr.begin(), arr.end(), 0);
  if (random) { std::shuffle(arr.begin(), arr.end(), std::mt19937{0xd62847d0}); }
  return arr;
}

// Check encode/decode function.
template <int N>
void check_encode_decode() {
  auto arr     = make_permutation<N>();
  auto code    = nda::encode(arr);
  auto decoded = nda::decode<N>(code);
  EXPECT_EQ(arr, decoded);
}

TEST(NDA, LayoutEncodeDecode) {
  check_encode_decode<0>();
  check_encode_decode<1>();
  check_encode_decode<2>();
  check_encode_decode<3>();
  check_encode_decode<4>();
  check_encode_decode<5>();
  check_encode_decode<6>();
  check_encode_decode<7>();
  check_encode_decode<8>();
  check_encode_decode<9>();
  check_encode_decode<10>();
  check_encode_decode<11>();
  check_encode_decode<12>();
  check_encode_decode<13>();
  check_encode_decode<14>();
  check_encode_decode<15>();
}

TEST(NDA, PermutationIsValid) {
  constexpr int size = 5;
  auto perm          = make_permutation<size>();
  EXPECT_TRUE(nda::permutations::is_valid(perm));
  EXPECT_FALSE(nda::permutations::is_valid(nda::stdutil::append(perm, size - 1)));
  EXPECT_FALSE(nda::permutations::is_valid(nda::stdutil::append(perm, size + 1)));
  EXPECT_FALSE(nda::permutations::is_valid(nda::stdutil::append(perm, -1)));
  perm[0] = size;
  EXPECT_FALSE(nda::permutations::is_valid(perm));
}

TEST(NDA, PermutationComposeAndInverse) {
  auto perm = std::array{1, 3, 0, 4, 2};
  auto id   = make_permutation<std::size(perm)>(false);

  EXPECT_EQ(id, nda::permutations::compose(id, id));
  EXPECT_EQ(id, nda::permutations::inverse(id));

  EXPECT_EQ(perm, nda::permutations::compose(id, perm));
  EXPECT_EQ(perm, nda::permutations::compose(perm, id));
  EXPECT_EQ(id, nda::permutations::compose(perm, nda::permutations::inverse(perm)));
  EXPECT_EQ(id, nda::permutations::compose(nda::permutations::inverse(perm), perm));
}

TEST(NDA, PermutationApplyAndApplyInverse) {
  auto perm = std::array{1, 3, 0, 4, 2};
  auto arr  = std::array{'a', 'e', 'i', 'o', 'u'};
  EXPECT_NE(arr, nda::permutations::apply(perm, arr));
  EXPECT_NE(arr, nda::permutations::apply_inverse(perm, arr));
  EXPECT_EQ(arr, nda::permutations::apply_inverse(perm, nda::permutations::apply(perm, arr)));
}

TEST(NDA, PermutationIdentityAndReverseIdentity) {
  constexpr int size = 5;
  auto id            = nda::permutations::identity<size>();
  auto rev_id        = nda::permutations::reverse_identity<size>();
  EXPECT_EQ(id, make_permutation<size>(false));
  EXPECT_EQ(id, nda::permutations::compose(rev_id, rev_id));
}

TEST(NDA, PermutationTransposition) {
  auto arr    = std::array{10, 100, 1000};
  auto transp = nda::permutations::transposition<std::size(arr)>(0, std::size(arr) - 1);
  EXPECT_EQ((std::array{1000, 100, 10}), nda::permutations::apply(transp, arr));
  EXPECT_EQ(arr, nda::permutations::apply(transp, nda::permutations::apply(transp, arr)));
}

TEST(NDA, PermutationCycle) {
  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(0));
  EXPECT_EQ((std::array{3, 0, 1, 2}), nda::permutations::cycle<4>(1));
  EXPECT_EQ((std::array{2, 3, 0, 1}), nda::permutations::cycle<4>(2));
  EXPECT_EQ((std::array{1, 2, 3, 0}), nda::permutations::cycle<4>(3));
  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(4));
  EXPECT_EQ((std::array{1, 2, 3, 0}), nda::permutations::cycle<4>(-1));
  EXPECT_EQ((std::array{2, 3, 0, 1}), nda::permutations::cycle<4>(-2));
  EXPECT_EQ((std::array{3, 0, 1, 2}), nda::permutations::cycle<4>(-3));
  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(-4));

  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(0, 2));
  EXPECT_EQ((std::array{1, 0, 2, 3}), nda::permutations::cycle<4>(1, 2));
  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(2, 2));
  EXPECT_EQ((std::array{1, 2, 0, 3}), nda::permutations::cycle<4>(-1, 3));
  EXPECT_EQ((std::array{2, 0, 1, 3}), nda::permutations::cycle<4>(-2, 3));
  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(-3, 3));

  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(0, 5));
  EXPECT_EQ((std::array{3, 0, 1, 2}), nda::permutations::cycle<4>(1, 5));
  EXPECT_EQ((std::array{2, 3, 0, 1}), nda::permutations::cycle<4>(2, 5));
  EXPECT_EQ((std::array{1, 2, 3, 0}), nda::permutations::cycle<4>(3, 5));
  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(4, 5));
  EXPECT_EQ((std::array{1, 2, 3, 0}), nda::permutations::cycle<4>(-1, 5));
  EXPECT_EQ((std::array{2, 3, 0, 1}), nda::permutations::cycle<4>(-2, 5));
  EXPECT_EQ((std::array{3, 0, 1, 2}), nda::permutations::cycle<4>(-3, 5));
  EXPECT_EQ((std::array{0, 1, 2, 3}), nda::permutations::cycle<4>(-4, 5));
}
