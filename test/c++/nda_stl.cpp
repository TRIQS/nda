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

#include "./test_common.hpp"

#include <nda/nda.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <vector>

TEST(NDA, STLContainersWithArrays) {
  // make STL containers which contain nda::array objects
  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  // vector of arrays
  std::vector<nda::array<long, 2>> vec;
  vec.push_back(A);

  // map of arrays
  std::map<std::string, nda::array<long, 2>> map;
  map["1"] = A;
}

TEST(NDA, STLCopyWithArrays) {
  std::vector<int> vec(10);
  for (int i = 0; i < 10; ++i) vec[i] = 10 + i;

  // copy contents of a std::vector to an nda::array
  nda::array<int, 1> A(vec.size());
  std::copy(vec.begin(), vec.end(), A.begin());
  for (int i = 0; i < 10; ++i) EXPECT_EQ(A(i), vec[i]);

  // copy content of an nda::array to a std::vector
  for (int i = 0; i < 10; ++i) A(i) *= 2;
  std::copy(A.begin(), A.end(), vec.begin());
  for (int i = 0; i < 10; ++i) EXPECT_EQ(vec[i], A(i));
}

TEST(NDA, STLAlgorithmsWithArrays) {
  nda::array<int, 1> A{1, 34, 2, 6, 23, 189, 8, 4};
  auto lt25 = [](int x) { return (x < 25); };

  // std::count_if
  EXPECT_EQ(std::count_if(A.begin(), A.end(), lt25), 6);

  // std::max_element
  EXPECT_EQ(*std::max_element(A.begin(), A.end()), 189);

  // std::replace_if
  std::replace_if(A.begin(), A.end(), lt25, 0);
  EXPECT_EQ(A, (nda::array<int, 1>{0, 34, 0, 0, 0, 189, 0, 0}));
}

TEST(NDA, STLSortAnArray) {
  nda::array<int, 1> V(10);
  for (int i = 0; i < 10; ++i) V[i] = 10 - i;
  std::sort(V.begin(), V.end());
  for (int i = 0; i < 10; ++i) EXPECT_EQ(V[i], 1 + i);
}

TEST(NDA, STLShuffleAndSortAnArray) {
  nda::array<int, 1> V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::mt19937 rng(std::random_device{}());
  std::shuffle(V.begin(), V.end(), rng);
  std::sort(V.begin(), V.end());
  for (unsigned int i = 0; i < 10; ++i) EXPECT_EQ(V[i], 1 + i);
}

TEST(NDA, STLShuffleAndSortAView) {
  nda::array<int, 1> A(20, -9);
  auto A_v = A(nda::range(0, 20, 2));
  for (int i = 0; i < 10; ++i) A_v[i] = 1 + i;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(A_v.begin(), A_v.end(), g);
  std::sort(A_v.begin(), A_v.end());

  EXPECT_EQ((A_v.begin()[3]), 4);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(A[2 * i], 1 + i);
    EXPECT_EQ(A[2 * i + 1], -9);
  }
}
