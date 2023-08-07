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
#include <vector>
#include <map>
#include <algorithm>
#include <random>

template <typename T>
using vector_t = nda::array<T, 1>;

// A few examples paying with STL containers and algorithms

// ==============================================================

// We can put nda::arrays in simple containers (there are regular types)
TEST(STL, Containers) { //NOLINT

  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  std::vector<nda::array<long, 2>> VV;
  VV.push_back(A);

  std::map<std::string, nda::array<long, 2>> MAP;
  MAP["1"] = A;
}

// ==============================================================

// STL algorithms works on nda::vector
TEST(STL, Vector) { //NOLINT

  std::vector<int> V(10);
  for (unsigned int i = 0; i < 10; ++i) V[i] = 10 + i;

  // Trying to put a vector in an array

  nda::array<int, 1> B(V.size()), C(V.size());
  // copy to B. Iterators on array are STL compliant so STL algorithms work.
  std::copy(V.begin(), V.end(), B.begin());

  // change B
  for (int i = 0; i < 10; ++i) B(i) *= 2;

  // come back !
  std::copy(B.begin(), B.end(), V.begin());

  for (unsigned int i = 0; i < 10; ++i) EXPECT_EQ(V[i], 2 * (10 + i));
}

// ==============================================================

TEST(STL, Algo1) { //NOLINT

  nda::array<int, 1> B{1, 34, 2, 6, 23, 189, 8, 4};

  auto te = [](int x) { return (x < 25); };

  EXPECT_EQ(std::count_if(B.begin(), B.end(), te), 6);
  EXPECT_EQ(*std::max_element(B.begin(), B.end()), 189);
  ///EXPECT_EQ(std::max_element(B.begin(), B.end()).indices()[0], 5);

  std::replace_if(B.begin(), B.end(), te, 0);

  EXPECT_EQ(B, (nda::array<int, 1>{0, 34, 0, 0, 0, 189, 0, 0}));
}

// ==============================================================

// NB Old bug, no issue
TEST(STL, Bugxxx) { //NOLINT

  {
    vector_t<double> a = {1, 3, 2}, b = {2, 3, 1};
    EXPECT_TRUE(std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()));
    EXPECT_FALSE(std::lexicographical_compare(b.begin(), b.end(), a.begin(), a.end()));
    //EXPECT_TRUE(a < b);
    //EXPECT_FALSE(b < a);
  }
  {
    vector_t<double> a = {1, 3, 2}, b = {1, 2, 10};
    EXPECT_FALSE(std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()));
    EXPECT_TRUE(std::lexicographical_compare(b.begin(), b.end(), a.begin(), a.end()));
    //EXPECT_TRUE(b < a);
    //EXPECT_FALSE(a < b);
  }
  //{
  //vector_t<int> a = {1, 3, 2}, b = a;
  //EXPECT_FALSE(a < b);
  //}
}

// ==============================================================

// random access iterator in d = 1
TEST(STL, RandomIteratorAndSort) { //NOLINT

  nda::array<int, 1> V(10);
  for (unsigned int i = 0; i < 10; ++i) V[i] = 10 - i;

  std::sort(V.begin(), V.end());

  for (unsigned int i = 0; i < 10; ++i) EXPECT_EQ(V[i], 1 + i);
}

// std::shuffle also require random access
// adapted from cppref example for it.
TEST(STL, RandomIteratorAndSort2) { //NOLINT

  nda::array<int, 1> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(v.begin(), v.end(), g);

  std::sort(v.begin(), v.end());

  for (unsigned int i = 0; i < 10; ++i) EXPECT_EQ(v[i], 1 + i);
}

// test it with a stride of 2
TEST(STL, RandomIteratorAndSortWithStride) { //NOLINT

  nda::array<int, 1> a(20);
  a      = -9;
  auto v = a(range(0, 20, 2));
  for (unsigned int i = 0; i < 10; ++i) v[i] = 1 + i;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(v.begin(), v.end(), g);

  std::sort(v.begin(), v.end());

  EXPECT_EQ((v.begin()[3]), 4);

  for (unsigned int i = 0; i < 10; ++i) {
    EXPECT_EQ(a[2 * i], 1 + i);
    EXPECT_EQ(a[2 * i + 1], -9);
  }
}
