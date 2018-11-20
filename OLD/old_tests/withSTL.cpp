
/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#include "start.hpp"
#include <type_traits>

#include <vector>
#include <map>
#include <algorithm>

// -----------------------------------------------

TEST(Array, STL_Vector) {
  array<long, 1> a{1, 2, 3};
  std::vector<array<long, 1>> VV;
  VV.push_back(a);
}

// -----------------------------------------------
TEST(Array, STL_Map) {
  array<long, 1> a{1, 2, 3};
  std::map<std::string, array<long, 1>> MAP;
  MAP["1"] = A;
}

// -----------------------------------------------
TEST(Array, STL_copy) {

  array<int, 1> B1{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  std::vector<int> V{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

  array<int, 1> B(V.size());

  std::copy(V.begin(), V.end(), B.begin());

  assert_all_close(B, B1, 1.e-15);
}

// -----------------------------------------------
TEST(Array, STL_copy2) {

  array<int, 1> B1{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  std::vector<int> V(10, 0);

  std::copy(B.begin(), B.end(), V.begin());

  for (int i = 0; i < 10; ++i) EXPECT_EQ(V[i], 10 + i);
}

// -----------------------------------------------
TEST(Array, STL_Countif) {

  array<int, 1> B{20, 22, 24, 26, 28, 30, 32, 34, 36, 38};

  auto c = std::count_if(B.begin(), B.end(), [](int x) { return x < 25; });
  EXPECT_EQ(c, 3);
}

// -----------------------------------------------
TEST(Array, STL_MaxElement) {

  array<int, 1> B{20, 22, 24, 26, 28, 30, 32, 34, 18, 0};

  // value of max
  EXPECT_EQ((*std::max_element(B.begin(), B.end())), 34);
  // position
  EXPECT_EQ((std::max_element(B.begin(), B.end()).indices()[0]), 7);
}

// -----------------------------------------------
TEST(Array, STL_ReplaceIf) {

  array<int, 1> B{20, 22, 24, 26, 28, 30, 32, 34, 36, 38};
  array<int, 1> R{20, 22, 24, 0, 0, 0, 0, 0, 0, 0};
  std::replace_if(B.begin(), B.end(), [](int x) { return x < 25; }, 0);
  assert_all_close(B, R, 1.e-15);
}

// -----------------------------------------------
TEST(Array, STL_Swap) {
  array<int, 1> a{1, 2, 3};
  array<int, 1> b{4, 5, 6};
  auto aa = a;
  auto bb = b;
  std::swap(a, b);
  assert_all_close(a, bb, 1.e-15);
  assert_all_close(b, aa, 1.e-15);
}

MAKE_MAIN;
