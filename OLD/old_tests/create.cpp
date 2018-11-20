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
static_assert(!std::is_pod<triqs::arrays::array<long, 2>>::value, "POD pb");

// -----------------------------------------------

TEST(Array, Create) {

  array<long, 2> A;
  A.resize(make_shape(3, 2));
  EXPECT_EQ(A.shape(), (mini_vector<size_t, 2>{3, 2}));

  matrix<double> M;
  M.resize(3, 3);

  EXPECT_EQ(M.shape(), (mini_vector<size_t, 2>{3, 3}));

  vector<double> V;
  V.resize(10);

  EXPECT_EQ(V.shape(), (mini_vector<size_t, 1>{10}));
}

// -----------------------------------------------

TEST(Array, InitList) {

  {
    array<double, 1> a = {1, 2, 3, 4};

    EXPECT_EQ(a.shape(), (mini_vector<size_t, 1>{4}));
    EXPECT_EQ(a(0), 1);
    EXPECT_EQ(a(1), 2);
    EXPECT_EQ(a(2), 3);
    EXPECT_EQ(a(3), 4);
  }

  {
    array<double, 2> b = {{1, 2}, {3, 4}, {5, 6}};

    EXPECT_EQ(b.shape(), (mini_vector<size_t, 2>{3, 2}));
    EXPECT_EQ(a(0, 0), 1);
    EXPECT_EQ(a(0, 1), 2);
    EXPECT_EQ(a(1, 0), 3);
    EXPECT_EQ(a(1, 1), 4);
    EXPECT_EQ(a(2, 0), 5);
    EXPECT_EQ(a(2, 1), 6);
  }

  {
    matrix<int> b = {{1, 2}, {3, 4}, {5, 6}};

    EXPECT_EQ(b.shape(), (mini_vector<size_t, 2>{3, 2}));
    EXPECT_EQ(a(0, 0), 1);
    EXPECT_EQ(a(0, 1), 2);
    EXPECT_EQ(a(1, 0), 3);
    EXPECT_EQ(a(1, 1), 4);
    EXPECT_EQ(a(2, 0), 5);
    EXPECT_EQ(a(2, 1), 6);
  }
}

// ----------------------------------------------------------

struct S {
  int x = 0, y = 0;
};

TEST(Array, NonNumeric1) {

  array<S, 2> A(2, 3);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = S{i, j};

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(A(i, j).x, i);
      EXPECT_EQ(A(i, j).y, j);
    }

  A() = S{1, 2};

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(A(i, j).x, 1);
      EXPECT_EQ(A(i, j).y, 2);
    }
}

TEST(Array, ArrayOfArray) {

  array<array<double, 1>, 2> AA(2, 3);

  array<double, 1> A0{0, 1} AA() = A0;

  EXPECT_ARRAY_NEAR(A(0, 0), A0, 1.e-15);
}

// --------------------------------------------

TEST(Array, MoveConstruct) {
  array<int, 1> A{1, 2, 3};
  array<int, 1> Aref{1, 2, 3};

  array<int, 1> B(std::move(A));

  EXPECT_TRUE(A.is_empty());
  EXPECT_EQ(B, Aref);
}

TEST(Array, MoveAssign) {
  array<int, 1> A{1, 2, 3};
  array<int, 1> Aref{1, 2, 3};

  array<double, 1> B;

  EXPECT_TRUE(A.is_empty());
  EXPECT_EQ(B, Aref);
}

// -----------------------------------------------------------

// check that ellipsis is the same as range, range ....
TEST(Array, Ellipsis) {

  auto _   = range{};
  auto ___ = ellipsis{};

  array<long, 3> a(2, 3, 4);
  array<long, 4> B(2, 3, 4, 5);
  int u = 0;
  for (auto &x : a) x = u++; // fill with different numbers
  for (auto &x : b) x = u++; // fill with different numbers

  assert_all_close(a(0, ___), a(0, _, _));
  assert_all_close(a(1, ___), a(1, _, _));

  assert_all_close(B(0, ___, 3), B(0, _, _, 3), 1.e-15);
  assert_all_close(B(0, ___, 2, 3), B(0, _, 2, 3), 1.e-15);
  assert_all_close(B(___, 2, 3), B(_, _, 2, 3), 1.e-15);
}

MAKE_MAIN;
