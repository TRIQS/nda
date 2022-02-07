// Copyright (c) 2019-2021 Simons Foundation
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

//static_assert(!std::is_pod<nda::array<long, 2>>::value, "POD pb");
static_assert(nda::is_scalar_for_v<int, matrix<std::complex<double>>> == 1, "oops");

// ==============================================================

TEST(NDA, Create1) { //NOLINT
  auto A = nda::array<long, 2>(3, 3);
  EXPECT_EQ(A.shape(), (nda::shape_t<2>{3, 3}));

  auto B = nda::array<long, 2>(std::array{3, 3});
  EXPECT_EQ(B.shape(), (nda::shape_t<2>{3, 3}));

  auto C = nda::array<long, 1>(3, 3);
  EXPECT_EQ(C.shape(), (nda::shape_t<1>{3}));
  EXPECT_EQ(C, (3 * nda::ones<long>(3)));
}

// -------------------------------------

TEST(Assign, Contiguous) { //NOLINT
  nda::array<long, 2> A{{1, 2}, {3, 4}, {5, 6}};
  nda::array<long, 2> B;
  B = A;

  EXPECT_ARRAY_NEAR(A, B);

  A(0, 1) = 87;
  B       = A(); // no resize

  EXPECT_EQ(A.indexmap().strides(), B.indexmap().strides());

  EXPECT_ARRAY_NEAR(A, B);
  EXPECT_EQ(B.shape(), (nda::shape_t<2>{3, 2}));
}

// -------------------------------------

TEST(Assign, Strided) { //NOLINT
  nda::array<long, 3> A(3, 5, 9);
  nda::array<long, 1> B;

  for (int i = 0; i < A.extent(0); ++i)
    for (int j = 0; j < A.extent(1); ++j)
      for (int k = 0; k < A.extent(2); ++k) A(i, j, k) = 1 + i + 10 * j + 100 * k;

  NDA_PRINT(A.shape());

  B = A(_, 0, 1);
  for (int i = 0; i < A.extent(0); ++i) EXPECT_EQ(A(i, 0, 1), B(i));

  B = A(1, _, 2);
  for (int i = 0; i < A.extent(1); ++i) EXPECT_EQ(A(1, i, 2), B(i));

  B = A(1, 3, _);

  for (int i = 0; i < A.extent(2); ++i) EXPECT_EQ(A(1, 3, i), B(i));

  // P =0 to force out of the first test of slice_layout_prop. We want to test the real algorithm
  EXPECT_EQ(
     nda::slice_static::slice_layout_prop(0, true, std::array<bool, 3>{1, 0, 0}, std::array<int, 3>{0, 1, 2}, nda::layout_prop_e::contiguous, 128, 0),
     nda::layout_prop_e::strided_1d);
  EXPECT_EQ(
     nda::slice_static::slice_layout_prop(0, true, std::array<bool, 3>{0, 1, 0}, std::array<int, 3>{0, 1, 2}, nda::layout_prop_e::contiguous, 128, 0),
     nda::layout_prop_e::strided_1d);
  EXPECT_EQ(
     nda::slice_static::slice_layout_prop(0, true, std::array<bool, 3>{0, 0, 1}, std::array<int, 3>{0, 1, 2}, nda::layout_prop_e::contiguous, 128, 0),
     nda::layout_prop_e::contiguous);

  static_assert(nda::get_layout_info<decltype(A(1, 3, _))>.prop == nda::layout_prop_e::contiguous);
  static_assert(nda::get_layout_info<decltype(A(1, _, 2))>.prop == nda::layout_prop_e::strided_1d);
  static_assert(nda::get_layout_info<decltype(A(_, 0, 1))>.prop == nda::layout_prop_e::strided_1d);
}

// -------------------------------------

TEST(Assign, Strided2) { //NOLINT
  nda::array<long, 3> A(3, 5, 9);
  nda::array<long, 1> B;

  for (int i = 0; i < A.extent(0); ++i)
    for (int j = 0; j < A.extent(1); ++j)
      for (int k = 0; k < A.extent(2); ++k) A(i, j, k) = 1 + i + 10 * j + 100 * k;

  B.resize(20);
  B = 0;

  static_assert(nda::get_layout_info<decltype(B(range(0, 2 * A.extent(0), 2)))>.prop == nda::layout_prop_e::strided_1d);

  B(range(0, 2 * A.extent(0), 2)) = A(_, 0, 1);
  NDA_PRINT(B);
  NDA_PRINT(A(_, 0, 1));

  for (int i = 0; i < A.extent(0); ++i) EXPECT_EQ(A(i, 0, 1), B(2 * i));

  B(range(0, 2 * A.extent(1), 2)) = A(1, _, 2);
  NDA_PRINT(B);
  NDA_PRINT(A(1, _, 2));

  for (int i = 0; i < A.extent(1); ++i) EXPECT_EQ(A(1, i, 2), B(2 * i));

  B(range(0, 2 * A.extent(2), 2)) = A(1, 3, _);
  NDA_PRINT(B);
  NDA_PRINT(A(1, 3, _));

  for (int i = 0; i < A.extent(2); ++i) EXPECT_EQ(A(1, 3, i), B(2 * i));
}

// -------------------------------------

TEST(NDA, Iterator1) { //NOLINT
  nda::array<long, 2> A{{0, 1, 2}, {3, 4, 5}};

  int i = 0;
  for (auto x : A) EXPECT_EQ(x, i++);
}

// -------------------------------------

TEST(NDA, CreateResize) { //NOLINT

  nda::array<long, 2> A;
  A.resize({3, 3});
  EXPECT_EQ(A.shape(), (nda::shape_t<2>{3, 3}));

  A.resize({4, 4});
  EXPECT_EQ(A.shape(), (nda::shape_t<2>{4, 4}));

  nda::array<double, 2> M;
  M.resize(3, 3);

  EXPECT_EQ(M.shape(), (nda::shape_t<2>{3, 3}));

  nda::array<long, 1> V;
  V.resize(10);

  EXPECT_EQ(V.shape(), (nda::shape_t<1>{10}));
}

// ==============================================================

TEST(NDA, InitList) { //NOLINT

  // 1d
  nda::array<double, 1> A = {1, 2, 3, 4};

  EXPECT_EQ(A.shape(), (nda::shape_t<1>{4}));

  for (int i = 0; i < 4; ++i) EXPECT_EQ(A(i), i + 1);

  // 2d
  nda::array<double, 2> B = {{1, 2}, {3, 4}, {5, 6}};

  EXPECT_EQ(B.shape(), (nda::shape_t<2>{3, 2}));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(B(i, j), j + 2 * i + 1);

  // 3d
  nda::array<double, 3> C = {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {{100, 200, 300, 400}, {500, 600, 700, 800}, {900, 1000, 1100, 1200}}};

  EXPECT_EQ(C.shape(), (nda::shape_t<3>{2, 3, 4}));
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) EXPECT_EQ(C(i, j, k), (i == 0 ? 1 : 100) * (k + 4 * j + 1));

  // matrix
  nda::matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
  EXPECT_EQ(M.shape(), (nda::shape_t<2>{3, 2}));

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(M(i, j), j + 2 * i + 1);
}

// ==============================================================

TEST(NDA, InitList2) { //NOLINT

  // testing more complex cases
  nda::array<std::array<double, 2>, 1> aa{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  EXPECT_EQ(aa(3), (std::array<double, 2>{1, 1}));

  nda::array<double, 1> a{1, 2, 3.2};
  EXPECT_EQ_ARRAY(a, (nda::array<double, 1>{1.0, 2.0, 3.2}));
}

// ==============================================================

TEST(NDA, MoveConstructor) { //NOLINT
  nda::array<double, 1> A(3);
  A() = 9;

  nda::array<double, 1> B(std::move(A));

  EXPECT_TRUE(A.empty());
  EXPECT_EQ(B.shape(), (nda::shape_t<1>{3}));
  for (int i = 0; i < 3; ++i) EXPECT_EQ(B(i), 9);
}

// ==============================================================

TEST(NDA, MoveAssignment) { //NOLINT

  nda::array<double, 1> A(3);
  A() = 9;

  nda::array<double, 1> B;
  B = std::move(A);

  EXPECT_TRUE(A.empty());
  EXPECT_EQ(B.shape(), (nda::shape_t<1>{3}));
  for (int i = 0; i < 3; ++i) EXPECT_EQ(B(i), 9);
}

// ===================== SWAP =========================================

TEST(Swap, StdSwap) { //NOLINT

  auto V = nda::array<long, 1>{3, 3, 3};
  auto W = nda::array<long, 1>{4, 4, 4, 4};

  std::swap(V, W);

  // THIS Should not compile : deleted function
  //auto VV = V(range(0, 2));
  //auto WW = W(range(0, 2));
  //std::swap(VV, WW);

  // V , W are swapped
  EXPECT_EQ(V, (nda::array<long, 1>{4, 4, 4, 4}));
  EXPECT_EQ(W, (nda::array<long, 1>{3, 3, 3}));
}

// ----------------------------------

TEST(Swap, SwapView) { //NOLINT

  auto V = nda::array<long, 1>{3, 3, 3};
  auto W = nda::array<long, 1>{4, 4, 4, 4};

  // swap the view, not the vectors. Views are pointers
  // FIXME should we keep this behaviour ?
  auto VV = V(range(0, 2));
  auto WW = W(range(0, 2));
  swap(VV, WW);

  // V, W unchanged
  EXPECT_EQ(V, (nda::array<long, 1>{3, 3, 3}));
  EXPECT_EQ(W, (nda::array<long, 1>{4, 4, 4, 4}));

  // VV, WW swapped
  EXPECT_EQ(WW, (nda::array<long, 1>{3, 3}));
  EXPECT_EQ(VV, (nda::array<long, 1>{4, 4}));
}

// ----------------------------------

// FIXME Rename as BLAS_SWAP (swap of blas). Only for vector of same size
TEST(Swap, DeepSwap) { //NOLINT
  auto V = nda::array<long, 1>{3, 3, 3};
  auto W = nda::array<long, 1>{4, 4, 4};

  deep_swap(V(), W());

  // V , W are swapped
  EXPECT_EQ(V, (nda::array<long, 1>{4, 4, 4}));
  EXPECT_EQ(W, (nda::array<long, 1>{3, 3, 3}));
}
// ----------------------------------

TEST(Swap, DeepSwapView) { //NOLINT
  auto V = nda::array<long, 1>{3, 3, 3};
  auto W = nda::array<long, 1>{4, 4, 4, 4};

  auto VV = V(range(0, 2));
  auto WW = W(range(0, 2));

  deep_swap(VV, WW);

  // VV, WW swapped
  EXPECT_EQ(WW, (nda::array<long, 1>{3, 3}));
  EXPECT_EQ(VV, (nda::array<long, 1>{4, 4}));

  // V, W changed
  EXPECT_EQ(V, (nda::array<long, 1>{4, 4, 3}));
  EXPECT_EQ(W, (nda::array<long, 1>{3, 3, 4, 4}));
}

// ==============================================================

TEST(NDA, Print) { //NOLINT
  nda::array<long, 2> A(2, 3), B;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  EXPECT_PRINT("\n[[0,1,2]\n [10,11,12]]", A);
}

// ===========   Cross construction  ===================================================

TEST(Array, CrossConstruct1) { //NOLINT
  nda::array<int, 1> Vi(3);
  Vi() = 3;
  nda::array<double, 1> Vd(Vi);
  EXPECT_ARRAY_NEAR(Vd, Vi);
}

// ------------------
TEST(Array, CrossConstruct2) { //NOLINT

  nda::array<long, 2> A(2, 3);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  std::vector<nda::array<long, 2>> V(3, A);

  std::vector<nda::array_view<long, 2>> W;
  for (auto &x : V) W.emplace_back(x);

  std::vector<nda::array_view<long, 2>> W2(W);

  for (int i = 1; i < 3; ++i) V[i] *= i;

  for (int i = 1; i < 3; ++i) EXPECT_ARRAY_NEAR(W2[i], i * A);
}

// ------------------

// check non ambiguity of resolution, solved by the check of value type in the constructor
struct A {};
struct B {};
std::ostream &operator<<(std::ostream &out, A) { return out; }
std::ostream &operator<<(std::ostream &out, B) { return out; }

int f1(nda::array<A, 1>) { return 1; }
int f1(nda::array<B, 1>) { return 2; }

TEST(Array, CrossConstruct3) { //NOLINT
  nda::array<A, 1> a(2);
  auto v = a();
  EXPECT_EQ(f1(v), 1);
}

// =============================================================

TEST(NDA, ConvertibleCR) { //NOLINT

  nda::array<double, 2> A(2, 2);
  nda::array<dcomplex, 2> B(2, 2);

  //A = B; // should not compile

  B = A;

  auto c = nda::array<dcomplex, 2>{A};

  // can convert an array of double to an array of complex
  static_assert(std::is_constructible_v<nda::array<dcomplex, 2>, nda::array<double, 2>>, "oops");

  // can not do the reverse !
  //static_assert(not std::is_constructible_v<nda::array<double, 2>, nda::array<dcomplex, 2>>, "oops");
}

// =============================================================

TEST(Assign, CrossStrideOrder) { //NOLINT

  // check that = is ok, specially in the contiguous case where we have linear optimisation
  // which should NOT be used in this case ...

  nda::array<long, 3> a(2, 3, 4);
  nda::array<long, 3, nda::F_layout> af(2, 3, 4);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) { a(i, j, k) = i + 10 * j + 100 * k; }

  af = a;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) { EXPECT_EQ(af(i, j, k), i + 10 * j + 100 * k); }
}
