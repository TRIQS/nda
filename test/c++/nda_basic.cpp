#include "./test_common.hpp"

static_assert(!std::is_pod<nda::array<long, 2>>::value, "POD pb");
//static_assert(is_scalar_for<int, matrix<std::complex<double>>>::type::value == 1, "oops");

// ==============================================================

TEST(NDA, Create1) { //NOLINT
  nda::array<long, 2> A(3, 3);
  EXPECT_EQ(A.shape(), (nda::shape_t<2>{3, 3}));

  std::cerr << A.indexmap() << std::endl;
}

// -------------------------------------

TEST(NDA, Assign) { //NOLINT
  nda::array<long, 2> A(3, 3);

  A() = 0;

  nda::array<long, 2> B;
  B = A;

  EXPECT_ARRAY_NEAR(A, B);
  A(0, 2) = 87;

  B = A(); // no resize

  EXPECT_EQ(A.indexmap().strides(), B.indexmap().strides());

  EXPECT_ARRAY_NEAR(A, B);
  EXPECT_EQ(B.shape(), (nda::shape_t<2>{3, 3}));
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

  // matrix
  nda::matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
  EXPECT_EQ(M.shape(), (nda::shape_t<2>{3, 2}));

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(M(i, j), j + 2 * i + 1);
}

// ==============================================================

TEST(NDA, MoveConstructor) { //NOLINT
  nda::array<double, 1> A(3);
  A() = 9;

  nda::array<double, 1> B(std::move(A));

  EXPECT_TRUE(A.is_empty());
  EXPECT_TRUE(A.size() != 0);
  EXPECT_EQ(B.shape(), (nda::shape_t<1>{3}));
  for (int i = 0; i < 3; ++i) EXPECT_EQ(B(i), 9);
}

// ==============================================================

TEST(NDA, MoveAssignment) { //NOLINT

  nda::array<double, 1> A(3);
  A() = 9;

  nda::array<double, 1> B;
  B = std::move(A);

  EXPECT_TRUE(A.is_empty());
  EXPECT_EQ(B.shape(), (nda::shape_t<1>{3}));
  for (int i = 0; i < 3; ++i) EXPECT_EQ(B(i), 9);
}
/*
// ===================== SWAP =========================================

TEST(NDA, Swap) { //NOLINT
  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4, 4};

  swap(V, W);

  // V , W are swapped
  EXPECT_EQ(V, (vector<double>{4, 4, 4, 4}));
  EXPECT_EQ(W, (vector<double>{3, 3, 3}));
}

// ----------------------------------

TEST(NDA, StdSwap) { // same are triqs swap for regular types //NOLINT

  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4, 4};

  std::swap(V, W);

  // THIS Should not compile : deleted function
  //auto VV = V(range(0, 2));
  //auto WW = W(range(0, 2));
  //std::swap(VV, WW);

  // V , W are swapped
  EXPECT_EQ(V, (vector<double>{4, 4, 4, 4}));
  EXPECT_EQ(W, (vector<double>{3, 3, 3}));
}

// ----------------------------------

TEST(NDA, SwapView) { //NOLINT

  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4, 4};

  // swap the view, not the vectors. Views are pointers
  // FIXME should we keep this behaviour ?
  auto VV = V(range(0, 2));
  auto WW = W(range(0, 2));
  swap(VV, WW);

  // V, W unchanged
  EXPECT_EQ(V, (vector<double>{3, 3, 3}));
  EXPECT_EQ(W, (vector<double>{4, 4, 4, 4}));

  // VV, WW swapped
  EXPECT_EQ(WW, (vector<double>{3, 3}));
  EXPECT_EQ(VV, (vector<double>{4, 4}));
}

// ----------------------------------

// FIXME Rename as BLAS_SWAP (swap of blas). Only for vector of same size
TEST(NDA, DeepSwap) { //NOLINT
  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4};

  deep_swap(V, W);

  // V , W are swapped
  EXPECT_EQ(V, (vector<double>{4, 4, 4}));
  EXPECT_EQ(W, (vector<double>{3, 3, 3}));
}
// ----------------------------------

TEST(NDA, DeepSwapView) { //NOLINT
  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4, 4};

  auto VV = V(range(0, 2));
  auto WW = W(range(0, 2));

  deep_swap(VV, WW);

  // VV, WW swapped
  EXPECT_EQ(WW, (vector<double>{3, 3}));
  EXPECT_EQ(VV, (vector<double>{4, 4}));
}
*/

// ==============================================================

TEST(NDA, Print) { //NOLINT
  nda::array<long, 2> A(2, 3), B;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  EXPECT_PRINT("\n[[0,1,2]\n [10,11,12]]", A);
}

// ==============================================================

TEST(NDA, Ellipsis) { //NOLINT
  nda::array<long, 3> A(2, 3, 4);
  A() = 7;

  EXPECT_ARRAY_NEAR(A(0, ___), A(0, _, _), 1.e-15);

  nda::array<long, 4> B(2, 3, 4, 5);
  B() = 8;

  EXPECT_ARRAY_NEAR(B(0, ___, 3), B(0, _, _, 3), 1.e-15);
  EXPECT_ARRAY_NEAR(B(0, ___, 2, 3), B(0, _, 2, 3), 1.e-15);
  EXPECT_ARRAY_NEAR(B(___, 2, 3), B(_, _, 2, 3), 1.e-15);
}

// ==============================================================

template <typename ArrayType>
auto sum0(ArrayType const &A) {
  nda::array<typename ArrayType::value_t, ArrayType::rank - 1> res = A(0, ___);
  for (size_t u = 1; u < A.shape()[0]; ++u) res += A(u, ___);
  return res;
}

TEST(NDA, Ellipsis2) { //NOLINT
  nda::array<double, 2> A(5, 2);
  A() = 2;
  nda::array<double, 3> B(5, 2, 3);
  B() = 3;
  EXPECT_ARRAY_NEAR(sum0(A), nda::array<double, 1>{10, 10}, 1.e-15);
  EXPECT_ARRAY_NEAR(sum0(B), nda::array<double, 2>{{15, 15, 15}, {15, 15, 15}}, 1.e-15);
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
// =============================================================

TEST(NDA, ConvertibleCR) { //NOLINT

  nda::array<double, 2> A(2, 2);
  nda::array<dcomplex, 2> B(2, 2);

  //A = B; // should not compile

  B = A;

  auto c = nda::array<dcomplex, 2>{A};

  // can convert an array of double to an array of complex
  static_assert(std::is_constructible_v<nda::array<dcomplex, 2>, nda::array<double, 2>>, "oops");

#ifndef __clang__
  // can not do the reverse !
  static_assert(not std::is_constructible_v<nda::array<double, 2>, nda::array<dcomplex, 2>>, "oops");
  // EXCEPT that clang REQUIRES is not enough to see this (not part of SFINAE). Test on gcc ...
#endif
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



