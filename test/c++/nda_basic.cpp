#include "./nda_test_common.hpp"

static_assert(!std::is_pod<nda::array<long, 2>>::value, "POD pb");
//static_assert(is_scalar_for<int, matrix<std::complex<double>>>::type::value == 1, "oops");

// ==============================================================

TEST(NDA, Create1) {
  nda::array<long, 2> A(3, 3);
  EXPECT_EQ(A.shape(), (myshape_t<2>{3, 3}));
}

// -------------------------------------

TEST(NDA, View) {
  nda::array<long, 3> a(3, 3, 4);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) a(i, j, k) = i + 10 * j + 100 * k;

  auto v = a(_, 1, 2);

  EXPECT_EQ(v.shape(), (myshape_t<1>{3}));

  EXPECT_EQ(a(1, 1, 2), 1 + 10 * 1 + 100 * 2);
  
  a(1,1,2) = -28;
  EXPECT_EQ(v(1), a(1, 1, 2));
}

// -------------------------------------

TEST(NDA, Assign) {
  nda::array<long, 2> A(3, 3);

  A()=0; 
 
  nda::array<long, 2> B;
  B = A;
  
  EXPECT_ARRAY_NEAR(A,B);
  A(0, 2) = 87;

  EXPECT_EQ(A.indexmap().layout() , B.indexmap().layout());
   
  B = A(); // no resize
  
  EXPECT_EQ(A.indexmap().layout() , B.indexmap().layout());
  EXPECT_EQ(A.indexmap().strides() , B.indexmap().strides());

  EXPECT_ARRAY_NEAR(A,B);
  EXPECT_EQ(B.shape(), (myshape_t<2>{3, 3}));
}



/*
// -------------------------------------

TEST(NDA, CreateResize) {

  array<long, 2> A;
  A.resize(make_shape(3, 3));
  EXPECT_EQ(A.shape(), (myshape_t<2>{3, 3}));

  matrix<double> M;
  M.resize(3, 3);

  EXPECT_EQ(M.shape(), (myshape_t<2>{3, 3}));

  vector<double> V;
  V.resize(10);

  EXPECT_EQ(V.shape(), (myshape_t<1>{10}));
}

// ==============================================================

TEST(NDA, InitList) {

  // 1d
  array<double, 1> A = {1, 2, 3, 4};

  EXPECT_EQ(A.shape(), (myshape_t<1>{4}));

  for (int i = 0; i < 4; ++i) EXPECT_EQ(A(i), i + 1);

  // 2d
  array<double, 2> B = {{1, 2}, {3, 4}, {5, 6}};

  EXPECT_EQ(B.shape(), (myshape_t<2>{3, 2}));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(B(i, j), j + 2 * i + 1);

  // matrix
  matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
  EXPECT_EQ(M.shape(), (myshape_t<2>{3, 2}));

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(M(i, j), j + 2 * i + 1);
}

// ==============================================================

TEST(NDA, MoveConstructor) {
  array<double, 1> A(3);
  A() = 9;

  array<double, 1> B(std::move(A));

  EXPECT_TRUE(A.is_empty());
  EXPECT_EQ(B.shape(), (myshape_t<1>{3}));
  for (int i = 0; i < 3; ++i) EXPECT_EQ(B(i), 9);
}

// ==============================================================

TEST(NDA, MoveAssignment) {

  array<double, 1> A(3);
  A() = 9;

  array<double, 1> B;
  B = std::move(A);

  EXPECT_TRUE(A.is_empty());
  EXPECT_EQ(B.shape(), (myshape_t<1>{3}));
  for (int i = 0; i < 3; ++i) EXPECT_EQ(B(i), 9);
}

// ===================== SWAP =========================================

TEST(NDA, Swap) {
  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4, 4};

  swap(V, W);

  // V , W are swapped
  EXPECT_EQ(V, (vector<double>{4, 4, 4, 4}));
  EXPECT_EQ(W, (vector<double>{3, 3, 3}));
}

// ----------------------------------

TEST(NDA, StdSwap) { // same are triqs swap for regular types

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

TEST(NDA, SwapView) {

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
TEST(NDA, DeepSwap) {
  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4};

  deep_swap(V, W);

  // V , W are swapped
  EXPECT_EQ(V, (vector<double>{4, 4, 4}));
  EXPECT_EQ(W, (vector<double>{3, 3, 3}));
}
// ----------------------------------

TEST(NDA, DeepSwapView) {
  auto V = vector<double>{3, 3, 3};
  auto W = vector<double>{4, 4, 4, 4};

  auto VV = V(range(0, 2));
  auto WW = W(range(0, 2));

  deep_swap(VV, WW);

  // VV, WW swapped
  EXPECT_EQ(WW, (vector<double>{3, 3}));
  EXPECT_EQ(VV, (vector<double>{4, 4}));
}

// ==============================================================

TEST(NDA, Print) {
  array<long, 2> A(2, 3), B;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  EXPECT_PRINT("\n[[0,1,2]\n [10,11,12]]", A);
}

// ==============================================================

TEST(NDA, Ellipsis) {
  array<long, 3> A(2, 3, 4);
  A() = 7;

  assert_all_close(A(0, ___), A(0, _, _), 1.e-15);

  array<long, 4> B(2, 3, 4, 5);
  B() = 8;

  assert_all_close(B(0, ___, 3), B(0, _, _, 3), 1.e-15);
  assert_all_close(B(0, ___, 2, 3), B(0, _, 2, 3), 1.e-15);
  assert_all_close(B(___, 2, 3), B(_, _, 2, 3), 1.e-15);
}

// ==============================================================

template <typename ArrayType> auto sum0(ArrayType const &A) {
  array<typename ArrayType::value_type, ArrayType::rank - 1> res = A(0, ___);
  for (size_t u = 1; u < A.shape()[0]; ++u) res += A(u, ___);
  return res;
}

TEST(NDA, Ellipsis2) {
  array<double, 2> A(5, 2);
  A() = 2;
  array<double, 3> B(5, 2, 3);
  B() = 3;
  assert_all_close(sum0(A), array<double, 1>{10, 10}, 1.e-15);
  assert_all_close(sum0(B), array<double, 2>{{15, 15, 15}, {15, 15, 15}}, 1.e-15);
}

// ==============================================================

TEST(NDA, AssignVectorArray) {

  vector<double> V;
  array<double, 1> Va(5);
  for (int i = 0; i < 5; ++i) Va(i) = i + 2;

  V = Va / 2.0;
  EXPECT_ARRAY_NEAR(V, array<double, 1>{1.0, 1.5, 2.0, 2.5, 3.0});
  EXPECT_ARRAY_NEAR(Va, array<double, 1>{2, 3, 4, 5, 6});

  V = Va;
  EXPECT_ARRAY_NEAR(V, array<double, 1>{2, 3, 4, 5, 6});
  EXPECT_ARRAY_NEAR(Va, array<double, 1>{2, 3, 4, 5, 6});
}

// ===========   Cross construction  ===================================================

TEST(Array, CrossConstruct1) {
  vector<int> Vi(3);
  Vi() = 3;
  vector<double> Vd(Vi);
  EXPECT_ARRAY_NEAR(Vd, Vi);
}

// ------------------
TEST(Array, CrossConstruct2) {

  array<long, 2> A(2, 3);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  std::vector<array<long, 2>> V(3, A);

  std::vector<array_view<long, 2>> W;
  for (auto &x : V) W.push_back(x);

  std::vector<array_view<long, 2>> W2(W);

  for (int i = 1; i < 3; ++i) V[i] *= i;

  for (int i = 1; i < 3; ++i) EXPECT_ARRAY_NEAR(W2[i], i * A);
}
// =============================================================

TEST(NDA, ConvertibleCR) {

  array<double, 2> A(2, 2);
  array<dcomplex, 2> B(2, 2);

  //A = B; // should not compile
  B = A;

  // can convert an array of double to an array of complex
  static_assert(std::is_convertible<array<double, 2>, array<dcomplex, 2>>::value, "oops");

  // can not do the reverse !
  static_assert(!std::is_convertible<array<dcomplex, 2>, array<double, 2>>::value, "oops");
}
*/
MAKE_MAIN
