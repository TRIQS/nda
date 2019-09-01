#include "./test_common.hpp"

// ==============================================================

TEST(NDA, Negate_Array) {

  static_assert(nda::is_ndarray_v<nda::array<double, 1>>, "EEE");

  nda::array<double, 1> A{4, 2, 3}, B{0, 0, 0};

  B = -A;

  EXPECT_ARRAY_NEAR(B, (nda::array<double, 1>{-4, -2, -3}), 1.e-12);
}

// ----------------------------------------------------

TEST(NDA, ExprTemplateArray) {

  static_assert(nda::get_algebra<nda::array<int, 1>> == 'A', "oops");

  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::r_is_scalar == true, "oops");
  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::l_is_scalar == false, "oops");
  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::algebra == 'A', "oops");

  nda::array<int, 1> A(3), B(3), C;
  nda::array<double, 1> D;
  B() = 2;
  A() = 3;

  C = A + B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{5, 5, 5});

  C = A * B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{6, 6, 6});

  C = 2 * B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{4, 4, 4});

  C = 2 * B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{4, 4, 4});

  D = 2.3 * B;
  EXPECT_ARRAY_NEAR(D, nda::array<double, 1>{4.6, 4.6, 4.6});

  D = A + B / 1.2;
  EXPECT_ARRAY_NEAR(D, nda::array<double, 1>{4.66666666666667, 4.66666666666667, 4.66666666666667});

  //auto x = A + B + 2 * A;
  //EXPECT_PRINT("(([3,3,3] + [2,2,2]) + (2 * [3,3,3]))", x);

  C = A + 2 * A + 3 * A - 2 * A + A - A + A + A * 3 + A + A + A + A + A + A + A + A + A + A + A + A + A;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{63, 63, 63});
}

// ==============================================================

TEST(NDA, DeductionRule) {

 nda::array<int, 1> A(3), B(3);
 B() = 2;
 A() = 3;

 auto S = nda::array{A+B}; //
 static_assert(std::is_same_v<decltype(S), nda::array<int,1>>,"oops");
 EXPECT_ARRAY_NEAR(S, nda::array<int, 1>{5, 5, 5});

 auto NO = nda::array{1,2}; // Should not compile (yet). Init list
}

// ==============================================================

//TEST(NDA, Negate_Vector) {

//vector<double> A{4, 2, 3}, B{0, 0, 0};

//B = -A;

//EXPECT_EQ(B, (nda::array<double, 1>{-4, -2, -3}));
//}

//TEST(NDA, Negate_Matrix) {

//matrix<double> A{{1, 2}, {3, 4}}, B(2, 2);
//B() = 0;
//B   = -A;
//EXPECT_EQ(B, (matrix<double>{{-1, -2}, {-3, -4}}));
//}

// ==============================================================

//TEST(NDA, ExprTemplateMatrix) {

//matrix<long> A(2, 2), B(2, 2), C(2, 2);

//for (int i = 0; i < 2; ++i)
//for (int j = 0; j < 2; ++j) {
//A(i, j) = 10 * i + j;
//B(i, j) = i + 2 * j;
//}

////FIXME : does not compile. TO BE CLEANED when cleaning expression tempate
////EXPECT_EQ((A + 2 * B).shape(), (myshape_t<2>{2, 2}));

//C = A + 2 * B;

//for (int i = 0; i < 2; ++i)
//for (int j = 0; j < 2; ++j) EXPECT_EQ(C(i, j), A(i, j) + 2 * B(i, j));

//C = std::plus<matrix<long>>()(A, B);

//for (int i = 0; i < 2; ++i)
//for (int j = 0; j < 2; ++j) EXPECT_EQ(C(i, j), A(i, j) + B(i, j));

////
//EXPECT_EQ(make_matrix(2 * A), (matrix<long>{{0, 2}, {20, 22}}));
//EXPECT_EQ(make_matrix(A + 2), (matrix<long>{{2, 1}, {10, 13}}));
//EXPECT_EQ(make_matrix(1 + A), (matrix<long>{{1, 1}, {10, 12}}));
//}

//// ----------------------------------------------------

//TEST(NDA, ExprTemplateMatrixMult) {

//matrix<long> A(2, 2), B(2, 2), C(2, 2);

//for (int i = 0; i < 2; ++i)
//for (int j = 0; j < 2; ++j) {
//A(i, j) = 10 * i + j;
//B(i, j) = i + 2 * j;
//}

//// matrix multiplication
//matrix<double> Af(2, 2), Bf(2, 2), Cf(2, 2), id(2, 2);
//Af       = A;
//Bf       = B;
//Bf(0, 0) = 1;
//Cf()     = 0;
//id()     = 1;
//Cf       = Af * Bf;

//EXPECT_EQ(Cf, (matrix<long>{{1, 3}, {21, 53}}));
//EXPECT_EQ(matrix<double>(Af * Bf), (matrix<long>{{1, 3}, {21, 53}}));
//EXPECT_EQ(matrix<double>(Af * (Bf + Cf)), (matrix<long>{{22, 56}, {262, 666}}));

//// test division
//EXPECT_EQ(matrix<double>(2 / Af), (matrix<double>{{-2.2, 0.2}, {2.0, 0.0}}));
//EXPECT_EQ(matrix<double>(Af / 2), (matrix<double>{{0.0, 0.5}, {5.0, 5.5}}));

//EXPECT_ARRAY_NEAR(make_matrix(Af / Af), id, 1.e-14);
//}

// ----------------------------------------------------

//TEST(NDA, ExprTemplateVector) {

////  test the vector ops : scalar * vector, vector + vector, ...
//triqs::arrays::vector<double> V{1, 2, 3};
//triqs::arrays::vector<double> V2{10, 20, 30};

//auto VV = make_vector(V2 + 2.0 * V);
//for (int i = 0; i < 3; ++i) EXPECT_EQ(VV(i), V2(i) + 2 * V(i));
//}

MAKE_MAIN
