// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  nda::array<double, 2> make_arr2(long n0 = 5, long n1 = 4) {
    nda::array<double, 2> arr(n0, n1);
    for (long i = 0; i < n0; ++i)
      for (long j = 0; j < n1; ++j) arr(i, j) = static_cast<double>(i * 10 + j);
    return arr;
  }

  nda::array<double, 3> make_arr3() {
    nda::array<double, 3> arr(4, 5, 6);
    for (long i = 0; i < 4; ++i)
      for (long j = 0; j < 5; ++j)
        for (long k = 0; k < 6; ++k) arr(i, j, k) = static_cast<double>(i * 100 + j * 10 + k);
    return arr;
  }

} // namespace

// ============================================================
// Basic IndexContainer concept tests
// ============================================================

TEST(ExprIndexed, IndexContainerConcept) {
  // Types that should satisfy IndexContainer
  static_assert(nda::IndexContainer<std::vector<long>>);
  static_assert(nda::IndexContainer<std::vector<int>>);
  static_assert(nda::IndexContainer<std::array<long, 3>>);
  static_assert(nda::IndexContainer<std::span<const long>>);

  // Types that should NOT satisfy IndexContainer
  static_assert(!nda::IndexContainer<long>);
  static_assert(!nda::IndexContainer<int>);
  static_assert(!nda::IndexContainer<nda::range>);
  static_assert(!nda::IndexContainer<nda::range::all_t>);
  static_assert(!nda::IndexContainer<nda::ellipsis>);

  // Ranges of non-integer, bool or character values are not index containers
  static_assert(!nda::IndexContainer<std::vector<double>>);
  static_assert(!nda::IndexContainer<std::vector<bool>>);
  static_assert(!nda::IndexContainer<std::string>);

  // nda arrays are index containers only if they have rank 1
  static_assert(nda::IndexContainer<nda::array<long, 1>>);
  static_assert(!nda::IndexContainer<nda::array<long, 2>>);
}

TEST(ExprIndexed, ArrayConcept) {
  // expr_indexed should satisfy the Array concept
  nda::array<double, 2> arr(5, 4);
  std::vector<long> indices = {2, 0, 4};
  auto expr                 = arr(indices, nda::range::all);

  static_assert(nda::Array<decltype(expr)>);
  static_assert(nda::ArrayOfRank<decltype(expr), 2>);
}

// ============================================================
// 1D array indexing
// ============================================================

TEST(ExprIndexed, Basic1D) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i * i);

  std::vector<long> indices = {3, 7, 1, 5};

  auto expr = arr(indices);

  // Check shape
  EXPECT_EQ(expr.shape(), (std::array<long, 1>{4}));
  EXPECT_EQ(expr.size(), 4);

  // Check element access
  EXPECT_EQ(expr(0), arr(3)); // 9
  EXPECT_EQ(expr(1), arr(7)); // 49
  EXPECT_EQ(expr(2), arr(1)); // 1
  EXPECT_EQ(expr(3), arr(5)); // 25

  // Materialize to array
  nda::array<double, 1> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 1>{4}));
  EXPECT_EQ(result(0), 9.0);
  EXPECT_EQ(result(1), 49.0);
  EXPECT_EQ(result(2), 1.0);
  EXPECT_EQ(result(3), 25.0);
}

// ============================================================
// 2D array indexing - single dimension
// ============================================================

TEST(ExprIndexed, Basic2D_IndexFirstDim) {
  auto arr = make_arr2();

  std::vector<long> indices = {2, 0, 4};

  auto expr = arr(indices, nda::range::all);

  // Check shape: 3 rows (from indices), 4 columns (all)
  EXPECT_EQ(expr.shape(), (std::array<long, 2>{3, 4}));

  // Check element access
  EXPECT_EQ(expr(0, 0), arr(2, 0)); // 20
  EXPECT_EQ(expr(0, 3), arr(2, 3)); // 23
  EXPECT_EQ(expr(1, 0), arr(0, 0)); // 0
  EXPECT_EQ(expr(2, 2), arr(4, 2)); // 42

  // Materialize
  nda::array<double, 2> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 2>{3, 4}));
  EXPECT_EQ(result(0, 0), 20.0);
  EXPECT_EQ(result(1, 1), 1.0);
  EXPECT_EQ(result(2, 3), 43.0);
}

TEST(ExprIndexed, Basic2D_IndexSecondDim) {
  auto arr = make_arr2();

  std::vector<long> indices = {3, 1};

  auto expr = arr(nda::range::all, indices);

  // Check shape: 5 rows (all), 2 columns (from indices)
  EXPECT_EQ(expr.shape(), (std::array<long, 2>{5, 2}));

  // Check element access
  EXPECT_EQ(expr(0, 0), arr(0, 3)); // 3
  EXPECT_EQ(expr(0, 1), arr(0, 1)); // 1
  EXPECT_EQ(expr(4, 0), arr(4, 3)); // 43
  EXPECT_EQ(expr(4, 1), arr(4, 1)); // 41

  // Materialize
  nda::array<double, 2> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 2>{5, 2}));
}

// ============================================================
// 2D array indexing - both dimensions
// ============================================================

TEST(ExprIndexed, Basic2D_IndexBothDims) {
  auto arr = make_arr2();

  std::vector<long> idx1 = {2, 0};
  std::vector<long> idx2 = {3, 1, 0};

  auto expr = arr(idx1, idx2);

  // Check shape: Cartesian product 2 x 3
  EXPECT_EQ(expr.shape(), (std::array<long, 2>{2, 3}));

  // Check element access: expr(i, j) = arr(idx1[i], idx2[j])
  EXPECT_EQ(expr(0, 0), arr(2, 3)); // 23
  EXPECT_EQ(expr(0, 1), arr(2, 1)); // 21
  EXPECT_EQ(expr(0, 2), arr(2, 0)); // 20
  EXPECT_EQ(expr(1, 0), arr(0, 3)); // 3
  EXPECT_EQ(expr(1, 1), arr(0, 1)); // 1
  EXPECT_EQ(expr(1, 2), arr(0, 0)); // 0

  // Materialize
  nda::array<double, 2> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 2>{2, 3}));
  EXPECT_EQ(result(0, 0), 23.0);
  EXPECT_EQ(result(1, 2), 0.0);
}

// ============================================================
// Mixed indexing with ranges
// ============================================================

TEST(ExprIndexed, MixedWithRange) {
  auto arr = make_arr2(10, 8);

  std::vector<long> indices = {5, 2, 8};

  // Index first dim, range on second dim
  auto expr = arr(indices, nda::range(2, 6));

  // Check shape: 3 rows, 4 columns
  EXPECT_EQ(expr.shape(), (std::array<long, 2>{3, 4}));

  // Check element access
  EXPECT_EQ(expr(0, 0), arr(5, 2)); // 52
  EXPECT_EQ(expr(0, 3), arr(5, 5)); // 55
  EXPECT_EQ(expr(2, 0), arr(8, 2)); // 82
}

// ============================================================
// 3D array indexing
// ============================================================

TEST(ExprIndexed, Basic3D) {
  auto arr = make_arr3();

  std::vector<long> indices = {1, 3};

  auto expr = arr(indices, nda::range::all, nda::range::all);

  // Check shape: 2 x 5 x 6
  EXPECT_EQ(expr.shape(), (std::array<long, 3>{2, 5, 6}));

  // Check element access
  EXPECT_EQ(expr(0, 0, 0), arr(1, 0, 0)); // 100
  EXPECT_EQ(expr(1, 2, 3), arr(3, 2, 3)); // 323
}

// ============================================================
// Ellipsis support
// ============================================================

TEST(ExprIndexed, WithEllipsis) {
  auto arr = make_arr3();

  std::vector<long> indices = {1, 3};

  // Index first dim, ellipsis for rest
  auto expr = arr(indices, nda::ellipsis{});

  // Check shape: 2 x 5 x 6
  EXPECT_EQ(expr.shape(), (std::array<long, 3>{2, 5, 6}));

  // Check element access
  EXPECT_EQ(expr(0, 2, 3), arr(1, 2, 3)); // 123
  EXPECT_EQ(expr(1, 4, 5), arr(3, 4, 5)); // 345
}

// ============================================================
// Different container types
// ============================================================

TEST(ExprIndexed, StdArrayContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::array<long, 3> indices = {5, 2, 8};

  auto expr   = arr(indices);
  auto result = nda::array<double, 1>{expr};

  EXPECT_EQ(result(0), 5.0);
  EXPECT_EQ(result(1), 2.0);
  EXPECT_EQ(result(2), 8.0);
}

TEST(ExprIndexed, SpanContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::vector<long> idx_vec = {5, 2, 8};
  std::span<const long> indices(idx_vec);

  auto expr   = arr(indices);
  auto result = nda::array<double, 1>{expr};

  EXPECT_EQ(result(0), 5.0);
  EXPECT_EQ(result(1), 2.0);
  EXPECT_EQ(result(2), 8.0);
}

TEST(ExprIndexed, IntVectorContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  // int should be convertible to long
  std::vector<int> indices = {5, 2, 8};

  auto expr   = arr(indices);
  auto result = nda::array<double, 1>{expr};

  EXPECT_EQ(result(0), 5.0);
  EXPECT_EQ(result(1), 2.0);
  EXPECT_EQ(result(2), 8.0);
}

// ============================================================
// Expression composition
// ============================================================

TEST(ExprIndexed, ArithmeticComposition) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::vector<long> indices = {3, 7, 1};

  // Compose with scalar multiplication
  auto expr                    = 2.0 * arr(indices);
  nda::array<double, 1> result = expr;

  EXPECT_EQ(result(0), 6.0);  // 2 * 3
  EXPECT_EQ(result(1), 14.0); // 2 * 7
  EXPECT_EQ(result(2), 2.0);  // 2 * 1
}

TEST(ExprIndexed, TwoExprIndexedComposition) {
  nda::array<double, 1> arr1(10);
  nda::array<double, 1> arr2(10);
  for (long i = 0; i < 10; ++i) {
    arr1(i) = static_cast<double>(i);
    arr2(i) = static_cast<double>(i * 2);
  }

  std::vector<long> indices = {3, 7, 1};

  // Add two indexed expressions
  auto expr                    = arr1(indices) + arr2(indices);
  nda::array<double, 1> result = expr;

  EXPECT_EQ(result(0), 3.0 + 6.0);  // arr1[3] + arr2[3]
  EXPECT_EQ(result(1), 7.0 + 14.0); // arr1[7] + arr2[7]
  EXPECT_EQ(result(2), 1.0 + 2.0);  // arr1[1] + arr2[1]
}

// ============================================================
// Matrix algebra preservation
// ============================================================

TEST(ExprIndexed, MatrixAlgebra) {
  nda::matrix<double> mat(5, 4);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) mat(i, j) = static_cast<double>(i * 10 + j);

  std::vector<long> indices = {2, 0, 4};

  auto expr = mat(indices, nda::range::all);

  // Should inherit matrix algebra
  static_assert(nda::get_algebra<decltype(expr)> == 'M');

  // Materialize as matrix
  nda::matrix<double> result = expr;
  EXPECT_EQ(result.shape(), (std::array<long, 2>{3, 4}));
}

// ============================================================
// Edge cases
// ============================================================

TEST(ExprIndexed, EmptyIndexContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::vector<long> empty_indices = {};

  auto expr = arr(empty_indices);

  // Shape should have zero in the indexed dimension
  EXPECT_EQ(expr.shape(), (std::array<long, 1>{0}));
  EXPECT_EQ(expr.size(), 0);

  // Materialization should produce empty array
  nda::array<double, 1> result = expr;
  EXPECT_EQ(result.size(), 0);
}

TEST(ExprIndexed, SingleElementContainer) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i * i);

  std::vector<long> indices = {5};

  auto expr = arr(indices);

  EXPECT_EQ(expr.shape(), (std::array<long, 1>{1}));
  EXPECT_EQ(expr.size(), 1);
  EXPECT_EQ(expr(0), 25.0); // 5^2
}

TEST(ExprIndexed, DuplicateIndices) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  // Same index repeated multiple times
  std::vector<long> indices = {3, 3, 3, 3};

  auto expr = arr(indices);

  EXPECT_EQ(expr.shape(), (std::array<long, 1>{4}));

  // All elements should be arr(3) = 3.0
  EXPECT_EQ(expr(0), 3.0);
  EXPECT_EQ(expr(1), 3.0);
  EXPECT_EQ(expr(2), 3.0);
  EXPECT_EQ(expr(3), 3.0);
}

TEST(ExprIndexed, ReversedIndices) {
  nda::array<double, 1> arr(5);
  for (long i = 0; i < 5; ++i) arr(i) = static_cast<double>(i);

  // Reverse order
  std::vector<long> indices = {4, 3, 2, 1, 0};

  auto expr                    = arr(indices);
  nda::array<double, 1> result = expr;

  EXPECT_EQ(result(0), 4.0);
  EXPECT_EQ(result(1), 3.0);
  EXPECT_EQ(result(2), 2.0);
  EXPECT_EQ(result(3), 1.0);
  EXPECT_EQ(result(4), 0.0);
}

// ============================================================
// Write access through expr_indexed
// ============================================================

TEST(ExprIndexed, WriteAccess1D) {
  nda::array<double, 1> arr(10);
  for (long i = 0; i < 10; ++i) arr(i) = static_cast<double>(i);

  std::vector<long> indices = {3, 7, 1};

  auto expr = arr(indices);

  // Write through expr_indexed
  expr(0) = 100.0;
  expr(1) = 200.0;

  // Verify writes went to underlying array
  EXPECT_EQ(arr(3), 100.0);
  EXPECT_EQ(arr(7), 200.0);
  EXPECT_EQ(arr(1), 1.0); // Unchanged
}

TEST(ExprIndexed, WriteAccess2D) {
  auto arr = nda::zeros<double>(5, 4);

  std::vector<long> indices = {2, 0, 4};

  auto expr = arr(indices, nda::range::all);

  expr(0, 0) = 42.0;
  expr(1, 3) = 99.0;

  EXPECT_EQ(arr(2, 0), 42.0);
  EXPECT_EQ(arr(0, 3), 99.0);
}

TEST(ExprIndexed, WriteAccessDuplicateIndices) {
  auto arr = nda::zeros<double>(10);

  // Same index repeated - writes to same location
  std::vector<long> indices = {3, 3, 3};

  auto expr = arr(indices);

  expr(0) = 10.0;
  EXPECT_EQ(arr(3), 10.0);

  expr(1) = 20.0;
  EXPECT_EQ(arr(3), 20.0);

  expr(2) = 30.0;
  EXPECT_EQ(arr(3), 30.0);
}

// ============================================================
// Direct assignment to array
// ============================================================

TEST(ExprIndexed, DirectAssignment) {
  auto arr = make_arr2(10, 8);

  std::vector<long> indices = {5, 2, 8, 1};

  // Direct assignment to new array
  nda::array<double, 2> result = arr(indices, nda::range::all);

  EXPECT_EQ(result.shape(), (std::array<long, 2>{4, 8}));

  // Check values
  EXPECT_EQ(result(0, 0), arr(5, 0)); // 50
  EXPECT_EQ(result(1, 3), arr(2, 3)); // 23
  EXPECT_EQ(result(2, 7), arr(8, 7)); // 87
  EXPECT_EQ(result(3, 0), arr(1, 0)); // 10
}

// ============================================================
// Assignment to expr_indexed from arrays/scalars
// ============================================================

TEST(ExprIndexed, AssignFromArray1D) {
  auto arr = nda::zeros<double>(10);

  std::vector<long> indices = {3, 7, 1};

  nda::array<double, 1> src = {100.0, 200.0, 300.0};

  arr(indices) = src;

  EXPECT_EQ(arr(3), 100.0);
  EXPECT_EQ(arr(7), 200.0);
  EXPECT_EQ(arr(1), 300.0);
  EXPECT_EQ(arr(0), 0.0); // Unchanged
}

TEST(ExprIndexed, AssignFromArray2D) {
  auto arr = nda::zeros<double>(5, 4);

  std::vector<long> indices = {2, 0};

  nda::array<double, 2> src(2, 4);
  for (long i = 0; i < 2; ++i)
    for (long j = 0; j < 4; ++j) src(i, j) = static_cast<double>((i + 1) * 10 + j);

  arr(indices, nda::range::all) = src;

  // Row 2 should have 10, 11, 12, 13
  // Row 0 should have 20, 21, 22, 23
  EXPECT_EQ(arr(2, 0), 10.0);
  EXPECT_EQ(arr(2, 3), 13.0);
  EXPECT_EQ(arr(0, 0), 20.0);
  EXPECT_EQ(arr(0, 3), 23.0);
}

TEST(ExprIndexed, AssignFromScalar) {
  auto arr = nda::zeros<double>(10);

  std::vector<long> indices = {3, 7, 1};

  arr(indices) = 42.0;

  EXPECT_EQ(arr(3), 42.0);
  EXPECT_EQ(arr(7), 42.0);
  EXPECT_EQ(arr(1), 42.0);
  EXPECT_EQ(arr(0), 0.0); // Unchanged
}

TEST(ExprIndexed, AssignFromExpression) {
  auto arr = nda::zeros<double>(10);

  nda::array<double, 1> src = {1.0, 2.0, 3.0};

  std::vector<long> indices = {3, 7, 1};

  arr(indices) = 2.0 * src;

  EXPECT_EQ(arr(3), 2.0);
  EXPECT_EQ(arr(7), 4.0);
  EXPECT_EQ(arr(1), 6.0);
}

// ============================================================
// Slicing expr_indexed
// ============================================================

TEST(ExprIndexed, SliceNonIndexedDimWithRangeAll) {
  auto arr = make_arr2();

  std::vector<long> indices = {2, 0, 4};

  auto expr = arr(indices, nda::range::all);

  // Slice the non-indexed dimension with range::all (should keep same shape)
  auto sliced = expr(nda::range::all, nda::range::all);

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{3, 4}));
  EXPECT_EQ(sliced(0, 0), arr(2, 0)); // 20
  EXPECT_EQ(sliced(1, 3), arr(0, 3)); // 3
}

TEST(ExprIndexed, SliceNonIndexedDimWithRange) {
  auto arr = make_arr2(5, 8);

  std::vector<long> indices = {2, 0, 4};

  auto expr = arr(indices, nda::range::all);

  // Slice the non-indexed dimension (columns 2-5)
  auto sliced = expr(nda::range::all, nda::range(2, 6));

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{3, 4}));
  EXPECT_EQ(sliced(0, 0), arr(2, 2)); // 22
  EXPECT_EQ(sliced(0, 3), arr(2, 5)); // 25
  EXPECT_EQ(sliced(2, 0), arr(4, 2)); // 42
}

TEST(ExprIndexed, SliceIndexedDimWithRangeAll) {
  auto arr = make_arr2();

  std::vector<long> indices = {2, 0, 4, 1};

  auto expr = arr(indices, nda::range::all);

  // Slice the indexed dimension with range::all (should keep same shape)
  auto sliced = expr(nda::range::all, nda::range::all);

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{4, 4}));
  EXPECT_EQ(sliced(0, 0), arr(2, 0)); // 20
  EXPECT_EQ(sliced(3, 0), arr(1, 0)); // 10
}

TEST(ExprIndexed, SliceIndexedDimWithRange) {
  auto arr = make_arr2();

  std::vector<long> indices = {2, 0, 4, 1};

  auto expr = arr(indices, nda::range::all);

  // Slice the indexed dimension (select first 2 from index container)
  auto sliced = expr(nda::range(0, 2), nda::range::all);

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{2, 4}));
  EXPECT_EQ(sliced(0, 0), arr(2, 0)); // indices[0] = 2, so arr(2, 0) = 20
  EXPECT_EQ(sliced(1, 0), arr(0, 0)); // indices[1] = 0, so arr(0, 0) = 0
}

TEST(ExprIndexed, SliceBothDimsWithRange) {
  auto arr = make_arr2(10, 8);

  std::vector<long> indices = {5, 2, 8, 1, 9};

  auto expr = arr(indices, nda::range::all);

  // Slice both dims
  auto sliced = expr(nda::range(1, 4), nda::range(2, 6));

  EXPECT_EQ(sliced.shape(), (std::array<long, 2>{3, 4}));
  // sliced(0,0) = expr(1, 2) = arr(indices[1], 2) = arr(2, 2) = 22
  EXPECT_EQ(sliced(0, 0), arr(2, 2));
  // sliced(2,3) = expr(3, 5) = arr(indices[3], 5) = arr(1, 5) = 15
  EXPECT_EQ(sliced(2, 3), arr(1, 5));
}

// ============================================================
// Construction with long and ellipsis arguments
// ============================================================

TEST(ExprIndexed, ConstructWithLong) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {3, 1};

  auto e1 = arr(2, indices);
  static_assert(decltype(e1)::rank == 1);
  EXPECT_EQ(e1.shape(), (std::array<long, 1>{2}));
  EXPECT_EQ(e1(0), arr(2, 3));
  EXPECT_EQ(e1(1), arr(2, 1));

  auto e2 = arr(indices, 2);
  static_assert(decltype(e2)::rank == 1);
  EXPECT_EQ(e2.shape(), (std::array<long, 1>{2}));
  EXPECT_EQ(e2(0), arr(3, 2));
  EXPECT_EQ(e2(1), arr(1, 2));
}

TEST(ExprIndexed, ConstructWithLeadingEllipsis) {
  auto arr                  = make_arr3();
  std::vector<long> indices = {5, 0, 2};

  auto e = arr(nda::ellipsis{}, indices);
  static_assert(decltype(e)::rank == 3);
  EXPECT_EQ(e.shape(), (std::array<long, 3>{4, 5, 3}));
  EXPECT_EQ(e(1, 2, 0), arr(1, 2, 5));
  EXPECT_EQ(e(3, 4, 2), arr(3, 4, 2));
}

TEST(ExprIndexed, ConstructWithEllipsisBetweenContainers) {
  auto arr                   = make_arr3();
  std::vector<long> indices1 = {3, 0};
  std::vector<long> indices2 = {5, 1};

  auto e = arr(indices1, nda::ellipsis{}, indices2);
  EXPECT_EQ(e.shape(), (std::array<long, 3>{2, 5, 2}));
  EXPECT_EQ(e(0, 2, 0), arr(3, 2, 5));
  EXPECT_EQ(e(1, 4, 1), arr(0, 4, 1));
}

TEST(ExprIndexed, ConstructWithLongAndEllipsis) {
  auto arr                  = make_arr3();
  std::vector<long> indices = {4, 2};

  auto e1 = arr(1, nda::ellipsis{}, indices);
  static_assert(decltype(e1)::rank == 2);
  EXPECT_EQ(e1.shape(), (std::array<long, 2>{5, 2}));
  EXPECT_EQ(e1(0, 0), arr(1, 0, 4));
  EXPECT_EQ(e1(4, 1), arr(1, 4, 2));

  std::vector<long> indices0 = {3, 2};
  auto e2                    = arr(indices0, nda::ellipsis{}, 2);
  static_assert(decltype(e2)::rank == 2);
  EXPECT_EQ(e2.shape(), (std::array<long, 2>{2, 5}));
  EXPECT_EQ(e2(0, 3), arr(3, 3, 2));
  EXPECT_EQ(e2(1, 0), arr(2, 0, 2));
}

TEST(ExprIndexed, ConstructWithEmptyEllipsis) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {3, 1, 0};

  auto e = arr(nda::ellipsis{}, indices, 1);
  static_assert(decltype(e)::rank == 1);
  EXPECT_EQ(e.shape(), (std::array<long, 1>{3}));
  for (long i = 0; i < 3; ++i) EXPECT_EQ(e(i), arr(indices[i], 1));
}

// ============================================================
// Slicing with long and ellipsis arguments
// ============================================================

TEST(ExprIndexed, SliceNonIndexedDimWithLong) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {2, 0, 4};
  auto e                    = arr(indices, nda::range::all);

  auto s = e(nda::range::all, 1);
  static_assert(decltype(s)::rank == 1);
  static_assert(not nda::is_view_v<decltype(s)>);
  EXPECT_EQ(s.shape(), (std::array<long, 1>{3}));
  for (long i = 0; i < 3; ++i) EXPECT_EQ(s(i), arr(indices[i], 1));
}

TEST(ExprIndexed, SliceIndexedDimWithLongReturnsView) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {2, 0, 4};
  auto e                    = arr(indices, nda::range::all);

  auto s = e(1, nda::range::all);
  static_assert(nda::is_view_v<decltype(s)>);
  static_assert(decltype(s)::rank == 1);
  EXPECT_EQ(s.shape(), (std::array<long, 1>{4}));
  for (long j = 0; j < 4; ++j) EXPECT_EQ(s(j), arr(0, j));
}

TEST(ExprIndexed, WriteThroughSlice) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {2, 0, 4};
  auto e                    = arr(indices, nda::range::all);

  e(1, nda::range::all) = 7.0;
  for (long j = 0; j < 4; ++j) EXPECT_EQ(arr(0, j), 7.0);

  e(nda::range(0, 2), 1) = 9.0;
  EXPECT_EQ(arr(2, 1), 9.0);
  EXPECT_EQ(arr(0, 1), 9.0);
  EXPECT_EQ(arr(4, 1), 41.0);
}

TEST(ExprIndexed, SliceTwoContainersWithLong) {
  auto arr                   = make_arr2();
  std::vector<long> indices1 = {2, 0};
  std::vector<long> indices2 = {3, 1, 0};
  auto e                     = arr(indices1, indices2);

  auto s1 = e(1, nda::range::all);
  static_assert(decltype(s1)::rank == 1);
  static_assert(not nda::is_view_v<decltype(s1)>);
  EXPECT_EQ(s1.shape(), (std::array<long, 1>{3}));
  for (long j = 0; j < 3; ++j) EXPECT_EQ(s1(j), arr(0, indices2[j]));

  auto s2 = e(nda::range::all, 2);
  EXPECT_EQ(s2.shape(), (std::array<long, 1>{2}));
  for (long i = 0; i < 2; ++i) EXPECT_EQ(s2(i), arr(indices1[i], 0));

  auto s3 = e(nda::range(0, 1), 2);
  EXPECT_EQ(s3.shape(), (std::array<long, 1>{1}));
  EXPECT_EQ(s3(0), arr(2, 0));

  EXPECT_EQ(e(1, 2), arr(0, 0));
}

TEST(ExprIndexed, SliceWithEllipsis) {
  auto arr                  = make_arr3();
  std::vector<long> indices = {4, 1};
  auto e                    = arr(nda::range::all, indices, nda::range::all);

  auto s0 = e(nda::ellipsis{});
  EXPECT_EQ(s0.shape(), (std::array<long, 3>{4, 2, 6}));
  EXPECT_EQ(s0(3, 1, 5), arr(3, 1, 5));

  auto s1 = e(nda::ellipsis{}, 3);
  static_assert(decltype(s1)::rank == 2);
  EXPECT_EQ(s1.shape(), (std::array<long, 2>{4, 2}));
  for (long i = 0; i < 4; ++i)
    for (long j = 0; j < 2; ++j) EXPECT_EQ(s1(i, j), arr(i, indices[j], 3));

  auto s2 = e(2, nda::ellipsis{});
  EXPECT_EQ(s2.shape(), (std::array<long, 2>{2, 6}));
  for (long j = 0; j < 2; ++j)
    for (long k = 0; k < 6; ++k) EXPECT_EQ(s2(j, k), arr(2, indices[j], k));

  auto s3 = e(nda::ellipsis{}, nda::range(1, 4));
  EXPECT_EQ(s3.shape(), (std::array<long, 3>{4, 2, 3}));
  EXPECT_EQ(s3(1, 0, 0), arr(1, 4, 1));
  EXPECT_EQ(s3(3, 1, 2), arr(3, 1, 3));

  auto s4 = e(1, nda::ellipsis{}, 2);
  static_assert(decltype(s4)::rank == 1);
  EXPECT_EQ(s4.shape(), (std::array<long, 1>{2}));
  EXPECT_EQ(s4(0), arr(1, 4, 2));
  EXPECT_EQ(s4(1), arr(1, 1, 2));

  auto s5 = e(1, 0, nda::ellipsis{});
  static_assert(nda::is_view_v<decltype(s5)>);
  EXPECT_EQ(s5.shape(), (std::array<long, 1>{6}));
  for (long k = 0; k < 6; ++k) EXPECT_EQ(s5(k), arr(1, 4, k));

  EXPECT_EQ(e(nda::ellipsis{}, 1, 0, 2), arr(1, 4, 2));
}

TEST(ExprIndexed, SliceEllipsisCoveringIndexedDim) {
  auto arr                  = make_arr3();
  std::vector<long> indices = {3, 0};
  auto e                    = arr(indices, nda::range::all, nda::range::all);

  auto s = e(nda::ellipsis{}, 2);
  EXPECT_EQ(s.shape(), (std::array<long, 2>{2, 5}));
  for (long i = 0; i < 2; ++i)
    for (long j = 0; j < 5; ++j) EXPECT_EQ(s(i, j), arr(indices[i], j, 2));
}

TEST(ExprIndexed, SliceMatrixAlgebra) {
  nda::matrix<double> mat(5, 4);
  mat()                     = 1.0;
  std::vector<long> indices = {2, 0, 4};
  auto e                    = mat(indices, nda::range::all);

  static_assert(nda::get_algebra<decltype(e(nda::range::all, 1))> == 'V');
  static_assert(nda::get_algebra<decltype(e(1, nda::range::all))> == 'V');
}

TEST(ExprIndexed, SliceConstPropagation) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {2, 0, 4};
  auto e                    = arr(indices, nda::range::all);
  auto const &ce            = e;

  static_assert(std::is_const_v<typename decltype(ce(1, nda::range::all))::value_type>);
  static_assert(not std::is_const_v<typename decltype(e(1, nda::range::all))::value_type>);
}

// ============================================================
// Traits, address space and noexcept
// ============================================================

TEST(ExprIndexed, Traits) {
  nda::array<double, 2> arr(5, 4);
  std::vector<long> indices = {2, 0, 4};
  auto e                    = arr(indices, nda::range::all);

  static_assert(nda::is_expression<decltype(e)>);
  static_assert(nda::mem::get_addr_space<decltype(e)> == nda::mem::Host);
  static_assert(not noexcept(arr(indices, nda::range::all)));
  static_assert(not noexcept(arr(nda::range::all, indices)));
}

TEST(ExprIndexed, MakeRegularAndCTAD) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {2, 0, 4};

  auto r1 = nda::make_regular(arr(indices, nda::range::all));
  static_assert(std::is_same_v<decltype(r1), nda::array<double, 2>>);
  EXPECT_EQ(r1.shape(), (std::array<long, 2>{3, 4}));
  EXPECT_EQ(r1(0, 1), 21.0);
  EXPECT_EQ(r1(1, 1), 1.0);
  EXPECT_EQ(r1(2, 1), 41.0);

  nda::basic_array r2{arr(indices, nda::range::all)};
  EXPECT_ARRAY_EQ(r1, r2);
}

// ============================================================
// Rvalue source array
// ============================================================

TEST(ExprIndexed, RvalueArray) {
  std::vector<long> indices = {1, 3};
  auto e                    = make_arr2()(indices, nda::range::all);
  static_assert(nda::is_regular_v<decltype(e.a)>);

  nda::array<double, 2> r = e;
  EXPECT_EQ(r.shape(), (std::array<long, 2>{2, 4}));
  EXPECT_EQ(r(0, 2), 12.0);
  EXPECT_EQ(r(1, 2), 32.0);

  auto e2 = make_arr2()(nda::range(1, 4), indices);
  EXPECT_EQ(e2.shape(), (std::array<long, 2>{3, 2}));
  EXPECT_EQ(e2(0, 0), 11.0);
  EXPECT_EQ(e2(2, 1), 33.0);
}

// ============================================================
// Assignment between expressions of the same type
// ============================================================

TEST(ExprIndexed, AssignSameType) {
  nda::array<double, 1> a = {0, 1, 2, 3, 4};
  nda::array<double, 1> b = {10, 11, 12, 13, 14};
  std::vector<long> i1    = {0, 1, 2};
  std::vector<long> i2    = {1, 2, 3};

  a(i1) = b(i2);
  EXPECT_ARRAY_EQ(a, (nda::array<double, 1>{11, 12, 13, 3, 4}));

  a(std::vector<long>{3, 4}) = a(std::vector<long>{0, 1});
  EXPECT_ARRAY_EQ(a, (nda::array<double, 1>{11, 12, 13, 11, 12}));

  // std::array and std::span containers are normalized to the same expression type
  a(std::array<long, 2>{0, 1}) = a(std::span<long const>(i2.data(), 2));
  EXPECT_ARRAY_EQ(a, (nda::array<double, 1>{12, 13, 13, 11, 12}));
}

// ============================================================
// Bounds checking
// ============================================================

TEST(ExprIndexed, ElementOutOfBounds) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {2, 0, 4};
  auto e                    = arr(indices, nda::range::all);

  EXPECT_THROW((void)e(3, 0), std::runtime_error);
  EXPECT_THROW((void)e(-1, 0), std::runtime_error);
  EXPECT_THROW((void)e(0, 4), std::runtime_error);
  EXPECT_NO_THROW((void)e(2, 3));
}

TEST(ExprIndexed, SliceContainerOutOfBounds) {
  auto arr                  = make_arr2();
  std::vector<long> indices = {2, 0, 4};
  auto e                    = arr(indices, nda::range::all);

  EXPECT_THROW((void)e(nda::range(0, 8), nda::range::all), std::runtime_error);
  EXPECT_THROW((void)e(nda::range(-1, 2), nda::range::all), std::runtime_error);
  EXPECT_NO_THROW((void)e(nda::range(0, 3), nda::range::all));

  auto s = e(nda::range(1, 3), nda::range::all);
  EXPECT_EQ(s.shape(), (std::array<long, 2>{2, 4}));
  EXPECT_EQ(s(0, 0), 0.0);
  EXPECT_EQ(s(1, 0), 40.0);
}

// ============================================================
// Block-wise assignment
// ============================================================

namespace {

  // Reference implementation: assign element by element through the containers.
  template <typename A, typename RHS>
  void assign_ref(A &arr, std::vector<long> const &rows, std::vector<long> const &cols, RHS const &rhs) {
    for (long i = 0; i < std::ssize(rows); ++i)
      for (long j = 0; j < std::ssize(cols); ++j) arr(rows[i], cols[j]) = rhs(i, j);
  }

} // namespace

TEST(ExprIndexed, BlockAssignLayouts) {
  std::vector<long> rows = {3, 0, 4};
  std::vector<long> cols = {2, 0};
  auto all               = std::vector<long>{0, 1, 2, 3, 4};
  auto all4              = std::vector<long>{0, 1, 2, 3};

  auto check = [&](auto const &rhs_first, auto const &rhs_second) {
    auto arr                   = make_arr2();
    auto ref                   = make_arr2();
    arr(rows, nda::range::all) = rhs_first;
    assign_ref(ref, rows, all4, rhs_first);
    EXPECT_ARRAY_EQ(arr, ref);

    arr(nda::range::all, cols) = rhs_second;
    assign_ref(ref, all, cols, rhs_second);
    EXPECT_ARRAY_EQ(arr, ref);
  };

  nda::array<double, 2> c_first(3, 4), c_second(5, 2);
  for (long i = 0; i < 3; ++i)
    for (long j = 0; j < 4; ++j) c_first(i, j) = -static_cast<double>(i * 10 + j);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 2; ++j) c_second(i, j) = -static_cast<double>(i * 100 + j);
  check(c_first, c_second);

  nda::array<double, 2, nda::F_layout> f_first(c_first), f_second(c_second);
  check(f_first, f_second);

  // strided views
  nda::array<double, 2> big_first(6, 8), big_second(10, 4);
  big_first  = 1.5;
  big_second = -2.5;
  check(big_first(nda::range(0, 6, 2), nda::range(0, 8, 2)), big_second(nda::range(0, 10, 2), nda::range(0, 4, 2)));

  // lazy expressions
  check(2.0 * c_first + f_first, c_second - 1.0);
}

TEST(ExprIndexed, BlockAssignBothDimsIndexed) {
  auto arr               = make_arr2();
  auto ref               = make_arr2();
  std::vector<long> rows = {3, 0, 4};
  std::vector<long> cols = {2, 0};
  nda::array<double, 2> rhs{{1, 2}, {3, 4}, {5, 6}};

  arr(rows, cols) = rhs;
  assign_ref(ref, rows, cols, rhs);
  EXPECT_ARRAY_EQ(arr, ref);

  arr(rows, cols) = 7.0;
  assign_ref(ref, rows, cols, nda::array<double, 2>(nda::ones<double>(3, 2) * 7.0));
  EXPECT_ARRAY_EQ(arr, ref);
}

TEST(ExprIndexed, BlockAssignRank3MiddleDim) {
  auto arr              = make_arr3();
  auto ref              = make_arr3();
  std::vector<long> idx = {4, 1};
  nda::array<double, 3> rhs(4, 2, 6);
  for (long i = 0; i < 4; ++i)
    for (long j = 0; j < 2; ++j)
      for (long k = 0; k < 6; ++k) rhs(i, j, k) = -static_cast<double>(i * 100 + j * 10 + k);

  arr(nda::range::all, idx, nda::range::all) = rhs;
  for (long i = 0; i < 4; ++i)
    for (long j = 0; j < 2; ++j)
      for (long k = 0; k < 6; ++k) ref(i, idx[j], k) = rhs(i, j, k);
  EXPECT_ARRAY_EQ(arr, ref);

  arr(nda::range::all, idx, nda::range::all) = 3.0;
  for (long i = 0; i < 4; ++i)
    for (long j = 0; j < 2; ++j)
      for (long k = 0; k < 6; ++k) ref(i, idx[j], k) = 3.0;
  EXPECT_ARRAY_EQ(arr, ref);
}

TEST(ExprIndexed, BlockAssignScalarLayouts) {
  auto arr               = make_arr2();
  auto ref               = make_arr2();
  std::vector<long> rows = {3, 0, 4};
  std::vector<long> cols = {2, 0};

  arr(rows, nda::range::all) = -1.0;
  for (long r : rows)
    for (long j = 0; j < 4; ++j) ref(r, j) = -1.0;
  EXPECT_ARRAY_EQ(arr, ref);

  arr(nda::range::all, cols) = -2.0;
  for (long i = 0; i < 5; ++i)
    for (long c : cols) ref(i, c) = -2.0;
  EXPECT_ARRAY_EQ(arr, ref);

  // matrix algebra: the blocks are vectors, so the scalar fills all elements (no diagonal semantics)
  nda::matrix<double> mat(5, 4);
  mat                        = 0.0;
  mat(rows, nda::range::all) = 5.0;
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 4; ++j) EXPECT_EQ(mat(i, j), (std::find(rows.begin(), rows.end(), i) != rows.end()) ? 5.0 : 0.0);
}

TEST(ExprIndexed, AssignFromArrayAdapterFallsBack) {
  auto arr               = make_arr2();
  auto ref               = make_arr2();
  std::vector<long> rows = {3, 0, 4};
  auto rhs               = nda::array_adapter{std::array{3l, 4l}, [](long i, long j) { return static_cast<double>(-(i * 10 + j)); }};

  arr(rows, nda::range::all) = rhs;
  for (long i = 0; i < 3; ++i)
    for (long j = 0; j < 4; ++j) ref(rows[i], j) = rhs(i, j);
  EXPECT_ARRAY_EQ(arr, ref);
}
