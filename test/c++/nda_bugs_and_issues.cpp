// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <type_traits>

// Multiply an m-by-m matrix A by an m-by-n-by-p tensor B, combining the last two indices of B.
nda::array<double, 3> arraymult(nda::matrix<double, nda::F_layout> a, nda::array<double, 3, nda::F_layout> b) {
  auto [m, n, p] = b.shape();
  auto brs       = nda::reshape(b, std::array{m, n * p});
  auto bmat      = nda::matrix_const_view<double, nda::F_layout>(brs);
  return nda::reshape(a * bmat, std::array{m, n, p});
}

TEST(NDA, Issue34) {
  // issue concerning array/matrix/tensor multiplication
  int m = 1;
  int n = 2;
  int p = 3;

  auto A = transpose(nda::eye<double>(m));
  auto B = nda::array<double, 3, nda::F_layout>(m, n, p);
  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) { B(i, j, k) = double(j + n * k); }
    }
  }

  // compute the products the old-fashioned way
  auto C_exp = nda::array<double, 3, nda::F_layout>(m, n, p);
  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < n; ++j) { C_exp(nda::range::all, j, k) = A * nda::vector_const_view<double>(B(nda::range::all, j, k)); }
  }

  // compute products with the arraymult function
  auto C = arraymult(A, B);

  // check results
  EXPECT_ARRAY_NEAR(C(0, nda::ellipsis{}), C_exp(0, nda::ellipsis{}));
}

TEST(NDA, Bug2) {
  // bug concerning matrix views and their assignment
  nda::array<double, 3> A(10, 2, 2);
  A() = 0;

  A(4, nda::range::all, nda::range::all) = 1;
  A(5, nda::range::all, nda::range::all) = 2;

  nda::matrix_view<double> M1 = A(4, nda::range::all, nda::range::all);
  nda::matrix_view<double> M2 = A(5, nda::range::all, nda::range::all);

  EXPECT_ARRAY_NEAR(M1, nda::matrix<double>{{1, 1}, {1, 1}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<double>{{2, 2}, {2, 2}});

  M1 = M2;

  EXPECT_ARRAY_NEAR(M1, nda::matrix<double>{{2, 2}, {2, 2}});
  EXPECT_ARRAY_NEAR(M2, nda::matrix<double>{{2, 2}, {2, 2}});
}

TEST(NDA, DanglingScalarIssue) {
  // scalars should be stored as values not as references in lazy expressions
  nda::array<long, 1> a{4, 2, 3}, b{8, 4, 6};

  auto f = [&a]() {
    long x = 2;
    return x * a;
  };

  static_assert(!std::is_reference_v<decltype(f().l)>);
  EXPECT_EQ_ARRAY((nda::array<long, 1>{f()}), b);
}

TEST(NDA, LayoutInfoBug) {
  using expr_t = nda::expr<'+', nda::basic_array<std::complex<double>, 2, nda::C_layout, 'M', nda::heap<>> &,
                           const nda::basic_array_view<const std::complex<double>, 2, nda::basic_layout<0, 16, nda::layout_prop_e::none>, 'A',
                                                       nda::default_accessor, nda::borrowed<>> &>;

  EXPECT_EQ((nda::layout_prop_e::none & nda::layout_prop_e::contiguous), nda::layout_prop_e::none);
  EXPECT_EQ(expr_t::l_is_scalar, false);
  EXPECT_EQ(expr_t::r_is_scalar, false);
  EXPECT_EQ(expr_t::compute_layout_info().prop, nda::layout_prop_e::none);
  EXPECT_EQ(nda::get_layout_info<expr_t::R_t>.prop, nda::layout_prop_e::none);
  EXPECT_EQ(nda::get_layout_info<expr_t::L_t>.prop, nda::layout_prop_e::contiguous);
  EXPECT_EQ(nda::get_layout_info<expr_t::R_t>.prop & nda::get_layout_info<expr_t::L_t>.prop, nda::layout_prop_e::none);
}

TEST(NDA, STLAlgorithmBug) {
  nda::array<double, 1> A{1, 3, 2}, B{2, 3, 1};
  EXPECT_TRUE(std::lexicographical_compare(A.begin(), A.end(), B.begin(), B.end()));
  EXPECT_FALSE(std::lexicographical_compare(B.begin(), B.end(), A.begin(), A.end()));

  nda::array<double, 1> C = {1, 3, 2}, D = {1, 2, 10};
  EXPECT_FALSE(std::lexicographical_compare(C.begin(), C.end(), D.begin(), D.end()));
  EXPECT_TRUE(std::lexicographical_compare(D.begin(), D.end(), C.begin(), C.end()));
}

TEST(NDA, AmbiguousAssignmentOperatorIssue) {
  // issue concerning an ambiguous assignment operator overload when assigning a contiguous range
  nda::array<std::string, 1> A(3);
  A() = std::string("foo");
  for (int i = 0; i < 3; ++i) { EXPECT_EQ(A(i), std::string("foo")); }

  nda::array<std::array<int, 3>, 1> B(3);
  B = std::array{1, 2, 3};
  for (int i = 0; i < 3; ++i) { EXPECT_EQ(B(i), (std::array{1, 2, 3})); }
}

TEST(NDA, AssignmentTo1DNegativeStridedViewsIssue) {
  // issue concerning the assignment of scalar values to 1D strided views with negative strides
  nda::array<int, 1> A(10);
  for (int i = 0; auto &x : A) x = i++;

  auto B = A(nda::range(9, -1, -1));
  for (int i = 9; auto x : B) EXPECT_EQ(x, i--);

  B = 1;
  for (auto x : B) EXPECT_EQ(x, 1);
}

TEST(NDA, IsStrided1DAndIsContiguousIssue) {
  // issue concerning the nda::idx_map::is_strided_1d and the nda::idx_map::is_contiguous function
  nda::array<int, 2> A(10, 10);
  EXPECT_TRUE(A.indexmap().is_contiguous());
  EXPECT_TRUE(A.indexmap().has_positive_strides());
  EXPECT_TRUE(A.indexmap().is_strided_1d());

  auto A_v1 = A(nda::range::all, nda::range(9, -1, -1));
  EXPECT_TRUE(A_v1.indexmap().is_contiguous());
  EXPECT_FALSE(A_v1.indexmap().has_positive_strides());
  EXPECT_TRUE(A_v1.indexmap().is_strided_1d());

  auto A_v2 = A(nda::range(9, -1, -1), nda::range::all);
  EXPECT_TRUE(A_v2.indexmap().is_contiguous());
  EXPECT_FALSE(A_v2.indexmap().has_positive_strides());
  EXPECT_TRUE(A_v2.indexmap().is_strided_1d());

  auto A_v3 = A(nda::range(9, -1, -1), nda::range(9, -1, -1));
  EXPECT_TRUE(A_v3.indexmap().is_contiguous());
  EXPECT_FALSE(A_v3.indexmap().has_positive_strides());
  EXPECT_TRUE(A_v3.indexmap().is_strided_1d());

  nda::array<int, 2> B(10, 1);
  EXPECT_TRUE(B.indexmap().is_contiguous());
  EXPECT_TRUE(B.indexmap().has_positive_strides());
  EXPECT_TRUE(B.indexmap().is_strided_1d());

  auto B_v1 = B(nda::range(0, 10, 2), nda::range::all);
  EXPECT_FALSE(B_v1.indexmap().is_contiguous());
  EXPECT_TRUE(B_v1.indexmap().has_positive_strides());
  EXPECT_TRUE(B_v1.indexmap().is_strided_1d());
}

TEST(NDA, LayoutAfterReshapeIssue) {
  // issue concerning the resulting layout after a reshape operation
  auto check = [](const auto &a) {
    auto b = nda::flatten(a);
    EXPECT_EQ(b.stride_order(), (std::array<int, 1>{0}));
    EXPECT_EQ(b.strides(), (std::array<long, 1>{1}));
    EXPECT_TRUE(b.is_contiguous());
    EXPECT_TRUE(b.has_positive_strides());
    EXPECT_TRUE(b.is_stride_order_C());
  };
  nda::array<int, 3> A(2, 2, 2);
  check(A);
  nda::array<int, 3, nda::F_layout> B(2, 2, 2);
  check(B);
  nda::array<int, 3, nda::basic_layout<0, nda::encode(std::array{1, 2, 0}), nda::layout_prop_e::contiguous>> C(2, 2, 2);
  check(C);
}

// Unqualified scalar math calls with `using namespace nda` (as done in cppdlr) were ambiguous: libc++ declares generic
// templates like `pow(A1, A2)` and `sqrt(Integral)` in the global namespace (and libstdc++ in std), which tied with the
// nda function templates. The checks cover every mapped function with a std counterpart, plus their array overloads.
using namespace nda;

TEST(NDA, ScalarMathLookupIssue) {
  // The scalar overloads of the <cmath> family in nda are the std ones (using-declarations), with the std result types.
  static_assert(std::is_same_v<decltype(nda::pow(2.0f, 3.0f)), float>);
  static_assert(std::is_same_v<decltype(nda::pow(2.0f, 3)), double>);
  static_assert(std::is_same_v<decltype(nda::pow(std::complex<double>{}, 2)), std::complex<double>>);
  static_assert(std::is_same_v<decltype(nda::sqrt(5.0f)), float>);
  static_assert(std::is_same_v<decltype(nda::abs(-3)), int>);
  static_assert(std::is_same_v<decltype(nda::isnan(1)), bool>);

  int j    = 1;
  long l   = 1;
  float f  = 0.5f;
  double d = 0.5;
  std::complex<double> z{0.5, 0.25};
  auto x = nda::vector<double>{0.5, 0.25};

  // Compare an unqualified call with the std one for real scalars and the array overload.
  auto check_real = [&](auto nda_f, auto std_f) {
    EXPECT_EQ(nda_f(j), std_f(j));
    EXPECT_EQ(nda_f(l), std_f(l));
    EXPECT_EQ(nda_f(f), std_f(f));
    EXPECT_EQ(nda_f(d), std_f(d));
    EXPECT_ARRAY_NEAR(nda::vector<double>{nda_f(x)}, (nda::vector<double>{std_f(x(0)), std_f(x(1))}));
  };

  // Same as check_real and also for complex scalars.
  auto check = [&](auto nda_f, auto std_f) {
    check_real(nda_f, std_f);
    EXPECT_EQ(nda_f(z), std_f(z));
  };

  EXPECT_EQ(pow(-1, j), std::pow(-1, j));
  EXPECT_EQ(pow(j, 2), std::pow(j, 2));
  EXPECT_EQ(pow(j, 2.0), std::pow(j, 2.0));
  EXPECT_EQ(pow(f, 2), std::pow(f, 2));
  EXPECT_EQ(pow(f, f), std::pow(f, f));
  EXPECT_EQ(pow(d, 2), std::pow(d, 2));
  EXPECT_EQ(pow(d, 2.0), std::pow(d, 2.0));
  EXPECT_EQ(pow(x(1), 2), std::pow(x(1), 2));
  EXPECT_EQ(pow(z, 2), std::pow(z, 2));
  EXPECT_EQ(pow(z, 0.5), std::pow(z, 0.5));
  EXPECT_EQ(pow(d, z), std::pow(d, z));
  EXPECT_ARRAY_NEAR(nda::vector<double>{pow(x, 2)}, (nda::vector<double>{0.25, 0.0625}));
  EXPECT_EQ(abs(-1), std::abs(-1));
  EXPECT_FALSE(isnan(j));
  EXPECT_FALSE(isnan(f));
  EXPECT_FALSE(isnan(d));
  EXPECT_FALSE(isnan(z));
  check([](auto const &a) { return abs(a); }, [](auto const &a) { return std::abs(a); });
  check_real([](auto const &a) { return floor(a); }, [](auto const &a) { return std::floor(a); });
  check([](auto const &a) { return exp(a); }, [](auto const &a) { return std::exp(a); });
  check([](auto const &a) { return cos(a); }, [](auto const &a) { return std::cos(a); });
  check([](auto const &a) { return sin(a); }, [](auto const &a) { return std::sin(a); });
  check([](auto const &a) { return tan(a); }, [](auto const &a) { return std::tan(a); });
  check([](auto const &a) { return cosh(a); }, [](auto const &a) { return std::cosh(a); });
  check([](auto const &a) { return sinh(a); }, [](auto const &a) { return std::sinh(a); });
  check([](auto const &a) { return tanh(a); }, [](auto const &a) { return std::tanh(a); });
  check([](auto const &a) { return acos(a); }, [](auto const &a) { return std::acos(a); });
  check([](auto const &a) { return asin(a); }, [](auto const &a) { return std::asin(a); });
  check([](auto const &a) { return atan(a); }, [](auto const &a) { return std::atan(a); });
  check([](auto const &a) { return log(a); }, [](auto const &a) { return std::log(a); });
  check([](auto const &a) { return sqrt(a); }, [](auto const &a) { return std::sqrt(a); });
}
