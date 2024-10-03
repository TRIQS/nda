// Copyright (c) 2024 Simons Foundation
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
// Authors: Thomas Hahn

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <limits>

// Test fixture for testing various mathematical functions.
struct NDAMathFunction : public ::testing::Test {
  protected:
  NDAMathFunction() {
    A_d = nda::array<double, 3>::rand(shape) - 0.5;
    A_c = nda::array<std::complex<double>, 3>::rand(shape) - std::complex<double>{0.5, 0.5};
  }

  std::array<long, 3> shape{2, 3, 4};
  nda::array<double, 3> A_d;
  nda::array<std::complex<double>, 3> A_c;
  static auto constexpr nan = std::numeric_limits<double>::quiet_NaN();
  static auto constexpr dbl_c = -0.1234;
  static auto constexpr cplx_c = std::complex<double>{-1.234, 2.345};
};

TEST_F(NDAMathFunction, Abs) {
  auto B_d = nda::abs(A_d);
  auto B_c = nda::abs(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::abs(A_d(idxs...)));
    EXPECT_DOUBLE_EQ(B_c(idxs...), std::abs(A_c(idxs...)));
  });
  EXPECT_DOUBLE_EQ(nda::abs(dbl_c), std::abs(dbl_c));
  EXPECT_DOUBLE_EQ(nda::abs(cplx_c), std::abs(cplx_c));
}

TEST_F(NDAMathFunction, Abs2) {
  auto B_d = nda::abs2(A_d);
  auto B_c = nda::abs2(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::abs(A_d(idxs...)) * std::abs(A_d(idxs...)));
    EXPECT_DOUBLE_EQ(B_c(idxs...), std::abs(A_c(idxs...)) * std::abs(A_c(idxs...)));
  });
  EXPECT_DOUBLE_EQ(nda::abs2(dbl_c), std::abs(dbl_c) * std::abs(dbl_c));
  EXPECT_DOUBLE_EQ(nda::abs2(cplx_c), std::abs(cplx_c) * std::abs(cplx_c));
}

TEST_F(NDAMathFunction, Acos) {
  auto B_d = nda::acos(A_d);
  auto B_c = nda::acos(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::acos(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::acos(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::acos(dbl_c), std::acos(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::acos(cplx_c), std::acos(cplx_c));
}

TEST_F(NDAMathFunction, Asin) {
  auto B_d = nda::asin(A_d);
  auto B_c = nda::asin(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::asin(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::asin(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::asin(dbl_c), std::asin(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::asin(cplx_c), std::asin(cplx_c));
}

TEST_F(NDAMathFunction, Atan) {
  auto B_d = nda::atan(A_d);
  auto B_c = nda::atan(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::atan(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::atan(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::atan(dbl_c), std::atan(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::atan(cplx_c), std::atan(cplx_c));
}

TEST_F(NDAMathFunction, Conj) {
  auto B_d = nda::conj(A_d);
  auto B_c = nda::conj(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_EQ(B_d(idxs...), std::conj(A_d(idxs...)));
    EXPECT_EQ(B_c(idxs...), std::conj(A_c(idxs...)));
  });
  EXPECT_EQ(nda::conj(dbl_c), std::conj(dbl_c));
  EXPECT_EQ(nda::conj(cplx_c), std::conj(cplx_c));
}

TEST_F(NDAMathFunction, Cos) {
  auto B_d = nda::cos(A_d);
  auto B_c = nda::cos(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::cos(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::cos(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::cos(dbl_c), std::cos(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::cos(cplx_c), std::cos(cplx_c));
}

TEST_F(NDAMathFunction, Cosh) {
  auto B_d = nda::cosh(A_d);
  auto B_c = nda::cosh(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::cosh(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::cosh(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::cosh(dbl_c), std::cosh(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::cosh(cplx_c), std::cosh(cplx_c));
}

TEST_F(NDAMathFunction, Dagger) {
  using mat_d = nda::array<double, 2>;
  using mat_c = nda::array<std::complex<double>, 2>;

  auto size = 5;
  mat_d M_d = nda::array<double, 2>::rand(size, size) - 0.5;
  mat_c M_c = nda::array<std::complex<double>, 2>::rand(size, size) - std::complex<double>{0.5, 0.5};
  auto B_d  = nda::dagger(M_d);
  auto B_c  = nda::dagger(M_c);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      EXPECT_EQ(B_d(i, j), M_d(j, i));
      EXPECT_EQ(B_c(i, j), std::conj(M_c(j, i)));
    }
  }
}

TEST_F(NDAMathFunction, Exp) {
  auto B_d = nda::exp(A_d);
  auto B_c = nda::exp(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::exp(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::exp(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::exp(dbl_c), std::exp(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::exp(cplx_c), std::exp(cplx_c));
}

TEST_F(NDAMathFunction, Floor) {
  auto B_d = nda::make_regular(nda::floor(A_d));
  nda::for_each(shape, [&](auto... idxs) { EXPECT_EQ(B_d(idxs...), std::floor(A_d(idxs...))); });
  EXPECT_DOUBLE_EQ(nda::floor(dbl_c), std::floor(dbl_c));
}

TEST_F(NDAMathFunction, FrobeniusNorm) {
  auto M        = nda::array<std::complex<double>, 2>::rand(10, 10);
  auto norm     = nda::frobenius_norm(M);
  auto exp_norm = std::sqrt(nda::sum(nda::abs2(M)));
  EXPECT_DOUBLE_EQ(norm, exp_norm);
}

TEST_F(NDAMathFunction, Imag) {
  auto B_d = nda::imag(A_d);
  auto B_c = nda::imag(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_EQ(B_d(idxs...), std::imag(A_d(idxs...)));
    EXPECT_EQ(B_c(idxs...), std::imag(A_c(idxs...)));
  });
  EXPECT_EQ(nda::imag(dbl_c), std::imag(dbl_c));
  EXPECT_EQ(nda::imag(cplx_c), std::imag(cplx_c));
}

TEST_F(NDAMathFunction, Isnan) {
  using namespace nda::clef::literals;
  A_d(nda::ellipsis{}, nda::range(0, shape.back(), 2)) = nan;
  A_c(nda::ellipsis{}, nda::range(0, shape.back(), 2)) = std::complex<double>{2.0, nan};

  auto B_d = nda::isnan(A_d);
  auto B_c = nda::isnan(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_EQ(B_d(idxs...), std::isnan(A_d(idxs...)));
    EXPECT_EQ(B_c(idxs...), nda::isnan(A_c(idxs...)));
  });
  EXPECT_EQ(nda::isnan(dbl_c), std::isnan(dbl_c));
  EXPECT_EQ(nda::isnan(cplx_c), nda::detail::isnan(cplx_c));
}

TEST_F(NDAMathFunction, Log) {
  A_d += 0.5;
  auto B_d = nda::log(A_d);
  auto B_c = nda::log(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::log(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::log(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::log(nda::abs(dbl_c)), std::log(std::abs(dbl_c)));
  EXPECT_COMPLEX_NEAR(nda::log(cplx_c), std::log(cplx_c));
}

TEST_F(NDAMathFunction, MapCustom) {
  auto B_d = nda::map([](double x) { return std::cos(x) * std::cos(x) + std::sin(x) * std::sin(x); })(A_d);
  auto C_d = nda::map([](double x, double y) { return std::cos(x) * std::cos(x) + std::sin(y) * std::sin(y); })(A_d, A_d);
  auto exp = nda::ones<double>(shape);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), exp(idxs...));
    EXPECT_DOUBLE_EQ(C_d(idxs...), exp(idxs...));
  });
}

TEST_F(NDAMathFunction, Pow) {
  auto B_d = nda::pow(A_d, 2);
  auto B_c = nda::pow(A_c, 2);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::pow(A_d(idxs...), 2));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::pow(A_c(idxs...), 2), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::pow(dbl_c, 2), std::pow(dbl_c, 2));
  EXPECT_COMPLEX_NEAR(nda::pow(cplx_c, 2), std::pow(cplx_c, 2));
}

TEST_F(NDAMathFunction, Real) {
  auto B_d = nda::real(A_d);
  auto B_c = nda::real(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_EQ(B_d(idxs...), std::real(A_d(idxs...)));
    EXPECT_EQ(B_c(idxs...), std::real(A_c(idxs...)));
  });
  EXPECT_EQ(nda::real(dbl_c), std::real(dbl_c));
  EXPECT_EQ(nda::real(cplx_c), std::real(cplx_c));
}

TEST_F(NDAMathFunction, Sin) {
  auto B_d = nda::sin(A_d);
  auto B_c = nda::sin(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::sin(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::sin(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::sin(dbl_c), std::sin(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::sin(cplx_c), std::sin(cplx_c));
}

TEST_F(NDAMathFunction, Sinh) {
  auto B_d = nda::sinh(A_d);
  auto B_c = nda::sinh(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::sinh(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::sinh(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::sinh(dbl_c), std::sinh(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::sinh(cplx_c), std::sinh(cplx_c));
}

TEST_F(NDAMathFunction, Sqrt) {
  A_d += 0.5;
  auto B_d = nda::sqrt(A_d);
  auto B_c = nda::sqrt(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::sqrt(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::sqrt(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::sqrt(nda::abs(dbl_c)), std::sqrt(std::abs(dbl_c)));
  EXPECT_COMPLEX_NEAR(nda::sqrt(cplx_c), std::sqrt(cplx_c));
}

TEST_F(NDAMathFunction, Tan) {
  auto B_d = nda::tan(A_d);
  auto B_c = nda::tan(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::tan(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::tan(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::tan(dbl_c), std::tan(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::tan(cplx_c), std::tan(cplx_c));
}

TEST_F(NDAMathFunction, Tanh) {
  auto B_d = nda::tanh(A_d);
  auto B_c = nda::tanh(A_c);
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), std::tanh(A_d(idxs...)));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::tanh(A_c(idxs...)), 1e-14);
  });
  EXPECT_DOUBLE_EQ(nda::tanh(dbl_c), std::tanh(dbl_c));
  EXPECT_COMPLEX_NEAR(nda::tanh(cplx_c), std::tanh(cplx_c));
}

TEST_F(NDAMathFunction, Trace) {
  auto M   = nda::array<std::complex<double>, 2>::rand(10, 10);
  auto tr  = nda::trace(M);
  auto exp = nda::sum(nda::diagonal(M));
  EXPECT_COMPLEX_NEAR(tr, exp, 1e-14);
}

TEST_F(NDAMathFunction, Combinations) {
  auto B_d = nda::pow(nda::pow(nda::abs(A_d), 1.5), 2.0 / 3.0);
  auto C_d = nda::sqrt(nda::pow(A_d, 2));
  auto D_d = nda::log(nda::exp(A_d));
  nda::for_each(shape, [&](auto... idxs) {
    EXPECT_NEAR(B_d(idxs...), std::abs(A_d(idxs...)), 1e-10);
    EXPECT_NEAR(C_d(idxs...), std::abs(A_d(idxs...)), 1e-10);
    EXPECT_NEAR(D_d(idxs...), A_d(idxs...), 1e-10);
  });
}

TEST_F(NDAMathFunction, CombineMathFunctionsWithArithmeticOps) {
  using arr_t = nda::array<double, 2>;
  using mat_t = nda::matrix<std::complex<double>>;

  arr_t A(3, 3), B(3, 3), A_sqr(3, 3), B_x2_abs(3, 3), A_p10B(3, 3), A_p10B_abs(3, 3), A_p10B_max(3, 3), A_pow2(3, 3);
  mat_t C(3, 3), C_conj(3, 3), C_transp(3, 3);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      A(i, j) = i + 2 * j + 1;
      B(i, j) = i - 3 * j;
      C(i, j) = A(i, j) + 1i * B(i, j);

      A_pow2(i, j)     = A(i, j) * A(i, j);
      A_sqr(i, j)      = A(i, j) * A(i, j);
      B_x2_abs(i, j)   = std::abs(2 * B(i, j));
      A_p10B(i, j)     = A(i, j) + 10 * B(i, j);
      A_p10B_abs(i, j) = std::abs(A(i, j) + 10 * B(i, j));
      A_p10B_max(i, j) = std::max(A(i, j), 10 * B(i, j));
      C_conj(i, j)     = A(i, j) - 1i * B(i, j);
      C_transp(j, i)   = A(i, j) + 1i * B(i, j);
    }

  auto fabs_map = nda::map([](double x) { return std::fabs(x); });
  auto max_map  = nda::map([](double x, double y) { return std::max(x, y); });
  auto sqr_map  = nda::map([](double x) { return x * x; });

  EXPECT_ARRAY_NEAR(arr_t(pow(arr_t{A}, 2)), A_sqr);
  EXPECT_ARRAY_NEAR(arr_t(sqr_map(A)), A_sqr);
  EXPECT_ARRAY_NEAR(arr_t(fabs_map(B + B)), B_x2_abs);
  EXPECT_ARRAY_NEAR(arr_t(A + 10 * B), A_p10B);
  EXPECT_ARRAY_NEAR(arr_t(fabs_map(A + 10 * B)), A_p10B_abs);
  EXPECT_ARRAY_NEAR(arr_t(max_map(A, 10 * B)), A_p10B_max);
  EXPECT_ARRAY_NEAR(mat_t(conj(C)), C_conj);
  EXPECT_ARRAY_NEAR(mat_t(transpose(C)), C_transp);
  EXPECT_ARRAY_NEAR(mat_t(C * conj(transpose(C))), mat_t(transpose(conj(C) * transpose(C))));
}

