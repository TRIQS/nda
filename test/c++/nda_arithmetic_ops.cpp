// Copyright (c) 2019-2022 Simons Foundation
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

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <algorithm>
#include <array>
#include <complex>

// Naive matrix multiplication.
template <typename A, typename B>
auto matrix_multiplication(const A &a, const B &b) {
  nda::get_regular_t<A> res(a.extent(0), b.extent(1));
  for (long i = 0; i < a.extent(0); ++i) {
    for (long j = 0; j < b.extent(1); ++j) {
      res(i, j) = 0;
      for (long k = 0; k < a.extent(1); ++k) res(i, j) += a(i, k) * b(k, j);
    }
  }
  return res;
}

// Naive matrix-vector multiplication.
template <typename A, typename B>
auto matrix_vector_multiplication(const A &a, const B &b) {
  nda::get_regular_t<B> res(a.extent(0));
  for (long i = 0; i < a.extent(0); ++i) {
    res(i) = 0;
    for (long j = 0; j < a.extent(1); ++j) { res(i) += a(i, j) * b(j); }
  }
  return res;
}

// Test fixture for testing various algorithms.
struct NDAArithmeticOps : public ::testing::Test {
  protected:
  NDAArithmeticOps() {
    A_d      = nda::array<double, 3>::rand(arr_shape) - 0.5;
    A_c      = nda::array<std::complex<double>, 3>::rand(arr_shape) - std::complex<double>{0.5, 0.5};
    M_d_sq   = nda::matrix<double>::rand(mat_sq_shape[0], mat_sq_shape[1]) - 0.5;
    M_c_sq   = nda::matrix<std::complex<double>>::rand(mat_sq_shape[0], mat_sq_shape[1]) - std::complex<double>{0.5, 0.5};
    M_d_rect = nda::matrix<double>::rand(mat_rect_shape[0], mat_rect_shape[1]) - 0.5;
    M_c_rect = nda::matrix<std::complex<double>>::rand(mat_rect_shape[0], mat_rect_shape[1]) - std::complex<double>{0.5, 0.5};
    V_d      = nda::vector<double>::rand(vec_shape[0]) - 0.5;
    V_c      = nda::vector<std::complex<double>>::rand(vec_shape[0]) - std::complex<double>{0.5, 0.5};
  }

  std::array<long, 3> arr_shape{2, 3, 4};
  nda::array<double, 3> A_d;
  nda::array<std::complex<double>, 3> A_c;
  std::array<long, 2> mat_sq_shape{3, 3};
  nda::matrix<double> M_d_sq;
  nda::matrix<std::complex<double>> M_c_sq;
  std::array<long, 2> mat_rect_shape{3, 5};
  nda::matrix<double> M_d_rect;
  nda::matrix<std::complex<double>> M_c_rect;
  std::array<long, 1> vec_shape{5};
  nda::vector<double> V_d;
  nda::vector<std::complex<double>> V_c;
  static constexpr auto z = std::complex<double>{2.0, 3.0};
  static constexpr auto x = 2.0;
};

TEST_F(NDAArithmeticOps, ScalarMultiplication) {
  // array - scalar multiplication
  auto B_d  = x * A_d;
  auto B_c  = A_c * z;
  auto B_d2 = A_d;
  B_d2 *= x;
  auto B_c2 = A_c;
  B_c2 *= z;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), x * A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), z * A_c(idxs...));
    EXPECT_DOUBLE_EQ(B_d2(idxs...), x * A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c2(idxs...), z * A_c(idxs...));
  });

  // matrix - scalar multiplication
  auto N_d  = M_d_rect * x;
  auto N_c  = z * M_c_rect;
  auto N_d2 = M_d_rect;
  N_d2 *= x;
  auto N_c2 = M_c_rect;
  N_c2 *= z;
  nda::for_each(mat_rect_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(N_d(idxs...), x * M_d_rect(idxs...));
    EXPECT_COMPLEX_NEAR(N_c(idxs...), z * M_c_rect(idxs...));
    EXPECT_DOUBLE_EQ(N_d2(idxs...), x * M_d_rect(idxs...));
    EXPECT_COMPLEX_NEAR(N_c2(idxs...), z * M_c_rect(idxs...));
  });

  // vector - scalar multiplication
  auto W_d  = V_d * x;
  auto W_c  = z * V_c;
  auto W_d2 = V_d;
  W_d2 *= x;
  auto W_c2 = V_c;
  W_c2 *= z;
  nda::for_each(vec_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(W_d(idxs...), x * V_d(idxs...));
    EXPECT_COMPLEX_NEAR(W_c(idxs...), z * V_c(idxs...));
    EXPECT_DOUBLE_EQ(W_d2(idxs...), x * V_d(idxs...));
    EXPECT_COMPLEX_NEAR(W_c2(idxs...), z * V_c(idxs...));
  });
}

TEST_F(NDAArithmeticOps, ScalarAddition) {
  // array - scalar addition
  auto B_d  = x + A_d;
  auto B_c  = A_c + z;
  auto B_d2 = A_d;
  B_d2 += x;
  auto B_c2 = A_c;
  B_c2 += z;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), x + A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), z + A_c(idxs...));
    EXPECT_DOUBLE_EQ(B_d2(idxs...), x + A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c2(idxs...), z + A_c(idxs...));
  });

  // matrix - scalar addition
  auto N_d  = M_d_rect + x;
  auto N_c  = z + M_c_rect;
  auto N_d2 = M_d_rect;
  N_d2 += x;
  auto N_c2 = M_c_rect;
  N_c2 += z;
  nda::for_each(mat_rect_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(N_d(idxs...), (are_equal(idxs...) ? x + M_d_rect(idxs...) : M_d_rect(idxs...)));
    EXPECT_COMPLEX_NEAR(N_c(idxs...), (are_equal(idxs...) ? z + M_c_rect(idxs...) : M_c_rect(idxs...)));
    EXPECT_DOUBLE_EQ(N_d2(idxs...), (are_equal(idxs...) ? x + M_d_rect(idxs...) : M_d_rect(idxs...)));
    EXPECT_COMPLEX_NEAR(N_c2(idxs...), (are_equal(idxs...) ? z + M_c_rect(idxs...) : M_c_rect(idxs...)));
  });
}

TEST_F(NDAArithmeticOps, ScalarSubtraction) {
  // array - scalar subtraction
  auto B_d  = x - A_d;
  auto B_c  = A_c - z;
  auto B_c2 = A_c;
  B_c2 -= z;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), x - A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), A_c(idxs...) - z);
    EXPECT_COMPLEX_NEAR(B_c2(idxs...), A_c(idxs...) - z);
  });

  // matrix - scalar subtraction
  auto N_d  = M_d_rect - x;
  auto N_c  = z - M_c_rect;
  auto N_d2 = M_d_rect;
  N_d2 -= x;
  nda::for_each(mat_rect_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(N_d(idxs...), (are_equal(idxs...) ? M_d_rect(idxs...) - x : M_d_rect(idxs...)));
    EXPECT_COMPLEX_NEAR(N_c(idxs...), (are_equal(idxs...) ? z - M_c_rect(idxs...) : -M_c_rect(idxs...)));
    EXPECT_DOUBLE_EQ(N_d2(idxs...), (are_equal(idxs...) ? M_d_rect(idxs...) - x : M_d_rect(idxs...)));
  });
}

TEST_F(NDAArithmeticOps, ScalarDivision) {
  // array - scalar division
  auto B_d  = x / A_d;
  auto B_c  = A_c / z;
  auto B_c2 = A_c;
  B_c2 /= z;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), x / A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), A_c(idxs...) / z);
    EXPECT_COMPLEX_NEAR(B_c2(idxs...), A_c(idxs...) / z);
  });

  // scalar - matrix division
  auto N_d        = x / M_d_sq;
  auto inv_M_d_sq = nda::inverse(M_d_sq);
  nda::for_each(mat_sq_shape, [&](auto... idxs) { EXPECT_DOUBLE_EQ(N_d(idxs...), x * inv_M_d_sq(idxs...)); });

  // matrix - scalar division
  auto N_c  = M_c_rect / z;
  auto N_c2 = M_c_rect;
  N_c2 /= z;
  nda::for_each(mat_rect_shape, [&](auto... idxs) {
    EXPECT_COMPLEX_NEAR(N_c(idxs...), M_c_rect(idxs...) / z);
    EXPECT_COMPLEX_NEAR(N_c2(idxs...), M_c_rect(idxs...) / z);
  });
}

TEST_F(NDAArithmeticOps, ArrayMultiplication) {
  // array - array multiplication
  auto B_d  = A_d * A_d;
  auto B_c  = A_c * A_c;
  auto B_d2 = A_d;
  B_d2 *= A_d;
  auto B_c2 = A_c;
  B_c2 *= A_c;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_NEAR(B_d(idxs...), A_d(idxs...) * A_d(idxs...), 1e-10);
    EXPECT_COMPLEX_NEAR(B_c(idxs...), A_c(idxs...) * A_c(idxs...));
    EXPECT_NEAR(B_d2(idxs...), A_d(idxs...) * A_d(idxs...), 1e-10);
    EXPECT_COMPLEX_NEAR(B_c2(idxs...), A_c(idxs...) * A_c(idxs...));
  });

  // square matrix - matrix multiplication
  auto N_d_sq  = M_d_sq * M_d_sq;
  auto N_c_sq  = M_c_sq * M_c_sq;
  auto N_d_sq2 = M_d_sq;
  N_d_sq2 *= M_d_sq;
  auto N_c_sq2 = M_c_sq;
  N_c_sq2 *= M_c_sq;
  auto N_d_sq_exp = matrix_multiplication(M_d_sq, M_d_sq);
  auto N_c_sq_exp = matrix_multiplication(M_c_sq, M_c_sq);
  nda::for_each(mat_sq_shape, [&](auto... idxs) {
    EXPECT_NEAR(N_d_sq(idxs...), N_d_sq_exp(idxs...), 1e-10);
    EXPECT_COMPLEX_NEAR(N_c_sq(idxs...), N_c_sq_exp(idxs...));
    EXPECT_NEAR(N_d_sq2(idxs...), N_d_sq_exp(idxs...), 1e-10);
    EXPECT_COMPLEX_NEAR(N_c_sq2(idxs...), N_c_sq_exp(idxs...));
  });

  // rectangular matrix - matrix multiplication
  auto N_d_rect  = M_d_rect * nda::transpose(M_d_rect);
  auto N_c_rect  = M_c_rect * nda::transpose(M_c_rect);
  auto N_d_rect2 = M_d_rect;
  N_d_rect2 *= nda::transpose(M_d_rect);
  auto N_c_rect2 = M_c_rect;
  N_c_rect2 *= nda::transpose(M_c_rect);
  auto N_d_rect_exp = matrix_multiplication(M_d_rect, nda::transpose(M_d_rect));
  auto N_c_rect_exp = matrix_multiplication(M_c_rect, nda::transpose(M_c_rect));
  nda::for_each(N_d_rect.shape(), [&](auto... idxs) {
    EXPECT_NEAR(N_d_rect(idxs...), N_d_rect_exp(idxs...), 1e-10);
    EXPECT_COMPLEX_NEAR(N_c_rect(idxs...), N_c_rect_exp(idxs...));
    EXPECT_NEAR(N_d_rect2(idxs...), N_d_rect_exp(idxs...), 1e-10);
    EXPECT_COMPLEX_NEAR(N_c_rect2(idxs...), N_c_rect_exp(idxs...));
  });

  // matrix - vector multiplication
  auto W_d     = M_d_rect * V_d;
  auto W_c     = M_c_rect * V_c;
  auto W_d_exp = matrix_vector_multiplication(M_d_rect, V_d);
  auto W_c_exp = matrix_vector_multiplication(M_c_rect, V_c);
  nda::for_each(W_d.shape(), [&](auto... idxs) {
    EXPECT_NEAR(W_d(idxs...), W_d_exp(idxs...), 1e-10);
    EXPECT_COMPLEX_NEAR(W_c(idxs...), W_c_exp(idxs...));
  });
}

TEST_F(NDAArithmeticOps, ArrayAddition) {
  // array - array addition
  auto B_d  = A_d + A_d;
  auto B_c  = A_c + A_c;
  auto B_d2 = A_d;
  B_d2 += A_d;
  auto B_c2 = A_c;
  B_c2 += A_c;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), A_d(idxs...) + A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), A_c(idxs...) + A_c(idxs...));
    EXPECT_DOUBLE_EQ(B_d2(idxs...), A_d(idxs...) + A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c2(idxs...), A_c(idxs...) + A_c(idxs...));
  });

  // matrix - matrix addition
  auto N_d  = M_d_rect + M_d_rect;
  auto N_c  = M_c_rect + M_c_rect;
  auto N_d2 = M_d_rect;
  N_d2 += M_d_rect;
  auto N_c2 = M_c_rect;
  N_c2 += M_c_rect;
  nda::for_each(mat_rect_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(N_d(idxs...), M_d_rect(idxs...) + M_d_rect(idxs...));
    EXPECT_COMPLEX_NEAR(N_c(idxs...), M_c_rect(idxs...) + M_c_rect(idxs...));
    EXPECT_DOUBLE_EQ(N_d2(idxs...), M_d_rect(idxs...) + M_d_rect(idxs...));
    EXPECT_COMPLEX_NEAR(N_c2(idxs...), M_c_rect(idxs...) + M_c_rect(idxs...));
  });

  // vector - vector addition
  auto W_d  = V_d + V_d;
  auto W_c  = V_c + V_c;
  auto W_d2 = V_d;
  W_d2 += V_d;
  auto W_c2 = V_c;
  W_c2 += V_c;
  nda::for_each(vec_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(W_d(idxs...), V_d(idxs...) + V_d(idxs...));
    EXPECT_COMPLEX_NEAR(W_c(idxs...), V_c(idxs...) + V_c(idxs...));
    EXPECT_DOUBLE_EQ(W_d2(idxs...), V_d(idxs...) + V_d(idxs...));
    EXPECT_COMPLEX_NEAR(W_c2(idxs...), V_c(idxs...) + V_c(idxs...));
  });
}

TEST_F(NDAArithmeticOps, ArraySubtraction) {
  // array - array addition
  auto B_d  = A_d - A_d;
  auto B_c  = A_c - A_c;
  auto B_d2 = A_d;
  B_d2 -= A_d;
  auto B_c2 = A_c;
  B_c2 -= A_c;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), 0.0);
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::complex<double>{0.0, 0.0});
    EXPECT_DOUBLE_EQ(B_d2(idxs...), 0.0);
    EXPECT_COMPLEX_NEAR(B_c2(idxs...), std::complex<double>{0.0, 0.0});
  });

  // matrix - matrix addition
  auto N_d  = M_d_rect - M_d_rect;
  auto N_c  = M_c_rect - M_c_rect;
  auto N_d2 = M_d_rect;
  N_d2 -= M_d_rect;
  auto N_c2 = M_c_rect;
  N_c2 -= M_c_rect;
  nda::for_each(mat_rect_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(N_d(idxs...), 0.0);
    EXPECT_COMPLEX_NEAR(N_c(idxs...), std::complex<double>{0.0, 0.0});
    EXPECT_DOUBLE_EQ(N_d2(idxs...), 0.0);
    EXPECT_COMPLEX_NEAR(N_c2(idxs...), std::complex<double>{0.0, 0.0});
  });

  // vector - vector addition
  auto W_d  = V_d - V_d;
  auto W_c  = V_c - V_c;
  auto W_d2 = V_d;
  W_d2 -= V_d;
  auto W_c2 = V_c;
  W_c2 -= V_c;
  nda::for_each(vec_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(W_d(idxs...), 0.0);
    EXPECT_COMPLEX_NEAR(W_c(idxs...), std::complex<double>{0.0, 0.0});
    EXPECT_DOUBLE_EQ(W_d2(idxs...), 0.0);
    EXPECT_COMPLEX_NEAR(W_c2(idxs...), std::complex<double>{0.0, 0.0});
  });
}

TEST_F(NDAArithmeticOps, ArrayDivision) {
  // array - array division
  auto B_d  = A_d / A_d;
  auto B_c  = A_c / A_c;
  auto B_d2 = A_d;
  B_d2 /= A_d;
  auto B_c2 = A_c;
  B_c2 /= A_c;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_NEAR(B_d(idxs...), 1.0, 1e-10);
    EXPECT_COMPLEX_NEAR(B_c(idxs...), std::complex<double>{1.0, 0.0});
    EXPECT_NEAR(B_d2(idxs...), 1.0, 1e-10);
    EXPECT_COMPLEX_NEAR(B_c2(idxs...), std::complex<double>{1.0, 0.0});
  });

  // matrix - matrix division
  auto N_d  = M_d_sq / M_d_sq;
  auto N_c  = M_c_sq / M_c_sq;
  auto N_d2 = M_d_sq;
  N_d2 /= M_d_sq;
  auto N_c2 = M_c_sq;
  N_c2 /= M_c_sq;
  nda::for_each(mat_sq_shape, [&](auto... idxs) {
    EXPECT_NEAR(N_d(idxs...), (are_equal(idxs...) ? 1.0 : 0.0), 1e-10);
    EXPECT_COMPLEX_NEAR(N_c(idxs...), (are_equal(idxs...) ? std::complex<double>{1.0, 0.0} : std::complex<double>{0.0, 0.0}));
    EXPECT_NEAR(N_d2(idxs...), (are_equal(idxs...) ? 1.0 : 0.0), 1e-10);
    EXPECT_COMPLEX_NEAR(N_c2(idxs...), (are_equal(idxs...) ? std::complex<double>{1.0, 0.0} : std::complex<double>{0.0, 0.0}));
  });
}

TEST_F(NDAArithmeticOps, ArrayNegation) {
  // array negation
  auto B_d = -A_d;
  auto B_c = -A_c;
  nda::for_each(arr_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(B_d(idxs...), -A_d(idxs...));
    EXPECT_COMPLEX_NEAR(B_c(idxs...), -A_c(idxs...));
  });

  // matrix negation
  auto N_d = -M_d_rect;
  auto N_c = -M_c_rect;
  nda::for_each(mat_rect_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(N_d(idxs...), -M_d_rect(idxs...));
    EXPECT_COMPLEX_NEAR(N_c(idxs...), -M_c_rect(idxs...));
  });

  // vector negation
  auto W_d = -V_d;
  auto W_c = -V_c;
  nda::for_each(vec_shape, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(W_d(idxs...), -V_d(idxs...));
    EXPECT_COMPLEX_NEAR(W_c(idxs...), -V_c(idxs...));
  });
}

TEST_F(NDAArithmeticOps, MatrixExpressions) {
  nda::matrix<long> A(2, 2), B(2, 2), C(2, 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      A(i, j) = 10 * i + j;
      B(i, j) = i + 2 * j;
    }

  // scalar multiplication and matrix addition
  EXPECT_EQ((A + 2 * B).shape(), A.shape());
  C = A + 2 * B;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(C(i, j), A(i, j) + 2 * B(i, j));

  // matrix addition
  C = std::plus<>{}(A, B);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(C(i, j), A(i, j) + B(i, j));

  // matrix multiplication
  nda::matrix<double> A_d(2, 2), B_d(2, 2), C_d(2, 2), id(2, 2);
  A_d       = A;
  B_d       = B;
  B_d(0, 0) = 1;
  C_d()     = 0;
  id()      = 1;
  C_d       = A_d * B_d;
  EXPECT_ARRAY_NEAR(C_d, (nda::matrix<long>{{1, 3}, {21, 53}}));
  EXPECT_ARRAY_NEAR(nda::matrix<double>(A_d * B_d), (nda::matrix<long>{{1, 3}, {21, 53}}));
  EXPECT_ARRAY_NEAR(nda::matrix<double>(A_d * (B_d + C_d)), (nda::matrix<long>{{22, 56}, {262, 666}}));

  // matrix division
  EXPECT_ARRAY_NEAR(nda::matrix<double>(2 * nda::inverse(A_d)), nda::matrix<double>(2 / A_d));

  // scalar division
  EXPECT_ARRAY_NEAR(nda::matrix<double>(A_d / 2), (nda::matrix<double>{{0.0, 0.5}, {5.0, 5.5}}));
}

TEST_F(NDAArithmeticOps, ArrayExpressions) {
  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::r_is_scalar == true);
  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::l_is_scalar == false);
  static_assert(nda::expr<'/', nda::array<int, 1> &, double>::algebra == 'A');

  nda::array<int, 1> A(3), B(3), C;
  nda::array<double, 1> D;
  B = 2;
  A = 3;

  // scalar multiplication and array addition
  C = 2 * A + B;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{8, 8, 8});

  // scalar multiplication of double and integer array
  D = 2.3 * B;
  EXPECT_ARRAY_NEAR(D, nda::array<double, 1>{4.6, 4.6, 4.6});

  // scalar division of integer array and double and array addition
  D = A + B / 1.2;
  EXPECT_ARRAY_NEAR(D, nda::array<double, 1>{4.66666666666667, 4.66666666666667, 4.66666666666667});

  // long expression
  C = A + 2 * A + 3 * A - 2 * A + A - A + A + A * 3 + A + A + A + A + A + A + A + A + A + A + A + A + A;
  EXPECT_ARRAY_NEAR(C, nda::array<int, 1>{63, 63, 63});
}
