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

#include <array>
#include <complex>
#include <numeric>
#include <type_traits>
#include <vector>

// Test fixture for testing various factories and transformations.
struct NDAFactoriesTransformations : public ::testing::Test {
  protected:
  static auto constexpr czero = std::complex<double>{0.0, 0.0};
  static auto constexpr cunit = std::complex<double>{1.0, 0.0};
};

// Custom type to test array factories.
struct foo {
  int i = 0;
};

TEST_F(NDAFactoriesTransformations, Arange) {
  // test range of different sizes and with different steps
  for (auto first : nda::range(-100, 100)) {
    for (auto last : nda::range(-100, 100)) {
      for (auto step : nda::range(-100, 100)) {
        if (step == 0) continue;
        auto a = nda::arange(first, last, step);
        int n  = 0;
        for (auto i = first; step > 0 ? i < last : i > last; i = i + step) EXPECT_EQ(a[n++], i);
      }
    }
  }

  // sum of the first n integers
  for (auto n : nda::range(100)) EXPECT_EQ(nda::sum(nda::arange(n)), n * (n - 1) / 2);
}

TEST_F(NDAFactoriesTransformations, Concatenate) {
  // initialize random arrays
  auto A1 = nda::array<double, 3>::rand(2, 4, 3);
  auto B1 = nda::array<double, 3>::rand(2, 5, 3);
  auto C1 = nda::array<double, 3>::rand(2, 6, 3);

  // concatenate along first axis
  auto const ABC_axis1_concat = concatenate<1>(A1, B1, C1);

  // check resulting array
  EXPECT_EQ(ABC_axis1_concat.shape(), (std::array<long, 3>{2, 15, 3}));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 15; ++j) {
      for (int k = 0; k < 3; ++k) {
        if (j < 4) {
          EXPECT_EQ(ABC_axis1_concat(i, j, k), A1(i, j, k));
        } else if (j < 9) {
          EXPECT_EQ(ABC_axis1_concat(i, j, k), B1(i, j - 4, k));
        } else if (j < 15) {
          EXPECT_EQ(ABC_axis1_concat(i, j, k), C1(i, j - 9, k));
        }
      }
    }
  }

  // initialize random arrays
  auto A2 = nda::array<double, 3>::rand(2, 3, 4);
  auto B2 = nda::array<double, 3>::rand(2, 3, 5);
  auto C2 = nda::array<double, 3>::rand(2, 3, 6);

  // concatenate along second axis
  auto const ABC_axis2_concat = concatenate<2>(A2, B2, C2);

  // check resulting array
  EXPECT_EQ(ABC_axis2_concat.shape(), (std::array<long, 3>{2, 3, 15}));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 15; ++k) {
        if (k < 4) {
          EXPECT_EQ(ABC_axis2_concat(i, j, k), A2(i, j, k));
        } else if (k < 9) {
          EXPECT_EQ(ABC_axis2_concat(i, j, k), B2(i, j, k - 4));
        } else if (k < 15) {
          EXPECT_EQ(ABC_axis2_concat(i, j, k), C2(i, j, k - 9));
        }
      }
    }
  }
}

TEST_F(NDAFactoriesTransformations, DiagDiagonalAndTrace) {
  // double matrix
  auto diag_d = nda::array<double, 1>{1.1, 2.2, 3.3};
  auto M_d    = nda::diag(diag_d);
  EXPECT_EQ(M_d.shape(), (std::array<long, 2>{3, 3}));
  int i = 0;
  nda::for_each(M_d.shape(), [&](auto... idxs) { EXPECT_EQ(M_d(idxs...), (are_equal(idxs...) ? diag_d[i++] : 0.0)); });
  auto d_d = nda::diagonal(M_d);
  nda::for_each(d_d.shape(), [&](auto idx) { EXPECT_EQ(d_d(idx), diag_d(idx)); });
  EXPECT_DOUBLE_EQ(nda::trace(M_d), std::accumulate(diag_d.begin(), diag_d.end(), 0.0));

  // complex matrix
  const auto z = std::complex<double>{1.1, -2.2};
  auto diag_c  = std::vector<std::complex<double>>(5, z);
  auto M_c     = nda::diag(diag_c);
  EXPECT_EQ(M_c.shape(), (std::array<long, 2>{5, 5}));
  nda::for_each(M_d.shape(), [&](auto... idxs) { EXPECT_EQ(M_c(idxs...), (are_equal(idxs...) ? z : czero)); });
  auto d_c = nda::diagonal(M_c);
  nda::for_each(d_c.shape(), [&](auto idx) { EXPECT_EQ(d_c(idx), diag_c[idx]); });
  EXPECT_COMPLEX_NEAR(nda::trace(M_c), std::accumulate(diag_c.begin(), diag_c.end(), czero));
}

TEST_F(NDAFactoriesTransformations, Eye) {
  auto constexpr size = 10;

  // double unit matrix
  auto M_d = nda::eye<double>(size);
  EXPECT_EQ(M_d.shape(), (std::array<long, 2>{size, size}));
  nda::for_each(M_d.shape(), [&](auto... idxs) { EXPECT_EQ(M_d(idxs...), (are_equal(idxs...) ? 1.0 : 0.0)); });

  // complex unit matrix
  auto M_c = nda::eye<std::complex<double>>(size);
  EXPECT_EQ(M_c.shape(), (std::array<long, 2>{size, size}));
  nda::for_each(M_c.shape(), [&](auto... idxs) { EXPECT_EQ(M_c(idxs...), (are_equal(idxs...) ? cunit : 0.0)); });
}

TEST_F(NDAFactoriesTransformations, GroupIndicesView) {
  // test is_partition_of_indices function
  EXPECT_TRUE(nda::detail::is_partition_of_indices<4>(std::array{0, 1}, std::array{2, 3}));
  EXPECT_TRUE(nda::detail::is_partition_of_indices<4>(std::array{2, 1}, std::array{3, 0}));
  EXPECT_TRUE(nda::detail::is_partition_of_indices<4>(std::array{3, 1}, std::array{0, 2}));
  EXPECT_FALSE(nda::detail::is_partition_of_indices<4>(std::array{0, 1}, std::array{1, 3}));
  EXPECT_THROW(nda::detail::is_partition_of_indices<4>(std::array{0, 1}, std::array{2, 4}), std::runtime_error);
  EXPECT_THROW(nda::detail::is_partition_of_indices<4>(std::array{0, -1}, std::array{2, 3}), std::runtime_error);

  // 4D array in C order
  auto A = nda::array<int, 4>(2, 2, 2, 2);
  int i  = 0;
  nda::for_each(A.shape(), [&](auto... idxs) { A(idxs...) = i++; });
  // group 0-1 and 2-3 indices
  auto A_01_23 = nda::group_indices_view(A, nda::idx_group<0, 1>, nda::idx_group<2, 3>);
  EXPECT_EQ(A_01_23.shape(), (std::array<long, 2>{4, 4}));
  i = 0;
  nda::for_each(A_01_23.shape(), [&](auto... idxs) { EXPECT_EQ(A_01_23(idxs...), i++); });
  // group 0-1-2 and 3 indices
  auto A_012_3 = nda::group_indices_view(A, nda::idx_group<0, 1, 2>, nda::idx_group<3>);
  EXPECT_EQ(A_012_3.shape(), (std::array<long, 2>{8, 2}));
  i = 0;
  nda::for_each(A_012_3.shape(), [&](auto... idxs) { EXPECT_EQ(A_012_3(idxs...), i++); });
  // group 0 and 1-2-3 indices
  auto A_0_123 = nda::group_indices_view(A, nda::idx_group<0>, nda::idx_group<1, 2, 3>);
  EXPECT_EQ(A_0_123.shape(), (std::array<long, 2>{2, 8}));
  i = 0;
  nda::for_each(A_0_123.shape(), [&](auto... idxs) { EXPECT_EQ(A_0_123(idxs...), i++); });

  // 4D array in Fortran order
  auto B = nda::array<int, 4, nda::F_layout>(2, 2, 2, 2);
  i      = 0;
  nda::for_each_static<0, nda::Fortran_stride_order<4>>(B.shape(), [&](auto... idxs) { B(idxs...) = i++; });
  // group 0-1 and 2-3 indices
  auto B_01_23 = nda::group_indices_view(B, nda::idx_group<0, 1>, nda::idx_group<2, 3>);
  EXPECT_EQ(B_01_23.shape(), (std::array<long, 2>{4, 4}));
  i = 0;
  nda::for_each_static<0, nda::Fortran_stride_order<2>>(B_01_23.shape(), [&](auto... idxs) { EXPECT_EQ(B_01_23(idxs...), i++); });
  // group 0-1-2 and 3 indices
  auto B_012_3 = nda::group_indices_view(B, nda::idx_group<0, 1, 2>, nda::idx_group<3>);
  EXPECT_EQ(B_012_3.shape(), (std::array<long, 2>{8, 2}));
  i = 0;
  nda::for_each_static<0, nda::Fortran_stride_order<2>>(B_012_3.shape(), [&](auto... idxs) { EXPECT_EQ(B_012_3(idxs...), i++); });
  // group 0 and 1-2-3 indices
  auto B_0_123 = nda::group_indices_view(B, nda::idx_group<0>, nda::idx_group<1, 2, 3>);
  EXPECT_EQ(B_0_123.shape(), (std::array<long, 2>{2, 8}));
  i = 0;
  nda::for_each_static<0, nda::Fortran_stride_order<2>>(B_0_123.shape(), [&](auto... idxs) { EXPECT_EQ(B_0_123(idxs...), i++); });

  // 4D array in custom order
  auto constexpr order    = nda::encode(std::array{1, 3, 0, 2});
  auto constexpr order_01 = nda::encode(std::array{0, 1});
  auto C                  = nda::array<int, 4, nda::contiguous_layout_with_stride_order<order>>(2, 2, 2, 2);
  i                       = 0;
  nda::for_each_static<0, order>(C.shape(), [&](auto... idxs) { C(idxs...) = i++; });
  // group 1-3 and 0-2 indices
  auto C_13_02 = nda::group_indices_view(C, nda::idx_group<1, 3>, nda::idx_group<0, 2>);
  EXPECT_EQ(C_13_02.shape(), (std::array<long, 2>{4, 4}));
  i = 0;
  nda::for_each_static<0, order_01>(C_13_02.shape(), [&](auto... idxs) { EXPECT_EQ(C_13_02(idxs...), i++); });
  // group 1-3-0 and 2 indices
  auto C_130_2 = nda::group_indices_view(C, nda::idx_group<1, 3, 0>, nda::idx_group<2>);
  EXPECT_EQ(C_130_2.shape(), (std::array<long, 2>{8, 2}));
  i = 0;
  nda::for_each_static<0, order_01>(C_130_2.shape(), [&](auto... idxs) { EXPECT_EQ(C_130_2(idxs...), i++); });
  // group 1 and 3-0-2 indices
  auto C_1_302 = nda::group_indices_view(C, nda::idx_group<1>, nda::idx_group<3, 0, 2>);
  EXPECT_EQ(C_1_302.shape(), (std::array<long, 2>{2, 8}));
  i = 0;
  nda::for_each_static<0, order_01>(C_1_302.shape(), [&](auto... idxs) { EXPECT_EQ(C_1_302(idxs...), i++); });
}

TEST_F(NDAFactoriesTransformations, MakeViews) {
  // initialize array/matrix/vect
  auto A = nda::array<double, 2>::rand(2, 3);
  auto M = nda::matrix<double>::rand(3, 4);
  auto V = nda::vector<std::complex<double>>::rand(4);

  // array view
  auto A_av = nda::make_array_view(A);
  static_assert(std::is_same_v<decltype(A_av)::value_type, decltype(A)::value_type>);
  static_assert(nda::get_algebra<decltype(A_av)> == 'A');
  auto M_av = nda::make_array_view(M());
  static_assert(std::is_same_v<decltype(M_av)::value_type, decltype(M)::value_type>);
  static_assert(nda::get_algebra<decltype(M_av)> == 'A');
  auto V_av = nda::make_array_view(V);
  static_assert(std::is_same_v<decltype(V_av)::value_type, decltype(V)::value_type>);
  static_assert(nda::get_algebra<decltype(V_av)> == 'A');

  // const array view
  auto A_cav = nda::make_array_const_view(A());
  static_assert(std::is_same_v<decltype(A_cav)::value_type, const decltype(A)::value_type>);
  static_assert(nda::get_algebra<decltype(A_cav)> == 'A');
  auto M_cav = nda::make_array_const_view(M);
  static_assert(std::is_same_v<decltype(M_cav)::value_type, const decltype(M)::value_type>);
  static_assert(nda::get_algebra<decltype(M_cav)> == 'A');
  auto V_cav = nda::make_array_const_view(V());
  static_assert(std::is_same_v<decltype(V_cav)::value_type, const decltype(V)::value_type>);
  static_assert(nda::get_algebra<decltype(V_cav)> == 'A');

  // matrix view
  auto A_mv = nda::make_matrix_view(A);
  static_assert(std::is_same_v<decltype(A_mv)::value_type, decltype(A)::value_type>);
  static_assert(nda::get_algebra<decltype(A_mv)> == 'M');
  auto M_mv = nda::make_matrix_view(M());
  static_assert(std::is_same_v<decltype(M_mv)::value_type, decltype(M)::value_type>);
  static_assert(nda::get_algebra<decltype(M_mv)> == 'M');

  // const view
  auto A_cv = nda::make_const_view(A());
  static_assert(std::is_same_v<decltype(A_cv)::value_type, const decltype(A)::value_type>);
  static_assert(nda::get_algebra<decltype(A_cv)> == 'A');
  auto M_cv = nda::make_const_view(M);
  static_assert(std::is_same_v<decltype(M_cv)::value_type, const decltype(M)::value_type>);
  static_assert(nda::get_algebra<decltype(M_cv)> == 'M');
  auto V_cv = nda::make_const_view(V());
  static_assert(std::is_same_v<decltype(V_cv)::value_type, const decltype(V)::value_type>);
  static_assert(nda::get_algebra<decltype(V_cv)> == 'V');
}

TEST_F(NDAFactoriesTransformations, MakeRegular) {
  // make a view regular
  auto A      = nda::array<int, 4>{1, 2, 3, 4};
  A()         = 5;
  auto A_view = A();
  static_assert(std::is_same_v<decltype(A_view), nda::get_view_t<decltype(A)>>);
  auto A_reg = nda::make_regular(A_view);
  static_assert(std::is_same_v<decltype(A_reg), decltype(A)>);

  // make a lazy expression regular
  auto ex     = A + A;
  auto ex_reg = nda::make_regular(ex);
  static_assert(std::is_same_v<decltype(ex_reg), decltype(A)>);
}

TEST_F(NDAFactoriesTransformations, OnesRandAndZeros) {
  // 0D array/scalar
  EXPECT_DOUBLE_EQ(nda::ones<double>(), 1.0);
  EXPECT_COMPLEX_NEAR(nda::ones<std::complex<double>>(), cunit);
  EXPECT_EQ(nda::ones<int>(), 1);
  EXPECT_DOUBLE_EQ(nda::zeros<double>(), 0.0);
  EXPECT_COMPLEX_NEAR(nda::zeros<std::complex<double>>(), czero);
  EXPECT_EQ(nda::zeros<int>(), 0);
  EXPECT_EQ(nda::zeros<foo>().i, 0);
  auto rand_d_0d = nda::rand<>();
  EXPECT_TRUE(0 <= rand_d_0d && rand_d_0d < 1);

  // 1D array
  auto constexpr shape_1d = std::array<long, 1>{5};
  auto ones_d_1d          = nda::ones<double>(shape_1d);
  auto ones_c_1d          = nda::ones<std::complex<double>>(shape_1d[0]);
  auto ones_i_1d          = nda::ones<int>(shape_1d);
  auto zeros_d_1d         = nda::zeros<double>(shape_1d);
  auto zeros_c_1d         = nda::zeros<std::complex<double>>(shape_1d[0]);
  auto zeros_i_1d         = nda::zeros<int>(shape_1d);
  auto zeros_foo_1d       = nda::zeros<foo>(shape_1d[0]);
  auto rand_d_1d          = nda::rand<>(shape_1d[0]);
  nda::for_each(shape_1d, [&](auto idx) {
    EXPECT_DOUBLE_EQ(ones_d_1d(idx), 1.0);
    EXPECT_COMPLEX_NEAR(ones_c_1d(idx), cunit);
    EXPECT_EQ(ones_i_1d(idx), 1);
    EXPECT_DOUBLE_EQ(zeros_d_1d(idx), 0.0);
    EXPECT_COMPLEX_NEAR(zeros_c_1d(idx), czero);
    EXPECT_EQ(zeros_i_1d(idx), 0);
    EXPECT_EQ(zeros_foo_1d(idx).i, 0);
    EXPECT_TRUE(0 <= rand_d_1d(idx) && rand_d_1d(idx) < 1);
  });

  // 3D array
  auto constexpr shape_3d = std::array<long, 3>{2, 3, 4};
  auto ones_d_3d          = nda::ones<double>(shape_3d[0], shape_3d[1], shape_3d[2]);
  auto ones_c_3d          = nda::ones<std::complex<double>>(shape_3d);
  auto ones_i_3d          = nda::ones<int>(shape_3d);
  auto zeros_d_3d         = nda::zeros<double>(shape_3d[0], shape_3d[1], shape_3d[2]);
  auto zeros_c_3d         = nda::zeros<std::complex<double>>(shape_3d);
  auto zeros_i_3d         = nda::zeros<int>(shape_3d);
  auto zeros_foo_3d       = nda::zeros<foo>(shape_3d);
  auto rand_f_3d          = nda::rand<float>(shape_3d[0], shape_3d[1], shape_3d[2]);
  nda::for_each(shape_3d, [&](auto... idxs) {
    EXPECT_DOUBLE_EQ(ones_d_3d(idxs...), 1.0);
    EXPECT_COMPLEX_NEAR(ones_c_3d(idxs...), cunit);
    EXPECT_EQ(ones_i_3d(idxs...), 1);
    EXPECT_DOUBLE_EQ(zeros_d_3d(idxs...), 0.0);
    EXPECT_COMPLEX_NEAR(zeros_c_3d(idxs...), czero);
    EXPECT_EQ(zeros_i_3d(idxs...), 0);
    EXPECT_EQ(zeros_foo_3d(idxs...).i, 0);
    EXPECT_TRUE(0 <= rand_f_3d(idxs...) && rand_f_3d(idxs...) < 1);
  });
}

TEST_F(NDAFactoriesTransformations, PermutedIndicesView) {
  // initialize 4D array
  auto A = nda::rand(2, 3, 4, 5);

  // permute indices
  constexpr auto p1 = std::array{1, 2, 0, 3};
  auto A_p1         = nda::permuted_indices_view<nda::encode(p1)>(A);
  EXPECT_EQ(A_p1.shape(), nda::permutations::apply_inverse(p1, A.shape()));
  EXPECT_EQ(A_p1.strides(), nda::permutations::apply_inverse(p1, A.strides()));
  EXPECT_EQ(A_p1.stride_order(), nda::permutations::apply(p1, A.stride_order()));
  nda::for_each(A_p1.shape(), [&](auto i, auto j, auto k, auto l) { EXPECT_EQ(A_p1(i, j, k, l), A(j, k, i, l)); });

  // unpermute the indices again, i.e. apply the inverse permutation
  auto A_p1_inv = nda::permuted_indices_view<nda::encode(nda::permutations::inverse(p1))>(A_p1);
  EXPECT_EQ(A_p1_inv.shape(), A.shape());
  EXPECT_EQ(A_p1_inv.strides(), A.strides());
  EXPECT_EQ(A_p1_inv.stride_order(), A.stride_order());
  EXPECT_ARRAY_EQ(A_p1_inv, A);

  // composition of permutations
  constexpr auto p2 = std::array{0, 3, 1, 2};
  constexpr auto p3 = std::array{3, 1, 0, 2};
  static_assert(nda::permutations::compose(p2, p1) == p3);
  auto A_p3   = nda::permuted_indices_view<nda::encode(p3)>(A);
  auto A_p2p1 = nda::permuted_indices_view<nda::encode(p2)>(A_p1);
  EXPECT_EQ(A_p3.shape(), A_p2p1.shape());
  EXPECT_EQ(A_p3.strides(), A_p2p1.strides());
  EXPECT_EQ(A_p3.stride_order(), A_p2p1.stride_order());
  nda::for_each(A_p3.shape(), [&](auto i, auto j, auto k, auto l) { EXPECT_EQ(A_p3(i, j, k, l), A_p2p1(i, j, k, l)); });
}

TEST_F(NDAFactoriesTransformations, ReinterpretAddFastDimsOfSizeOne) {
  // C order
  auto A   = nda::rand(2, 3);
  auto A_v = nda::reinterpret_add_fast_dims_of_size_one<2>(A);
  static_assert(nda::is_view_v<decltype(A_v)>);
  EXPECT_EQ(A_v.shape(), (std::array<long, 4>{2, 3, 1, 1}));
  EXPECT_EQ(A_v.stride_order(), nda::permutations::identity<4>());
  nda::for_each(A.shape(), [&](auto i, auto j) { EXPECT_EQ(A_v(i, j, 0, 0), A(i, j)); });

  // Fortran order
  auto B      = nda::array<double, 2, nda::F_layout>::rand(2, 3);
  auto B_copy = B;
  auto B_v    = nda::reinterpret_add_fast_dims_of_size_one<2>(std::move(B_copy));
  static_assert(nda::is_regular_v<decltype(B_v)>);
  EXPECT_EQ(B_v.shape(), (std::array<long, 4>{2, 3, 1, 1}));
  EXPECT_EQ(B_v.stride_order(), (std::array{1, 0, 2, 3}));
  nda::for_each(B.shape(), [&](auto i, auto j) { EXPECT_EQ(B_v(i, j, 0, 0), B(i, j)); });
}

TEST_F(NDAFactoriesTransformations, Reshape) {
  // initialize C and Fortran order 2D arrays
  auto A_c = nda::array<int, 2>(4, 4);
  auto A_f = nda::array<int, 2, nda::F_layout>(4, 4);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      A_c(i, j) = i * 4 + j;
      A_f(i, j) = j * 4 + i;
    }
  }

  // reshape the arrays to the given shape and check the resulting arrays
  auto check_shape = [&](const auto &shape) {
    auto B_c = nda::reshape(A_c, shape);
    EXPECT_EQ(B_c.shape(), shape);
    int i = 0;
    nda::for_each(B_c.shape(), [&](auto... idxs) { EXPECT_EQ(B_c(idxs...), i++); });
    auto B_f = nda::reshape(A_f, shape);
    EXPECT_EQ(B_f.shape(), shape);
    i = 0;
    nda::for_each_static<0, nda::Fortran_stride_order<B_f.rank>>(B_f.shape(), [&](auto... idxs) { EXPECT_EQ(B_f(idxs...), i++); });
  };

  // reshape to 1D array
  check_shape(std::array<long, 1>{16});

  // reshape to 2D array
  check_shape(std::array<long, 2>{4, 4});
  check_shape(std::array<long, 2>{2, 8});
  check_shape(std::array<long, 2>{8, 2});
  check_shape(std::array<long, 2>{16, 1});
  check_shape(std::array<long, 2>{1, 16});

  // reshape to 3D array
  check_shape(std::array<long, 3>{2, 2, 4});
  check_shape(std::array<long, 3>{2, 4, 2});
  check_shape(std::array<long, 3>{4, 2, 2});
  check_shape(std::array<long, 3>{4, 4, 1});
  check_shape(std::array<long, 3>{4, 1, 4});
  check_shape(std::array<long, 3>{1, 4, 4});

  // reshape to 4D array
  check_shape(std::array<long, 4>{2, 2, 2, 2});
  check_shape(std::array<long, 4>{2, 2, 4, 1});
  check_shape(std::array<long, 4>{2, 4, 2, 1});
  check_shape(std::array<long, 4>{4, 2, 2, 1});
  check_shape(std::array<long, 4>{2, 1, 2, 4});
  check_shape(std::array<long, 4>{1, 2, 2, 4});
}

TEST_F(NDAFactoriesTransformations, ReshapeRvalues) {
  nda::array<long, 1> A{1, 2, 3, 4, 5, 6};
  nda::array<long, 2> A_reshaped{{1, 2}, {3, 4}, {5, 6}};

  auto B = reshape(nda::basic_array{A}, std::array{3, 2});
  EXPECT_EQ(B, A_reshaped);
  static_assert(nda::is_regular_v<decltype(B)>);

  auto C = reshape(nda::basic_array{A}, 3, 2);
  EXPECT_EQ(C, A_reshaped);
  static_assert(nda::is_regular_v<decltype(C)>);

  EXPECT_EQ(A, flatten(A));
  EXPECT_EQ(A, flatten(B));
  EXPECT_EQ(A, flatten(C));
}

TEST_F(NDAFactoriesTransformations, ReshapeLvalues) {
  nda::array<long, 1> A{1, 2, 3, 4, 5, 6};
  nda::array<long, 2> A_reshaped{{1, 2}, {3, 4}, {5, 6}};

  auto B_v = reshape(A, std::array{3, 2});
  EXPECT_EQ(B_v, A_reshaped);
  static_assert(nda::is_view_v<decltype(B_v)>);
  B_v(0, nda::range::all) *= 10;
  EXPECT_EQ(A, (nda::array<long, 1>{10, 20, 3, 4, 5, 6}));

  auto C_v = reshape(A(), 3, 2);
  EXPECT_EQ(C_v, (nda::array<long, 2>{{10, 20}, {3, 4}, {5, 6}}));
  static_assert(nda::is_view_v<decltype(C_v)>);
}

TEST_F(NDAFactoriesTransformations, ResizeOrCheckIfView) {
  auto A = nda::array<int, 2>(4, 4);

  // resize arrays
  nda::resize_or_check_if_view(A, {8, 8});
  EXPECT_EQ(A.shape(), (std::array<long, 2>{8, 8}));
  nda::resize_or_check_if_view(A, {3, 2});
  EXPECT_EQ(A.shape(), (std::array<long, 2>{3, 2}));

  // check views
  auto A_v = A();
  nda::resize_or_check_if_view(A_v, {3, 2});
  EXPECT_THROW(nda::resize_or_check_if_view(A_v, {2, 3}), nda::runtime_error);
}

TEST_F(NDAFactoriesTransformations, Transpose) {
  // initialize 4D array
  auto A           = nda::rand(2, 3, 4, 5);
  constexpr auto p = nda::permutations::reverse_identity<4>();

  // transpose array
  auto A_t = nda::transpose(A);
  EXPECT_EQ(A_t.shape(), nda::permutations::apply_inverse(p, A.shape()));
  EXPECT_EQ(A_t.strides(), nda::permutations::apply_inverse(p, A.strides()));
  EXPECT_EQ(A_t.stride_order(), nda::permutations::apply(p, A.stride_order()));
  nda::for_each(A_t.shape(), [&](auto i, auto j, auto k, auto l) { EXPECT_EQ(A_t(i, j, k, l), A(l, k, j, i)); });
}

TEST_F(NDAFactoriesTransformations, TransposedView) {
  // initialize 4D array
  auto A = nda::rand(2, 3, 4, 5);

  // get transposed view
  constexpr auto p = nda::permutations::transposition<4>(1, 3);
  auto A_t         = nda::transposed_view<1, 3>(A);
  EXPECT_EQ(A_t.shape(), nda::permutations::apply_inverse(p, A.shape()));
  EXPECT_EQ(A_t.strides(), nda::permutations::apply_inverse(p, A.strides()));
  EXPECT_EQ(A_t.stride_order(), nda::permutations::apply(p, A.stride_order()));
  nda::for_each(A_t.shape(), [&](auto i, auto j, auto k, auto l) { EXPECT_EQ(A_t(i, j, k, l), A(i, l, k, j)); });
}

TEST_F(NDAFactoriesTransformations, Vstack) {
  // initialize random arrays
  auto A1 = nda::array<double, 2, nda::F_layout>::rand(2, 4);
  auto B1 = nda::array<double, 2, nda::F_layout>::rand(3, 4);

  // concatenate along first axis
  auto const A1B1 = vstack(A1, B1);

  // check resulting array
  EXPECT_EQ(A1B1.shape(), (std::array<long, 2>{5, 4}));
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 5; ++i) {
      if (i < 2)
        EXPECT_EQ(A1B1(i, j), A1(i, j));
      else
        EXPECT_EQ(A1B1(i, j), B1(i - 2, j));
    }
  }
}
