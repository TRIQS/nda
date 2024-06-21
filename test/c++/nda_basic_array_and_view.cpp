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
#include <type_traits>
#include <vector>
#include <tuple>

// Test fixture for testing arrays and views.
struct NDAArrayAndView : public ::testing::Test {
  protected:
  using lay_3d_t = nda::contiguous_layout_with_stride_order<nda::encode(std::array{1, 2, 0})>;

  NDAArrayAndView() : A_3d(shape_3d), A_3d_v(A_3d), A_3d_cv(A_3d), A_1d(shape_1d) {
    for (int i = 0; auto &x : A_3d) x = i++;
    for (int i = 0; auto &x : A_1d) x = i++;
  }

  std::array<long, 3> shape_3d{2, 3, 4};
  nda::array<int, 3> A_3d;
  nda::array_view<int, 3> A_3d_v;
  nda::array_view<const int, 3> A_3d_cv;
  std::array<long, 1> shape_1d{5};
  nda::array<int, 1> A_1d;
};

// Test constructors of arrays and views.
template <typename T>
void check_ctors() {
  using arr_t         = T;
  using view_t        = nda::get_view_t<arr_t>;
  using const_view_t  = nda::get_view_t<const arr_t>;
  using value_t       = typename arr_t::value_type;
  using layout_t      = typename arr_t::layout_t;
  using storage_t     = typename arr_t::storage_t;
  constexpr auto rank = arr_t::rank;
  auto shape          = nda::stdutil::make_initialized_array<rank>(5l);
  auto size           = nda::stdutil::product(shape);
  auto idxm           = layout_t(shape);

  // empty array
  arr_t A1;
  EXPECT_EQ(A1.size(), 0);
  EXPECT_EQ(A1.shape(), nda::stdutil::make_initialized_array<rank>(0l));
  EXPECT_TRUE(A1.is_contiguous());
  EXPECT_TRUE(A1.empty());
  EXPECT_EQ(A1.data(), nullptr);
  EXPECT_EQ(A1.begin(), A1.end());

  // empty view
  view_t A1_v;
  EXPECT_EQ(A1_v.size(), 0);
  EXPECT_EQ(A1_v.shape(), nda::stdutil::make_initialized_array<rank>(0l));
  EXPECT_TRUE(A1_v.is_contiguous());
  EXPECT_TRUE(A1_v.empty());
  EXPECT_TRUE(A1_v.data() == nullptr);
  EXPECT_EQ(A1, A1_v);
  EXPECT_EQ(A1_v.begin(), A1_v.end());
  const_view_t A1_cv;
  static_assert(std::is_const_v<typename decltype(A1_cv)::value_type>);

  // construct an array with a specific shape
  arr_t A2(shape);
  EXPECT_EQ(A2.size(), size);
  EXPECT_EQ(A2.shape(), shape);
  EXPECT_TRUE(!A2.empty());

  // construct an array from an nda::ArrayOfRank object
  auto aor = array_of_rank<value_t, rank>(shape);
  arr_t A3(aor);
  EXPECT_EQ(A3.size(), aor.size());
  EXPECT_EQ(A3.shape(), shape);
  for (auto const &x : A3) EXPECT_EQ(x, static_cast<value_t>(rank));

  // construct an array from an nda::ArrayInitializer object
  auto arr_init = array_initializer<value_t, rank>(shape);
  arr_t A4(arr_init);
  EXPECT_EQ(A4.size(), arr_init.size());
  EXPECT_EQ(A4.shape(), shape);
  int i4 = 0;
  for (auto const &x : A4) EXPECT_EQ(x, static_cast<value_t>(i4++));

  // construct a view from an nda::MemoryArrayOfRank object
  view_t A4_v1(A4);
  EXPECT_EQ(A4, A4_v1);
  EXPECT_EQ(A4.data(), A4_v1.data());

  // construct a view from a pointer and a shape
  view_t A4_v2(A4.shape(), A4.data());
  EXPECT_EQ(A4, A4_v2);
  EXPECT_EQ(A4.data(), A4_v2.data());

  // construct an array from an nda::idx_map
  arr_t A5(idxm);
  EXPECT_EQ(A5.size(), size);
  EXPECT_EQ(A5.shape(), shape);

  // construct an array from an nda::idx_map and a memory handle
  arr_t A6(idxm, storage_t(size));
  EXPECT_EQ(A6.size(), size);
  EXPECT_EQ(A6.shape(), shape);

  // rank specific constructors
  if constexpr (rank == 1) {
    // construct an array from an initializer list
    arr_t A7{1, 2, 3, 4, 5};
    EXPECT_EQ(A7.size(), 5);
    EXPECT_EQ(A7.shape(), nda::stdutil::make_initialized_array<rank>(5l));
    for (long i = 0; i < 5; ++i) EXPECT_EQ(A7(i), static_cast<value_t>(i + 1));

    // construct an array with a specific shape and value
    long size6 = 5;
    arr_t A8(size6, static_cast<value_t>(10));
    for (long i = 0; i < size6; ++i) EXPECT_EQ(A8(i), static_cast<value_t>(10));

    // construct a view of a std::array
    std::array<value_t, 5> std_arr{1, 2, 3, 4, 5};
    view_t A9_v(std_arr);
    EXPECT_EQ(A9_v, std_arr);

    // construct a view of a std::vector
    std::vector<value_t> std_vec{1, 2, 3, 4, 5};
    view_t A10_v(std_vec);
    EXPECT_EQ(A10_v, std_vec);
  } else if constexpr (rank == 2) {
    // construct an array from an initializer list
    arr_t A7{{1, 2}, {3, 4}};
    EXPECT_EQ(A7.size(), 4);
    EXPECT_EQ(A7.shape(), nda::stdutil::make_initialized_array<rank>(2l));
    nda::for_each(A7.shape(), [&A7](auto i, auto j) { EXPECT_EQ(A7(i, j), static_cast<value_t>(i * 2 + j + 1)); });
  } else if constexpr (rank == 3) {
    // construct an array from an initializer list
    arr_t A7{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    EXPECT_EQ(A7.size(), 8);
    EXPECT_EQ(A7.shape(), nda::stdutil::make_initialized_array<rank>(2l));
    nda::for_each(A7.shape(), [&A7](auto i, auto j, auto k) { EXPECT_EQ(A7(i, j, k), static_cast<value_t>(i * 4 + j * 2 + k + 1)); });
  }
}

// Test assignment operators of arrays and views.
template <typename T>
void check_assignments() {
  using arr_t         = T;
  using view_t        = nda::get_view_t<arr_t>;
  using value_t       = typename arr_t::value_type;
  constexpr auto rank = arr_t::rank;
  auto shape          = nda::stdutil::make_initialized_array<rank>(5l);

  // construct an array with a specific shape
  auto arr_init = array_initializer<value_t, rank>(shape);
  arr_t A(arr_init);
  view_t A_v(A);

  // assign an array to an empty array of the same type (resizes the array)
  arr_t A1;
  A1 = A;
  EXPECT_EQ(A, A1);
  EXPECT_NE(A.data(), A1.data());

  // assign an array to an existing array of the same shape
  auto a1_ptr = A1.data();
  A1          = A;
  EXPECT_NE(a1_ptr, A1.data());

  // assign a view to an empty array
  arr_t A2;
  A2 = A_v;
  EXPECT_EQ(A, A2);
  EXPECT_NE(A.data(), A2.data());

  // assign an nda::ArrayOfRank object to an empty array
  auto aor = array_of_rank<value_t, rank>(shape);
  arr_t A3;
  A3 = aor;
  EXPECT_EQ(A3.size(), aor.size());
  EXPECT_EQ(A3.shape(), shape);
  for (auto const &x : A3) EXPECT_EQ(x, static_cast<value_t>(rank));

  // assign a scalar to an empty array
  arr_t A4(shape);
  A4 = static_cast<value_t>(2);
  for (auto const &x : A4) EXPECT_EQ(x, static_cast<value_t>(2));

  // use an array initializer to assign to an array
  arr_t A5;
  A5 = arr_init;
  EXPECT_EQ(A, A5);

  // copy assign to a view
  arr_t A6(shape);
  view_t A6_v(A6);
  A6_v = A_v;
  EXPECT_EQ(A, A5);
  EXPECT_NE(A.data(), A5.data());

  // assign an nda::ArrayOfRank object to a view
  arr_t A7(shape);
  A7.as_array_view() = aor;
  EXPECT_EQ(A7.size(), aor.size());
  EXPECT_EQ(A7.shape(), shape);
  for (auto const &x : A7) EXPECT_EQ(x, static_cast<value_t>(rank));

  // assign an array to a view
  arr_t A8(shape);
  view_t A8_v(A8);
  A8_v = A;
  EXPECT_EQ(A, A8);

  // assign a scalar to a view
  arr_t A9(shape);
  view_t A9_v(A9);
  A9_v = static_cast<value_t>(2);
  for (auto const &x : A9) EXPECT_EQ(x, static_cast<value_t>(2));

  // use an array initializer to assign to a view
  arr_t A10(shape);
  view_t A10_v(A10);
  A10_v = arr_init;
  EXPECT_EQ(A, A10);

  // rank specific assignments
  if constexpr (rank == 1) {
    std::vector<value_t> vec(shape[0]);
    std::iota(vec.begin(), vec.end(), 0);

    // assign a contiguous range to an array
    arr_t A11;
    A11 = vec;
    for (long i = 0; i < shape[0]; ++i) EXPECT_EQ(A11(i), vec[i]);

    // assign a contiguous range to a view
    arr_t A12(shape);
    view_t A12_v(A12);
    A12_v = vec;
    for (long i = 0; i < shape[0]; ++i) EXPECT_EQ(A12(i), vec[i]);

    // assign a const vector to a view
    const std::vector<value_t> cvec{1, 2, 3};
    arr_t A13(3);
    view_t A13_v(A13);
    A13_v = cvec;
    for (long i = 0; i < 3; ++i) EXPECT_EQ(A13(i), cvec[i]);
  }
}

TEST_F(NDAArrayAndView, Constructors) {
  // C layout
  check_ctors<nda::array<int, 1>>();
  check_ctors<nda::array<long, 2>>();
  check_ctors<nda::array<double, 3>>();
  check_ctors<nda::array<std::complex<double>, 1>>();
  check_ctors<nda::array<std::complex<float>, 2>>();
  check_ctors<nda::array<std::complex<double>, 3>>();
  check_ctors<nda::array<foo_def_ctor, 1>>();
  check_ctors<nda::array<foo_def_ctor, 2>>();
  check_ctors<nda::array<foo_def_ctor, 3>>();

  // Fortran layout
  check_ctors<nda::array<long, 2, nda::F_layout>>();
  check_ctors<nda::array<double, 3, nda::F_layout>>();
  check_ctors<nda::array<std::complex<float>, 2, nda::F_layout>>();
  check_ctors<nda::array<std::complex<double>, 3, nda::F_layout>>();
  check_ctors<nda::array<foo_def_ctor, 2, nda::F_layout>>();
  check_ctors<nda::array<foo_def_ctor, 3, nda::F_layout>>();

  // custom layout
  check_ctors<nda::array<double, 3, lay_3d_t>>();
  check_ctors<nda::array<std::complex<double>, 3, lay_3d_t>>();
  check_ctors<nda::array<foo_def_ctor, 3, lay_3d_t>>();
}

TEST_F(NDAArrayAndView, CopyAndMoveConstructors) {
  // copy constructor for arrays
  auto B_copy = A_3d;
  EXPECT_EQ(A_3d, B_copy);
  EXPECT_NE(A_3d.data(), B_copy.data());
  B_copy(0, 0, 0) = 42;
  EXPECT_NE(A_3d, B_copy);
  B_copy(0, 0, 0) = A_3d(0, 0, 0);

  // move constructor for arrays
  auto B_mv = std::move(B_copy);
  EXPECT_EQ(A_3d, B_mv);
  EXPECT_TRUE(B_copy.empty());
  EXPECT_EQ(B_copy.data(), nullptr);
  EXPECT_EQ(B_copy.shape(), B_mv.shape());

  // copy constructor for views
  auto C_copy = A_3d_v;
  EXPECT_EQ(A_3d, C_copy);
  EXPECT_EQ(A_3d.data(), C_copy.data());
  EXPECT_EQ(A_3d_v.data(), C_copy.data());
  EXPECT_EQ(A_3d_v.shape(), C_copy.shape());

  // move constructor for views
  auto C_mv = std::move(A_3d_v);
  EXPECT_EQ(A_3d, C_mv);
  EXPECT_EQ(A_3d.data(), C_mv.data());
  EXPECT_EQ(A_3d_v.data(), C_mv.data());
  EXPECT_EQ(A_3d_v.shape(), C_mv.shape());
}

TEST_F(NDAArrayAndView, ConstructFromInitializerList) {
  // matrix
  nda::matrix<double> M = {{1, 2}, {3, 4}, {5, 6}};
  EXPECT_EQ(M.shape(), (shape_t<2>{3, 2}));
  for (int i = 1; auto x : M) EXPECT_EQ(x, i++);

  // array of an std::array
  nda::array<std::array<double, 2>, 1> A{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  EXPECT_EQ(A(0), (std::array<double, 2>{0.0, 0.0}));
  EXPECT_EQ(A(1), (std::array<double, 2>{0.0, 1.0}));
  EXPECT_EQ(A(2), (std::array<double, 2>{1.0, 0.0}));
  EXPECT_EQ(A(3), (std::array<double, 2>{1.0, 1.0}));
}

TEST_F(NDAArrayAndView, CrossConstructDoubleFromIntArray) {
  // construct a double array from an int array
  nda::array<int, 1> Vi(3);
  Vi() = 3;
  nda::array<double, 1> Vd(Vi);
  EXPECT_ARRAY_NEAR(Vd, Vi);
}

TEST_F(NDAArrayAndView, CrossConstructComplexFromDoubleArray) {
  // construct a complex array from a double array
  nda::array<double, 2> A_d(2, 2);
  nda::array<std::complex<double>, 2> A_c(2, 2);

  // double from complex does not compile
  // A = B;

  // assign double array to complex array
  A_c = A_d;
  EXPECT_ARRAY_NEAR(A_c, A_d, 1e-14);

  // construct a complex array from a double array
  auto B_c = nda::array<std::complex<double>, 2>{A_d};
  EXPECT_ARRAY_NEAR(A_c, B_c);

  static_assert(std::is_constructible_v<nda::array<std::complex<double>, 2>, nda::array<double, 2>>);
  static_assert(not std::is_constructible_v<nda::array<double, 2>, nda::array<std::complex<double>, 2>>);
}

TEST_F(NDAArrayAndView, ConstructViewsFromArrays) {
  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = 10 * i + j;

  std::vector<nda::array<long, 2>> vec_A(3, A);

  std::vector<nda::array_view<long, 2>> vec_A_v;
  for (auto &a : vec_A) vec_A_v.emplace_back(a);

  std::vector<nda::array_view<long, 2>> vec_A_v2(vec_A_v);

  for (int i = 1; i < 3; ++i) vec_A[i] *= i;
  for (int i = 1; i < 3; ++i) EXPECT_ARRAY_NEAR(vec_A_v2[i], i * A);
}

TEST_F(NDAArrayAndView, Assignments) {
  // C layout
  check_assignments<nda::array<int, 1>>();
  check_assignments<nda::array<long, 2>>();
  check_assignments<nda::array<double, 3>>();
  check_assignments<nda::array<std::complex<double>, 1>>();
  check_assignments<nda::array<std::complex<float>, 2>>();
  check_assignments<nda::array<std::complex<double>, 3>>();
  check_assignments<nda::array<foo_def_ctor, 1>>();
  check_assignments<nda::array<foo_def_ctor, 2>>();
  check_assignments<nda::array<foo_def_ctor, 3>>();

  // Fortran layout
  check_ctors<nda::array<long, 2, nda::F_layout>>();
  check_ctors<nda::array<double, 3, nda::F_layout>>();
  check_ctors<nda::array<std::complex<float>, 2, nda::F_layout>>();
  check_ctors<nda::array<std::complex<double>, 3, nda::F_layout>>();
  check_ctors<nda::array<foo_def_ctor, 2, nda::F_layout>>();
  check_ctors<nda::array<foo_def_ctor, 3, nda::F_layout>>();

  // custom layout
  check_ctors<nda::array<double, 3, lay_3d_t>>();
  check_ctors<nda::array<std::complex<double>, 3, lay_3d_t>>();
  check_ctors<nda::array<foo_def_ctor, 3, lay_3d_t>>();
}

TEST_F(NDAArrayAndView, AssignFortranToCArray) {
  nda::array<long, 2, nda::F_layout> A_f(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) A_f(i, j) = 10 * i + j;

  // copy to array with C layout
  nda::array<long, 2> B_c(A_f);

  // assign to array with C layout
  nda::array<long, 2> C_c;
  C_c = A_f;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(C_c(i, j), 10 * i + j);
      EXPECT_EQ(B_c(i, j), 10 * i + j);
    }
}

TEST_F(NDAArrayAndView, AssignScalarToFortranMatrixAndArray) {
  const int size = 5;
  nda::matrix<int, nda::F_layout> M(size, size);
  M() = 2;
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j) EXPECT_EQ(M(i, j), (i == j ? 2 : 0));

  nda::array_view<int, 2, nda::F_layout> A_v(M);
  A_v = 2;
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j) EXPECT_EQ(M(i, j), 2);
}

TEST_F(NDAArrayAndView, AssignStridedViewsToArray) {
  nda::array<long, 3> A(3, 5, 9);
  nda::array<long, 1> B;
  for (int i = 0; i < A.extent(0); ++i)
    for (int j = 0; j < A.extent(1); ++j)
      for (int k = 0; k < A.extent(2); ++k) A(i, j, k) = 1 + i + 10 * j + 100 * k;

  B = A(nda::range::all, 0, 1);
  for (int i = 0; i < A.extent(0); ++i) EXPECT_EQ(A(i, 0, 1), B(i));
  static_assert(nda::has_layout_strided_1d<decltype(A(nda::range::all, 0, 1))>);

  B = A(1, nda::range::all, 2);
  for (int i = 0; i < A.extent(1); ++i) EXPECT_EQ(A(1, i, 2), B(i));
  static_assert(nda::has_layout_strided_1d<decltype(A(1, nda::range::all, 2))>);

  B = A(1, 3, nda::range::all);
  for (int i = 0; i < A.extent(2); ++i) EXPECT_EQ(A(1, 3, i), B(i));
  static_assert(nda::has_contiguous_layout<decltype(A(1, 3, nda::range::all))>);
}

TEST_F(NDAArrayAndView, AssignStridedViewsToViews) {
  nda::array<long, 3> A(3, 5, 9);
  nda::array<long, 1> B;
  for (int i = 0; i < A.extent(0); ++i)
    for (int j = 0; j < A.extent(1); ++j)
      for (int k = 0; k < A.extent(2); ++k) A(i, j, k) = 1 + i + 10 * j + 100 * k;
  B.resize(20);
  B = 0;

  B(nda::range(0, 2 * A.extent(0), 2)) = A(nda::range::all, 0, 1);
  for (int i = 0; i < A.extent(0); ++i) EXPECT_EQ(A(i, 0, 1), B(2 * i));

  B(nda::range(0, 2 * A.extent(1), 2)) = A(1, nda::range::all, 2);
  for (int i = 0; i < A.extent(1); ++i) EXPECT_EQ(A(1, i, 2), B(2 * i));

  B(nda::range(0, 2 * A.extent(2), 2)) = A(1, 3, nda::range::all);
  for (int i = 0; i < A.extent(2); ++i) EXPECT_EQ(A(1, 3, i), B(2 * i));
}

TEST_F(NDAArrayAndView, CrossConstructAndAssignToDifferentLayouts) {
  nda::array<long, 3, nda::F_layout> A_f(A_3d), A_f2(2, 3, 4);
  A_f2 = A_3d;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(A_f(i, j, k), A_3d(i, j, k));
        EXPECT_EQ(A_f2(i, j, k), A_3d(i, j, k));
      }
    }
  }
}

TEST_F(NDAArrayAndView, FullAccess) {
  // full view on array
  auto A_v = A_3d();
  EXPECT_EQ(A_v, A_3d);

  // full view on view
  auto A_vv = A_3d_v();
  EXPECT_EQ(A_vv, A_3d);

  // full view on const view
  auto A_cv = A_3d_cv();
  static_assert(std::is_const_v<typename decltype(A_cv)::value_type>);
  EXPECT_EQ(A_cv, A_3d);

  // full view on view on const view
  auto A_cvv = A_cv();
  static_assert(std::is_const_v<typename decltype(A_cvv)::value_type>);
  EXPECT_EQ(A_cvv, A_3d);
}

TEST_F(NDAArrayAndView, SingleElementAccess) {
  for (long i = 0; i < 2; ++i) {
    for (long j = 0; j < 3; ++j) {
      for (long k = 0; k < 4; ++k) {
        EXPECT_EQ(A_3d(i, j, k), i * 12 + j * 4 + k);
        EXPECT_EQ(A_3d_v(i, j, k), i * 12 + j * 4 + k);
      }
    }
  }
  static_assert(std::is_reference_v<decltype(A_3d(1, 2, 3))>);
  static_assert(std::is_reference_v<decltype(A_3d_v(1, 2, 3))>);
  static_assert(!std::is_reference_v<decltype(std::move(A_3d)(1, 2, 3))>);

  // assign to a single element
  auto B = A_3d;
  EXPECT_EQ(B, A_3d);
  B(0, 1, 2) = 42;
  EXPECT_NE(B, A_3d);

  // assign to a single element of a view
  A_3d_v(0, 1, 2) = 42;
  EXPECT_EQ(B, A_3d);

  // assign to a single element of a const view -> does not compile
  // A_3d_cv(0, 1, 2) = 42;
}

TEST_F(NDAArrayAndView, LazyAccess) {
  using namespace nda::clef::literals;
  auto A_lazy = A_3d(i_, j_, k_);
  static_assert(nda::clef::is_lazy<decltype(A_lazy)>);
  EXPECT_EQ(nda::clef::eval(A_lazy, i_ = 0, j_ = 1, k_ = 2), A_3d(0, 1, 2));
  auto A_v_lazy = A_3d_v(i_, j_, k_);
  static_assert(nda::clef::is_lazy<decltype(A_v_lazy)>);
  EXPECT_EQ(nda::clef::eval(A_v_lazy, i_ = 0, j_ = 1, k_ = 2), A_3d(0, 1, 2));
  auto A_const_v_lazy = A_3d_cv(i_, j_, k_);
  static_assert(nda::clef::is_lazy<decltype(A_const_v_lazy)>);
  EXPECT_EQ(nda::clef::eval(A_const_v_lazy, i_ = 0, j_ = 1, k_ = 2), A_3d(0, 1, 2));
}

TEST_F(NDAArrayAndView, SliceAccessFull) {
  auto A_s = A_3d(nda::ellipsis{});
  EXPECT_EQ(A_s.shape(), shape_3d);
  EXPECT_EQ(A_s, A_3d);
  auto A_s2 = A_3d(nda::range::all, nda::range::all, nda::range::all);
  EXPECT_EQ(A_s2.shape(), shape_3d);
  EXPECT_EQ(A_s2, A_3d);
}

TEST_F(NDAArrayAndView, SliceAccess1D) {
  // full 1D slice
  auto B   = A_3d;
  auto B_s = B(0, nda::range::all, 2);
  static_assert(B_s.rank == 1);
  EXPECT_EQ(B_s.size(), 3);
  EXPECT_EQ(B_s.strides(), (std::array<long, 1>{4}));
  for (long i = 0; i < 3; ++i) EXPECT_EQ(B_s(i), A_3d(0, i, 2));
  B_s = 42;
  EXPECT_NE(B, A_3d);
  for (long i = 0; i < 3; ++i) EXPECT_EQ(B(0, i, 2), 42);
  // with ellipsis
  auto B_s2 = B(0, nda::ellipsis{}, 2);
  EXPECT_EQ(B_s2, B_s);

  // partial 1D slice
  auto C   = A_3d;
  auto C_s = C(0, 1, nda::range(1, 4, 2));
  static_assert(C_s.rank == 1);
  EXPECT_EQ(C_s.size(), 2);
  EXPECT_EQ(C_s.strides(), (std::array<long, 1>{2}));
  for (long i = 0; i < 2; ++i) EXPECT_EQ(C_s(i), A_3d(0, 1, 1 + 2 * i));
  C_s = 42;
  EXPECT_NE(C, A_3d);
}

TEST_F(NDAArrayAndView, SliceAccess2D) {
  // full 2D slices
  auto B   = A_3d;
  auto B_s = B(nda::range::all, 1, nda::range::all);
  static_assert(B_s.rank == 2);
  EXPECT_EQ(B_s.shape(), (std::array<long, 2>{2, 4}));
  EXPECT_EQ(B_s.strides(), (std::array<long, 2>{12, 1}));
  for (long i = 0; i < 2; ++i) {
    for (long j = 0; j < 4; ++j) EXPECT_EQ(B_s(i, j), A_3d(i, 1, j));
  }
  B_s = 42;
  EXPECT_NE(B, A_3d);

  B         = A_3d;
  auto B_s2 = B(nda::range::all, nda::range::all, 2);
  static_assert(B_s2.rank == 2);
  EXPECT_EQ(B_s2.shape(), (std::array<long, 2>{2, 3}));
  EXPECT_EQ(B_s2.strides(), (std::array<long, 2>{12, 4}));
  for (long i = 0; i < 2; ++i) {
    for (long j = 0; j < 3; ++j) EXPECT_EQ(B_s2(i, j), A_3d(i, j, 2));
  }
  B_s2 = 42;
  EXPECT_NE(B, A_3d);
  // with ellipsis
  auto B_s3 = B(nda::ellipsis{}, 2);
  EXPECT_EQ(B_s3, B_s2);

  // partial 2D slice
  auto C   = A_3d;
  auto C_s = C(0, nda::range(0, 2), nda::range(1, 4, 2));
  static_assert(C_s.rank == 2);
  EXPECT_EQ(C_s.shape(), (std::array<long, 2>{2, 2}));
  EXPECT_EQ(C_s.strides(), (std::array<long, 2>{4, 2}));
  for (long i = 0; i < 2; ++i) {
    for (long j = 0; j < 2; ++j) EXPECT_EQ(C_s(i, j), A_3d(0, i, 1 + 2 * j));
  }
  C_s = 42;
  EXPECT_NE(C, A_3d);

  C         = A_3d;
  auto C_s2 = C(nda::range(0, 1), 1, nda::range(2, 4));
  static_assert(C_s2.rank == 2);
  EXPECT_EQ(C_s2.shape(), (std::array<long, 2>{1, 2}));
  EXPECT_EQ(C_s2.strides(), (std::array<long, 2>{12, 1}));
  for (long j = 0; j < 2; ++j) EXPECT_EQ(C_s2(0, j), A_3d(0, 1, 2 + j));
  C_s2 = 42;
  EXPECT_NE(C, A_3d);

  C         = A_3d;
  auto C_s3 = C(nda::range::all, nda::range(0, 3, 2), 3);
  static_assert(C_s3.rank == 2);
  EXPECT_EQ(C_s3.shape(), (std::array<long, 2>{2, 2}));
  EXPECT_EQ(C_s3.strides(), (std::array<long, 2>{12, 8}));
}

TEST_F(NDAArrayAndView, SliceAccess3D) {
  // partial 3D slice
  auto B   = A_3d;
  auto B_s = B(nda::range::all, nda::range(0, 2), nda::range(1, 4, 2));
  static_assert(B_s.rank == 3);
  EXPECT_EQ(B_s.shape(), (std::array<long, 3>{2, 2, 2}));
  EXPECT_EQ(B_s.strides(), (std::array<long, 3>{12, 4, 2}));
  for (long i = 0; i < 2; ++i) {
    for (long j = 0; j < 2; ++j) {
      for (long k = 0; k < 2; ++k) EXPECT_EQ(B_s(i, j, k), A_3d(i, j, 1 + 2 * k));
    }
  }
  B_s = 42;
  EXPECT_NE(B, A_3d);

  B         = A_3d;
  auto B_s2 = B(nda::range::all, nda::range(1, 3), nda::range(2, 4));
  static_assert(B_s2.rank == 3);
  EXPECT_EQ(B_s2.shape(), (std::array<long, 3>{2, 2, 2}));
  EXPECT_EQ(B_s2.strides(), (std::array<long, 3>{12, 4, 1}));
  for (long i = 0; i < 2; ++i) {
    for (long j = 0; j < 2; ++j) {
      for (long k = 0; k < 2; ++k) EXPECT_EQ(B_s2(i, j, k), A_3d(i, 1 + j, 2 + k));
    }
  }
  B_s2 = 42;
  EXPECT_NE(B, A_3d);

  B         = A_3d;
  auto B_s3 = B(nda::ellipsis{}, nda::range(0, 4, 3));
  static_assert(B_s3.rank == 3);
  EXPECT_EQ(B_s3.shape(), (std::array<long, 3>{2, 3, 2}));
  EXPECT_EQ(B_s3.strides(), (std::array<long, 3>{12, 4, 3}));
  for (long i = 0; i < 2; ++i) {
    for (long j = 0; j < 3; ++j) {
      for (long k = 0; k < 2; ++k) EXPECT_EQ(B_s3(i, j, k), A_3d(i, j, 3 * k));
    }
  }
  B_s3 = 42;
  EXPECT_NE(B, A_3d);

  B         = A_3d;
  auto B_s4 = B(nda::range(0, 1), nda::ellipsis{});
  static_assert(B_s4.rank == 3);
  EXPECT_EQ(B_s4.shape(), (std::array<long, 3>{1, 3, 4}));
  EXPECT_EQ(B_s4.strides(), (std::array<long, 3>{12, 4, 1}));
  for (long j = 0; j < 3; ++j) {
    for (long k = 0; k < 4; ++k) EXPECT_EQ(B_s4(0, j, k), A_3d(0, j, k));
  }
  B_s4 = 42;
  EXPECT_NE(B, A_3d);
}

TEST_F(NDAArrayAndView, SliceAccessWithNegativeLastElement) {
  // -1 means the last element
  auto A_v  = A_1d(nda::range(0, -1, 2));
  auto A_v2 = A_1d(nda::range(0, 5, 2));
  EXPECT_EQ(A_v, A_v2);

  // > -1 means empty
  auto B_v = A_1d(nda::range(0, -2, 2));
  EXPECT_EQ(B_v.size(), 0);
}

TEST_F(NDAArrayAndView, AccessViaSubscriptOperator) {
  // single element access
  auto B = A_1d;
  for (long i = 0; i < 5; ++i) EXPECT_EQ(B[i], i);
  static_assert(std::is_reference_v<decltype(B[0])>);
  static_assert(!std::is_reference_v<decltype(std::move(B)[0])>);
  B[0] = 42;
  EXPECT_EQ(B[0], 42);
  EXPECT_NE(B, A_1d);

  // lazy access
  using namespace nda::clef::literals;
  auto A_lazy = A_1d[i_];
  static_assert(nda::clef::is_lazy<decltype(A_lazy)>);
  EXPECT_EQ(nda::clef::eval(A_lazy, i_ = 2), A_1d(2));

  // full access
  B        = A_1d;
  auto B_s = B[nda::range::all];
  EXPECT_EQ(B_s, A_1d);
  B_s = 42;
  EXPECT_NE(B, A_1d);

  // partial access
  B         = A_1d;
  auto B_s2 = B[nda::range(0, 5, 2)];
  EXPECT_EQ(B_s2.size(), 3);
  for (long i = 0; i < 3; ++i) EXPECT_EQ(B_s2[i], A_1d[2 * i]);
  B_s2 = 42;
  EXPECT_EQ(B, (nda::array<int, 1>{42, 1, 42, 3, 42}));
}

TEST_F(NDAArrayAndView, Indices) {
  auto indices = A_3d_v.indices();
  auto it      = indices.begin();
  nda::for_each(shape_3d, [&](auto... idxs) { EXPECT_EQ(*it++, std::make_tuple(idxs...)); });
}

TEST_F(NDAArrayAndView, EmptyIterator) {
  nda::array<int, 1> A;
  int c = 0;
  for (auto i : A) c += i;
  EXPECT_EQ(c, 0);
}

TEST_F(NDAArrayAndView, Contiguous1DIterator) {
  for (long c = 0; auto x : A_1d) { EXPECT_EQ(x, c++); }
}

TEST_F(NDAArrayAndView, Contiguous2DIterator) {
  nda::array<long, 2> A(2, 3);
  for (int i = 0; i < A.extent(0); ++i)
    for (int j = 0; j < A.extent(1); ++j) A(i, j) = 1 + i + 10 * j;

  auto it = A.begin();
  for (int i = 0; i < A.extent(0); ++i) {
    for (int j = 0; j < A.extent(1); ++j) {
      EXPECT_EQ(*it, A(i, j));
      EXPECT_FALSE(it == A.end());
      ++it;
    }
  }
  EXPECT_TRUE(it == A.end());
}

TEST_F(NDAArrayAndView, Contiguous3DIterator) {
  auto it = A_3d.begin();
  for (int i = 0; i < shape_3d[0]; ++i) {
    for (int j = 0; j < shape_3d[1]; ++j) {
      for (int k = 0; k < shape_3d[2]; ++k) {
        EXPECT_EQ(*it, A_3d(i, j, k));
        EXPECT_TRUE(it != A_3d.end());
        ++it;
      }
    }
  }
  EXPECT_TRUE(it == A_3d.end());
}

TEST_F(NDAArrayAndView, Strided3DIterator) {
  auto A_v = A_3d(nda::range::all, nda::range(0, 3, 2), nda::range(1, 4, 2));
  auto it  = A_v.begin();
  for (int i = 0; i < A_v.extent(0); ++i)
    for (int j = 0; j < A_v.extent(1); ++j)
      for (int k = 0; k < A_v.extent(2); ++k) {
        EXPECT_EQ(*it, A_v(i, j, k));
        EXPECT_TRUE(it != A_v.end());
        ++it;
      }
  EXPECT_TRUE(it == A_v.end());
}

TEST_F(NDAArrayAndView, BlockStrided3DIterator) {
  auto A   = nda::rand<>(3, 4, 5);
  auto A_v = A(nda::range::all, nda::range(0, 4, 2), nda::range::all);
  EXPECT_TRUE(get_block_layout(A(nda::range::all, nda::range(2), nda::range::all)));
  EXPECT_TRUE(get_block_layout(A(nda::range::all, nda::range(3), nda::range::all)));
  EXPECT_TRUE(get_block_layout(A(nda::range::all, nda::range(0, 4, 2), nda::range::all)));
  EXPECT_TRUE(!get_block_layout(A(nda::range::all, nda::range(0, 4, 3), nda::range::all)));

  // get block layout
  auto [n_blocks, block_size, block_str] = get_block_layout(A_v).value();
  EXPECT_EQ(n_blocks * block_size, A_v.size());

  // loop over the view with blocked layout and compare to pointer arithmetic based on the block size and stride
  auto *ptr = A.data();
  for (auto [n, val] : itertools::enumerate(A_v)) {
    auto [bl_idx, inner_idx] = std::ldiv(n, block_size);
    EXPECT_EQ(val, *(ptr + bl_idx * block_str + inner_idx));
  }
}

// Dummy structs and functions to test non-ambiguity in overloaded functions w.r.t. array value types.
struct A {};
struct B {};
char get_value_type(nda::array<A, 1>) { return 'A'; }
char get_value_type(nda::array<B, 1>) { return 'B'; }

TEST_F(NDAArrayAndView, AmbiguityOfOverloadedFunctions) {
  nda::array<A, 1> A(2);
  auto A_v = A();
  EXPECT_EQ(get_value_type(A_v), 'A');
}

TEST_F(NDAArrayAndView, StrideOrderOfArrays) {
  const int size = 5;
  nda::matrix<int> M_c(size, size);
  nda::matrix<int, nda::F_layout> M_f(size, size);
  nda::array<int, 1> v_c(size);
  nda::array<int, 1, nda::F_layout> v_f(size);

  EXPECT_FALSE(M_c.indexmap().is_stride_order_Fortran());
  EXPECT_TRUE(M_f.indexmap().is_stride_order_Fortran());
  EXPECT_TRUE(v_c.indexmap().is_stride_order_Fortran());
  EXPECT_TRUE(v_f.indexmap().is_stride_order_Fortran());

  EXPECT_TRUE(M_c.indexmap().is_stride_order_C());
  EXPECT_FALSE(M_f.indexmap().is_stride_order_C());
  EXPECT_TRUE(v_c.indexmap().is_stride_order_C());
  EXPECT_TRUE(v_f.indexmap().is_stride_order_C());
}

#if defined(__has_feature)
#if !__has_feature(address_sanitizer)
TEST_F(NDAArrayAndView, BadAlloc) {
  EXPECT_THROW(nda::vector<int>(long(1e16)), std::bad_alloc);
}
#endif
#endif
