// Copyright (c) 2019-2023 Simons Foundation
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

#include <gtest/gtest.h>

#include <nda/clef/literals.hpp>
#include <nda/gtest_tools.hpp>
#include <nda/h5.hpp>
#include <nda/nda.hpp>

#include <h5/h5.hpp>
#include <hdf5_hl.h>

using namespace nda::clef::literals;
using nda::ellipsis;
using nda::range;

// define HDF5 serializable type
class foo {
  int i{};

  public:
  // constructors
  foo() = default;
  foo(int i) : i(i) {}

  // get hdf5_format tag
  static std::string hdf5_format() { return "foo"; }

  // write to HDF5
  friend void h5_write(h5::group g, const std::string &subgroup_name, const foo &f) {
    auto sg = g.create_group(subgroup_name);
    h5::write_hdf5_format(sg, f); // NOLINT (slicing is fine here)
    h5::write(sg, "i", f.i);
  }

  // read from HDF5
  friend void h5_read(h5::group g, const std::string &subgroup_name, foo &f) {
    auto sg = g.open_group(subgroup_name);
    h5::assert_hdf5_format(sg, f);
    h5::read(sg, "i", f.i);
  }

  // stream operator
  friend std::ostream &operator<<(std::ostream &os, const foo &f) {
    os << f.i << "/n";
    return os;
  }

  // equal-to operator
  friend bool operator==(const foo &lhs, const foo &rhs) { return lhs.i == rhs.i; }
};

// check any type of 1d arrays and views
template <nda::MemoryArray A>
void check_1d_arrays_and_views(const A &a, h5::group group) {
  auto size = a.size();

  // write/read full array
  h5::write(group, "a", a);
  A a2;
  h5::read(group, "a", a2);
  EXPECT_EQ_ARRAY(a, a2);

  // write/read full view
  h5::write(group, "a(ellipsis{})", a(ellipsis{}));
  A a3;
  h5::read(group, "a(ellipsis{})", a3);
  EXPECT_EQ_ARRAY(a, a3);

  // write/read various views
  auto half_size = size / 2;
  h5::write(group, "a(range(half_size))", a(range(half_size)));
  h5::write(group, "a(range(half_size, size))", a(range(half_size, size)));
  A a4(size);
  auto a4_view1 = a4(range(half_size));
  h5::read(group, "a(range(half_size))", a4_view1);
  auto a4_view2 = a4(range(half_size, size));
  h5::read(group, "a(range(half_size, size))", a4_view2);
  EXPECT_EQ_ARRAY(a, a4);

  h5::write(group, "a(range(0, size, 2))", a(range(0, size, 2)));
  h5::write(group, "a(range(1, size, 2))", a(range(1, size, 2)));
  A a5(size);
  auto a5_view1 = a5(range(0, size, 2));
  h5::read(group, "a(range(0, size, 2))", a5_view1);
  auto a5_view2 = a5(range(1, size, 2));
  h5::read(group, "a(range(1, size, 2))", a5_view2);
  EXPECT_EQ_ARRAY(a, a5);

  // write/read a scalar view
  h5::write(group, "a(half_size)", a(half_size));
  typename A::value_type scalar;
  h5::read(group, "a(half_size)", scalar);
  EXPECT_EQ(a(half_size), scalar);

  // write/read slices
  if constexpr (nda::is_scalar_v<typename A::value_type>) {
    auto g      = h5::group(group);
    auto g2     = g.create_group("slices");
    int ds_rank = 1;
    h5::v_t h5_shape{static_cast<h5::hsize_t>(size), 2};
    constexpr auto is_complex = nda::is_complex_v<typename A::value_type>;
    if constexpr (is_complex) { ds_rank = 2; }

    // write/read full slice
    auto sp6 = H5Screate_simple(ds_rank, h5_shape.data(), nullptr);
    auto ds6 = g2.create_dataset("ds6", h5::hdf5_type<nda::get_value_t<A>>(), sp6);
    if (is_complex) h5::write_attribute(ds6, "__complex__", "1");
    h5::write(g2, "ds6", a, std::make_tuple(ellipsis{}));
    A a6;
    h5::read(g2, "ds6", a6, std::make_tuple(ellipsis{}));
    EXPECT_EQ_ARRAY(a, a6);

    // write/read every second element
    auto sp7 = H5Screate_simple(ds_rank, h5_shape.data(), nullptr);
    auto ds7 = g2.create_dataset("ds7", h5::hdf5_type<nda::get_value_t<A>>(), sp7);
    if (is_complex) h5::write_attribute(ds7, "__complex__", "1");
    h5::write(g2, "ds7", a(range(0, size, 2)), std::make_tuple(range(0, size, 2)));
    h5::write(g2, "ds7", a(range(1, size, 2)), std::make_tuple(range(1, size, 2)));
    A a7(size);
    auto a7_view1 = a7(range(0, size, 2));
    h5::read(g2, "ds7", a7_view1, std::make_tuple(range(0, size, 2)));
    auto a7_view2 = a7(range(1, size, 2));
    h5::read(g2, "ds7", a7_view2, std::make_tuple(range(1, size, 2)));
    EXPECT_EQ_ARRAY(a7, a);
  }
}

// check 3d arrays and views of scalar and generic types
template <nda::MemoryArray A>
void check_3d_arrays_and_views(const A &a, h5::group group) {
  auto shape = a.shape();

  // write/read full array
  h5::write(group, "a", a);
  A a2;
  h5::read(group, "a", a2);
  EXPECT_EQ_ARRAY(a, a2);

  // write/read full view
  h5::write(group, "a(ellipsis{})", a(ellipsis{}));
  A a3;
  h5::read(group, "a(ellipsis{})", a3);
  EXPECT_EQ_ARRAY(a, a3);

  // write/read various 2d views
  A a4(shape);
  for (int i = 0; i < shape[0]; ++i) {
    auto name = std::string("a(") + std::to_string(i) + ", ellipsis{})";
    h5::write(group, name, a(i, ellipsis{}));
    auto a4_view = a4(i, ellipsis{});
    h5::read(group, name, a4_view);
  }
  EXPECT_EQ_ARRAY(a, a4);

  A a5(shape);
  for (int i = 0; i < shape[1]; ++i) {
    auto name = std::string("a(range::all, ") + std::to_string(i) + ", range::all)";
    h5::write(group, name, a(range::all, i, range::all));
    auto a5_view = a5(range::all, i, range::all);
    h5::read(group, name, a5_view);
  }
  EXPECT_EQ_ARRAY(a, a5);

  A a6(shape);
  for (int i = 0; i < shape[2]; ++i) {
    auto name = std::string("a(ellipsis{}, ") + std::to_string(i) + ")";
    h5::write(group, name, a(ellipsis{}, i));
    auto a6_view = a6(ellipsis{}, i);
    h5::read(group, name, a6_view);
  }
  EXPECT_EQ_ARRAY(a, a6);

  // write/read 1d, 2d and 3d views
  h5::write(group, "a(0, range::all, 0)", a(0, range::all, 0));
  h5::write(group, "a(range(1, shape[0]), range::all, 0)", a(range(1, shape[0]), range::all, 0));
  h5::write(group, "a(0, range::all, range(1, shape[2]))", a(0, range::all, range(1, shape[2])));
  h5::write(group, "a(range(1, shape[0]), range::all, range(1, shape[2]))", a(range(1, shape[0]), range::all, range(1, shape[2])));
  A a7(shape);
  auto a7_view1 = a7(0, range::all, 0);
  h5::read(group, "a(0, range::all, 0)", a7_view1);
  auto a7_view2 = a7(range(1, shape[0]), range::all, 0);
  h5::read(group, "a(range(1, shape[0]), range::all, 0)", a7_view2);
  auto a7_view3 = a7(0, range::all, range(1, shape[2]));
  h5::read(group, "a(0, range::all, range(1, shape[2]))", a7_view3);
  auto a7_view4 = a7(range(1, shape[0]), range::all, range(1, shape[2]));
  h5::read(group, "a(range(1, shape[0]), range::all, range(1, shape[2]))", a7_view4);
  EXPECT_EQ_ARRAY(a, a7);

  // write/read a scalar view
  auto half_0 = shape[0] / 2;
  auto half_1 = shape[1] / 2;
  auto half_2 = shape[2] / 2;
  h5::write(group, "a(half_0, half_1, half_2)", a(half_0, half_1, half_2));
  typename A::value_type scalar;
  h5::read(group, "a(half_0, half_1, half_2)", scalar);
  EXPECT_EQ(a(half_0, half_1, half_2), scalar);

  // write/read slices
  if constexpr (nda::is_scalar_v<typename A::value_type>) {
    auto g      = h5::group(group);
    auto g2     = g.create_group("slices");
    int ds_rank = 3;
    h5::v_t h5_shape(4, 2);
    for (int i = 0; i < 3; ++i) h5_shape[i] = shape[i];
    constexpr auto is_complex = nda::is_complex_v<typename A::value_type>;
    if constexpr (is_complex) ds_rank = 4;

    // write/read full slice
    auto sp8 = H5Screate_simple(ds_rank, h5_shape.data(), nullptr);
    auto ds8 = g2.create_dataset("ds8", h5::hdf5_type<nda::get_value_t<A>>(), sp8);
    if (is_complex) h5::write_attribute(ds8, "__complex__", "1");
    h5::write(g2, "ds8", a, std::make_tuple(ellipsis{}));
    A a8;
    h5::read(g2, "ds8", a8, std::make_tuple(ellipsis{}));
    EXPECT_EQ_ARRAY(a, a8);

    // write/read 1d, 2d and 3d slices
    auto sp9 = H5Screate_simple(ds_rank, h5_shape.data(), nullptr);
    auto ds9 = g2.create_dataset("ds9", h5::hdf5_type<nda::get_value_t<A>>(), sp9);
    if (is_complex) h5::write_attribute(ds9, "__complex__", "1");
    h5::write(g2, "ds9", a(0, range::all, 0), std::make_tuple(0, range::all, 0));
    h5::write(g2, "ds9", a(range(1, shape[0]), range::all, 0), std::make_tuple(range(1, shape[0]), range::all, 0));
    h5::write(g2, "ds9", a(0, range::all, range(1, shape[2])), std::make_tuple(0, range::all, range(1, shape[2])));
    h5::write(g2, "ds9", a(range(1, shape[0]), range::all, range(1, shape[2])), std::make_tuple(range(1, shape[0]), range::all, range(1, shape[2])));
    A a9(shape);
    auto a9_view1 = a9(0, range::all, 0);
    h5::read(g2, "ds9", a9_view1, std::make_tuple(0, range::all, 0));
    auto a9_view2 = a9(range(1, shape[0]), range::all, 0);
    h5::read(g2, "ds9", a9_view2, std::make_tuple(range(1, shape[0]), range::all, 0));
    auto a9_view3 = a9(0, range::all, range(1, shape[2]));
    h5::read(g2, "ds9", a9_view3, std::make_tuple(0, range::all, range(1, shape[2])));
    auto a9_view4 = a9(range(1, shape[0]), range::all, range(1, shape[2]));
    h5::read(g2, "ds9", a9_view4, std::make_tuple(range(1, shape[0]), range::all, range(1, shape[2])));
    EXPECT_EQ_ARRAY(a, a9);
  }
}

TEST(NDA, H5StringArray) {
  // write/read arrays/views of strings
  h5::file file("string.h5", 'w');

  // 1d array
  nda::array<std::string, 1> a_1d(6);
  for (int i = 0; i < 6; ++i) a_1d(i) = "string " + std::to_string(i);
  check_1d_arrays_and_views(a_1d, file);

  // 2d array
  nda::array<std::string, 2> a_2d(2, 2);
  int i = 0;
  for (auto &str : a_2d) str = "string " + std::to_string(i++);

  // write/read every row
  h5::write(file, "a_2d(0, _)", a_2d(0, nda::range::all));
  h5::write(file, "a_2d(1, _)", a_2d(1, nda::range::all));
  nda::array<std::string, 2> b_2d(2, 2);
  auto b_2d_view1 = b_2d(0, nda::range::all);
  h5::read(file, "a_2d(0, _)", b_2d_view1);
  EXPECT_EQ_ARRAY(a_2d(0, nda::range::all), b_2d(0, nda::range::all));
  auto b_2d_view2 = b_2d(1, nda::range::all);
  h5::read(file, "a_2d(1, _)", b_2d_view2);
  EXPECT_EQ_ARRAY(a_2d, b_2d);

  // write/read every column
  h5::write(file, "a_2d(_, 0)", a_2d(nda::range::all, 0));
  h5::write(file, "a_2d(_, 1)", a_2d(nda::range::all, 1));
  nda::array<std::string, 2> c_2d(2, 2);
  auto c_2d_view1 = c_2d(nda::range::all, 0);
  h5::read(file, "a_2d(_, 0)", c_2d_view1);
  EXPECT_EQ_ARRAY(a_2d(nda::range::all, 0), c_2d(nda::range::all, 0));
  auto c_2d_view2 = c_2d(nda::range::all, 1);
  h5::read(file, "a_2d(_, 1)", c_2d_view2);
  EXPECT_EQ_ARRAY(a_2d, c_2d);
}

TEST(NDA, H5GenericArray) {
  // write/read arrays/views of a generic type
  h5::file file("generic_type.h5", 'w');
  h5::group root(file);

  // 1d array
  auto g_1d = root.create_group("1d");
  nda::array<foo, 1> a_1d(6);
  int i_1d = 0;
  for (auto &v : a_1d) v = foo(i_1d++);
  check_1d_arrays_and_views(a_1d, g_1d);

  // 3d array
  auto g_3d = root.create_group("3d");
  nda::array<foo, 3> a_3d(3, 4, 5);
  int i_3d = 0;
  for (auto &v : a_3d) v = foo(i_3d++);
  check_3d_arrays_and_views(a_3d, g_3d);

  // 3d array in Fortran layout
  auto g_3d_f                              = root.create_group("3d_f");
  nda::array<foo, 3, nda::F_layout> a_3d_f = a_3d;
  check_3d_arrays_and_views(a_3d_f, g_3d_f);
}

TEST(NDA, H5IntegerArray) {
  // write/read arrays/views of integers
  h5::file file("integer.h5", 'w');
  h5::group root(file);

  // 1d array
  auto g_1d = root.create_group("1d");
  nda::array<int, 1> a_1d(6);
  int i_1d = 0;
  for (auto &v : a_1d) v = i_1d++;
  check_1d_arrays_and_views(a_1d, g_1d);

  // 3d array
  auto g_3d = root.create_group("3d");
  nda::array<int, 3> a_3d(3, 4, 5);
  int i_3d = 0;
  for (auto &v : a_3d) v = i_3d++;
  check_3d_arrays_and_views(a_3d, g_3d);

  // 3d array Fortran layout
  auto g_3d_f                              = root.create_group("3d_f");
  nda::array<int, 3, nda::F_layout> a_3d_f = a_3d;
  check_3d_arrays_and_views(a_3d_f, g_3d_f);
}

TEST(NDA, H5ComplexArray) {
  // write/read arrays/views of complex numbers
  h5::file file("complex.h5", 'w');
  h5::group root(file);

  // 1d array
  auto g_1d = root.create_group("1d");
  nda::array<std::complex<double>, 1> a_1d(6);
  int r_1d = 0;
  int i_1d = 0;
  for (auto &v : a_1d) v = r_1d++ + i_1d++ * 1.0i;
  check_1d_arrays_and_views(a_1d, g_1d);

  // 3d array
  auto g_3d = root.create_group("3d");
  nda::array<std::complex<double>, 3> a_3d(3, 4, 5);
  int r_3d = 0;
  int i_3d = 0;
  for (auto &v : a_3d) v = r_3d++ + i_3d++ * 1.0i;
  check_3d_arrays_and_views(a_3d, g_3d);

  // 3d array Fortran layout
  auto g_3d_f                                               = root.create_group("3d_f");
  nda::array<std::complex<double>, 3, nda::F_layout> a_3d_f = a_3d;
  check_3d_arrays_and_views(a_3d_f, g_3d_f);
}

TEST(NDA, H5EmptyArray) {
  // write/read empty arrays
  h5::file file("empty.h5", 'w');

  // uninitialized array
  nda::array<long, 2> a(0, 10);
  h5::write(file, "a", a);
  nda::array<long, 2> a_in(5, 5);
  h5::read(file, "a", a_in);
  EXPECT_EQ(a_in.shape(), a.shape());

  // empty (default constructed) array
  nda::array<long, 2> b{};
  h5::write(file, "b", b);
  nda::array<long, 2> b_in(5, 5);
  h5::read(file, "b", b_in);
  EXPECT_EQ(b_in.shape(), b.shape());
}

TEST(NDA, H5DoubleIntoComplexArray) {
  // test reading double values into a complex array
  nda::array<double, 2> a_d(2, 3);
  a_d(i_, j_) << 10 * i_ + j_;

  // write to file
  h5::file file("double_into_complex.h5", 'w');
  h5::write(file, "a_d", a_d);

  // read into complex array
  nda::array<std::complex<double>, 2> a_c(2, 3);
  h5::read(file, "a_d", a_c);
  EXPECT_ARRAY_NEAR(a_c, a_d);
}

TEST(NDA, H5BlockMatrix) {
  // write/read an array of matrices
  using mat_t = nda::matrix<double>;
  nda::array<mat_t, 1> block_mat{mat_t{{1, 2}, {3, 4}}, mat_t{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}}, block_mat_in;

  // write to file
  h5::file file("block_matrix.h5", 'w');
  h5::write(file, "block_mat", block_mat);

  // read from file
  h5::read(file, "block_mat", block_mat_in);

  EXPECT_EQ(block_mat.extent(0), block_mat_in.extent(0));
  for (int i = 0; i < block_mat.extent(0); ++i) EXPECT_ARRAY_EQ(block_mat(i), block_mat_in(i));
}

TEST(NDA, H5ConstIssue) {
  // write a const array
  auto a                              = nda::zeros<double>(2, 2);
  nda::array<double, 2> const a_const = a;
  h5::file file("const_issue.h5", 'w');
  h5::write(file, "a_const", a_const());
}

TEST(NDA, H5SystematicViewsOf3dArray) {
  // write/read various views of a 3d array
  h5::file file("systematic_slices", 'w');
  int n1 = 3, n2 = 5, n3 = 8;
  int max_step = 3;
  nda::array<long, 3> a(n1, n2, n3), a_in;
  a(i_, j_, k_) << (i_ + 10 * j_ + 100 * k_);

  // write to file
  int count = 0;
  for (int i = 0; i < n1; ++i)
    for (int j = 0; j < n2; ++j)
      for (int k = 0; k < n3; ++k)
        for (int i2 = i + 1; i2 < n1; ++i2)
          for (int j2 = j + 1; j2 < n2; ++j2)
            for (int k2 = k + 1; k2 < n3; ++k2)
              for (int si = 1; si <= max_step; ++si)
                for (int sj = 1; sj <= max_step; ++sj)
                  for (int sk = 1; sk <= max_step; ++sk) {
                    h5::write(file, "slice" + std::to_string(count++), a(range(i, i2, si), range(j, j2, sj), range(k, k2, sk)));
                  }

  // read from file and compare
  count = 0;
  for (int i = 0; i < n1; ++i)
    for (int j = 0; j < n2; ++j)
      for (int k = 0; k < n3; ++k)
        for (int i2 = i + 1; i2 < n1; ++i2)
          for (int j2 = j + 1; j2 < n2; ++j2)
            for (int k2 = k + 1; k2 < n3; ++k2)
              for (int si = 1; si <= max_step; ++si)
                for (int sj = 1; sj <= max_step; ++sj)
                  for (int sk = 1; sk <= max_step; ++sk) {
                    h5::read(file, "slice" + std::to_string(count++), a_in);
                    EXPECT_EQ_ARRAY(a_in, a(range(i, i2, si), range(j, j2, sj), range(k, k2, sk)));
                  }
}
