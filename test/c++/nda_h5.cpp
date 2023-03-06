// Copyright (c) 2019-2020 Simons Foundation
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
#include <h5/h5.hpp>
#include <nda/h5.hpp>
#include <nda/clef/literals.hpp>

using namespace nda::clef::literals;

template <typename T>
void one_test(std::string name, T scalar) {

  nda::array<T, 1> a(5), a_check;
  nda::array<T, 2> b(2, 3), b_check;
  nda::array<T, 3> c(2, 3, 4), c_check;

  a(i_) << scalar * (10 * i_);
  b(i_, j_) << scalar * (10 * i_ + j_);
  c(i_, j_, k_) << scalar * (i_ + 10 * j_ + 100 * k_);

  std::string filename = "ess_" + name + ".h5";
  // WRITE the file
  {
    h5::file file(filename, 'w');
    h5::group top(file);
    auto G1 = top.create_group("G");

    h5_write(top, "A", a);
    h5_write(top, "B", b);
    h5_write(top, "C", c);
    h5_write(top, "scalar", scalar);

    // add some attribute to A
    auto id = top.open_dataset("A");
    h5_write_attribute(id, "AttrOfA1", 12);
    h5_write_attribute(id, "AttrOfA2", 8.9);

    // in a subgroup
    auto G = top.open_group("G");
    h5_write(G, "A2", a);
  }

  // READ the file
  {
    h5::file file(filename, 'r');
    h5::group top(file);

    h5_read(top, "A", a_check);
    EXPECT_EQ_ARRAY(a, a_check);

    h5_read(top, "B", b_check);
    EXPECT_EQ_ARRAY(b, b_check);

    h5_read(top, "C", c_check);
    EXPECT_EQ_ARRAY(c, c_check);

    EXPECT_EQ(scalar, h5::h5_read<T>(top, "scalar"));

    auto d = a;
    d()    = 0;
    h5_read(top, "G/A2", d);
    EXPECT_EQ_ARRAY(a, d);

    // read the attributes of A
    auto id   = top.open_dataset("A");
    int att1  = h5::h5_read_attribute<int>(id, "AttrOfA1");
    auto att2 = h5::h5_read_attribute<double>(id, "AttrOfA2");
    EXPECT_EQ(att1, 12);
    EXPECT_EQ(att2, 8.9);
  }
}

//------------------------------------
TEST(Basic, Int) { //NOLINT
  one_test<int>("int", 1);
}

TEST(Basic, Long) { //NOLINT
  one_test<long>("long", 1);
}

TEST(Basic, Double) { //NOLINT
  one_test<double>("double", 1.5);
}

TEST(Basic, Dcomplex) { //NOLINT
  one_test<dcomplex>("dcomplex", (1.0 + 1.0i));
}

//------------------------------------

TEST(Basic, GroupAttribute) { //NOLINT

  {
    h5::file file("ess_group_attr.h5", 'w');
    auto g = h5::group(file).create_group("mygroup");
    h5_write_attribute(g, "Scheme", "MYTAGTYPE");
  }
  {
    h5::file file("ess_group_attr.h5", 'r');
    h5::group top(file);
    auto g = top.open_group("mygroup");
    std::string s;
    h5::h5_read_attribute(g, "Scheme", s);
    EXPECT_EQ(s, std::string{"MYTAGTYPE"});
  }
}
//------------------------------------

TEST(Basic, Empty) { //NOLINT

  nda::array<long, 2> a(0, 10);
  {
    h5::file file("ess_empty.h5", 'w');
    //h5::group top(file);
    h5_write(file, "empty", a);
  }
  {
    h5::file file("ess_empty.h5", 'r');
    h5::group top(file);
    nda::array<long, 2> empty(5, 5);
    h5_read(top, "empty", empty);
    EXPECT_EQ(empty.shape(), (nda::shape_t<2>{0, 10}));
  }

  nda::array<long, 2> b{};
  {
    h5::file file("ess_default.h5", 'w');
    h5_write(file, "default", b);
  }
  {
    h5::file file("ess_default.h5", 'r');
    h5::group top(file);
    nda::array<long, 2> empty(5, 5);
    h5_read(top, "default", empty);
    EXPECT_EQ(empty.shape(), (nda::shape_t<2>{0, 0}));
  }
}

//------------------------------------

TEST(Basic, String) { //NOLINT

  {
    h5::file file("ess_string.h5", 'w');
    h5_write(file, "s", std::string("a nice chain"));
    h5_write(file, "sempty", "");
  }
  {
    h5::file file("ess_string.h5", 'r');
    nda::array<double, 2> empty(5, 5);

    std::string s2("----------------------------------");
    h5_read(file, "s", s2);
    EXPECT_EQ(s2, "a nice chain");

    std::string s3; //empty
    h5_read(file, "s", s3);
    EXPECT_EQ(s3, "a nice chain");

    std::string s4; //empty
    h5_read(file, "sempty", s4);
    EXPECT_EQ(s4, "");
  }
}

//------------------------------------

TEST(Array, H5) { //NOLINT

  nda::array<long, 2> A(2, 3), A2;
  nda::array<bool, 2> B(2, 3), B2;
  nda::array<double, 2> D(2, 3), D2;
  nda::array<dcomplex, 1> C(5), C2;
  //dcomplex z(1, 2);

  for (int i = 0; i < 5; ++i) C(i) = dcomplex(i, i);

  A(i_, j_) << 10 * i_ + j_;
  B(i_, j_) << (i_ < j_);
  D(i_, j_) << A(i_, j_) / 10.0;

  // WRITE the file
  {
    h5::file file("ess_gal.h5", 'w');
    h5::group top(file);

    h5_write(top, "A", A);
    h5_write(top, "B", B);
    h5_write(top, "D", D);
    h5_write(top, "C", C);
    h5::h5_write(top, "S", "");
    h5_write(top, "A_slice", A(nda::range::all, nda::range(1, 3)));
    h5_write(top, "empty", nda::array<double, 2>(0, 10));

    // add some attribute to A
    auto id = top.open_dataset("A");
    h5_write_attribute(id, "Attr1OfA", 12);
    h5_write_attribute(id, "Attr2OfA", 8.9);

    // scalar
    double x = 2.3;
    h5_write(top, "x", x);

    // dcomplex xx(2, 3);
    // h5_write(top, "xx", xx);

    h5_write(top, "s", std::string("a nice chain"));

    top.create_group("G");
    h5_write(top, "G/A", A);

    auto G = top.open_group("G");
    h5_write(G, "A2", A);
  }

  // READ the file
  {
    h5::file file("ess_gal.h5", 'r');
    h5::group top(file);

    h5_read(top, "A", A2);
    EXPECT_EQ_ARRAY(A, A2);

    h5_read(top, "B", B2);
    EXPECT_EQ_ARRAY(B, B2);

    // read the attributes of A
    auto id   = top.open_dataset("A");
    int att1  = h5::h5_read_attribute<int>(id, "Attr1OfA");
    auto att2 = h5::h5_read_attribute<double>(id, "Attr2OfA");

    EXPECT_EQ(att1, 12);
    EXPECT_EQ(att2, 8.9);

    h5_read(top, "D", D2);
    EXPECT_ARRAY_NEAR(D, D2);

    h5_read(top, "C", C2);
    EXPECT_ARRAY_NEAR(C, C2);

    nda::array<long, 2> a_sli;
    h5_read(top, "A_slice", a_sli);
    EXPECT_EQ_ARRAY(a_sli, A(nda::range::all, nda::range(1, 3)));

    double xxx = 0;
    h5_read(top, "x", xxx);
    EXPECT_DOUBLE_EQ(xxx, 2.3);

    std::string s2("----------------------------------");
    h5_read(top, "s", s2);
    EXPECT_EQ(s2, "a nice chain");
  }
}

// ==============================================================

TEST(Array, H5ArrLayout) { //NOLINT

  nda::array<long, 2, F_layout> Af(2, 3);
  Af(i_, j_) << 10 * i_ + j_;

  // write to file
  {
    h5::file f("test_nda_layout.h5", 'w');
    h5_write(f, "Af", Af);
  }

  // reread
  nda::array<long, 2, F_layout> Bf{};
  {
    h5::file f("test_nda_layout.h5", 'r');
    h5_read(f, "Af", Bf);
  }

  EXPECT_EQ(Af, Bf);
}

// ==============================================================

TEST(Vector, String) { //NOLINT

  // vector of string
  std::vector<std::string> V1, V2;
  V1.emplace_back("abcd");
  V1.emplace_back("de");

  // writing
  h5::file file("test_nda::array_string.h5", 'w');
  h5::group top(file);

  h5_write(top, "V", V1);

  // rereading
  h5_read(top, "V", V2);

  //comparing
  for (int i = 0; i < 2; ++i) { EXPECT_EQ(V1[i], V2[i]); }
}

// ==============================================================

TEST(Array, H5ArrayString) { //NOLINT

  // nda::array of string
  nda::array<std::string, 1> A(2), B;
  A(0) = "Nice String";
  A(1) = "Unicode €☺";

  // writing
  h5::file file("test_nda::array_string.h5", 'w');
  h5::group top(file);

  h5_write(top, "A", A);

  // rereading
  h5_read(top, "A", B);

  //comparing
  for (int i = 0; i < 2; ++i) { EXPECT_EQ(A(i), B(i)); }
}

// ==============================================================

// -----------------------------------------------------
// Testing the loading of nda::array of double into complex
// -----------------------------------------------------
TEST(Array, H5RealIntoComplex) { //NOLINT

  nda::array<double, 2> D(2, 3);
  D(i_, j_) << 10 * i_ + j_;

  // WRITE the file
  {
    h5::file file("ess_real_complex.h5", 'w');
    h5::group top(file);
    h5_write(top, "D", D);
  }

  nda::array<dcomplex, 2> C(2, 3);

  // READ the file
  {
    C() = 89.0 + 9i; // put garbage in it
    h5::file file("ess_real_complex.h5", 'r');
    h5::group top(file);
    h5_read(top, "D", C);
    EXPECT_ARRAY_NEAR(C, D);
  }
}

// ==============================================================

// -----------------------------------------------------
// Testing h5 for an nda::array of matrix
// -----------------------------------------------------

TEST(BlockMatrixH5, S1) { //NOLINT

  using mat_t = nda::array<double, 2>;
  nda::array<mat_t, 1> W, V{mat_t{{1, 2}, {3, 4}}, mat_t{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}};

  {
    h5::file file1("ess_non_pod.h5", 'w');
    h5_write(file1, "block_mat", V);
  }

  {
    h5::file file2("ess_non_pod.h5", 'r');
    h5_read(file2, "block_mat", W);
  }

  EXPECT_EQ(V.extent(0), W.extent(0));
  for (int i = 0; i < V.extent(0); ++i) EXPECT_ARRAY_NEAR(V(i), W(i));
}

// ==============================================================

TEST(Array, ConstIssue) { //NOLINT
  auto a = nda::zeros<double>(2, 2);

  nda::array<double, 2> const a_c = a;
  {
    h5::file file1("const_issue.h5", 'w');
    h5::write(file1, "a_c", a_c());
  }
}

// not yet implemented
// ==============================================================
/*
TEST(Array, H5ArrayString2) { //NOLINT


  // nda::array of string
  nda::array<std::string, 2> A(2, 2), B;
  A(0, 0) = "Nice String";
  A(1, 0) = "another";
  A(1, 1) = "really";
  A(0, 1) = "nice";

  // writing
  h5::file file("test_nda::array_string.h5", 'w');
  h5::group top(file);

  h5_write(top, "A", A);

  // rereading
  h5_read(top, "A", B);

  //comparing
  for (int i = 0; i < 2; ++i) { EXPECT_EQ(A, B); }
}
*/
// ==============================================================
/*

   // DECIDE if we want to implement such promotion int -> double in h5 reading
TEST(Array, Promotion) { //NOLINT


  nda::array<long, 2> a(2, 3);
  nda::array<double, 2> d;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      a(i, j) = 10 * i + j;
    }

  // WRITE the file
  {
    h5::file file("ess_prom.h5", 'w');
    //h5::group top(file);
    h5_write(file, "A", a);
  }

  // READ the file
  {
    h5::file file("ess_prom.h5", 'r');
    h5::group top(file);

    h5_read(top, "A", d);
    EXPECT_ARRAY_NEAR(a, d);
  }
}

TEST(Array, PromotionWrong1) { //NOLINT


  nda::array<long, 2> a(2, 3);
  nda::array<int, 2> d;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      a(i, j) = 10 * i + j;
    }

  // WRITE the file
  {
    h5::file file("ess_prom1.h5", 'w');
    //h5::group top(file);
    h5_write(file, "A", a);
  }

  // READ the file
  {
    h5::file file("ess_prom1.h5", 'r');
    h5::group top(file);

    h5_read(top, "A", d);
    EXPECT_ARRAY_NEAR(a, d);
  }
}


TEST(Array, PromotionWrong) { //NOLINT


  nda::array<double, 2> a(2, 3);
  nda::array<long, 2> d;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      a(i, j) = 2.8* i + j;
    }

  // WRITE the file
  {
    h5::file file("ess_prom2.h5", 'w');
    //h5::group top(file);
    h5_write(file, "A", a);
  }

  // READ the file
  {
    h5::file file("ess_prom2.h5", 'r');
    h5::group top(file);

    h5_read(top, "A", d);
    EXPECT_ARRAY_NEAR(a, d);
  }
}
*/
// ==============================================================
