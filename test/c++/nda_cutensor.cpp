// Copyright (c) 2019-2021 Simons Foundation
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

#include <algorithm>
#include <type_traits>
#include "test_common.hpp"
#include "nda/gtest_tools.hpp"

#include <nda/nda.hpp>
#include <nda/tensor.hpp>
#include <nda/traits.hpp>
//#include <nda/clef/literals.hpp>

using nda::C_layout;
using nda::F_layout;
using dcomplex = std::complex<double>;
//using namespace clef::literals;

//----------------------------

template <typename value_t, typename Layout>
void test_contract() {

  nda::range::all_t _ = {};
  // MAM: add tests passing matrix view's
  using other_layout = std::conditional_t<std::is_same_v<Layout, C_layout>, F_layout, C_layout>;
  { // ik,kj->ij
    nda::matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M2{{1, 1}, {1, 1}}, M3{{1, 0}, {0, 1}};
    nda::cumatrix<value_t, Layout> M1_d{M1}, M2_d{M2}, M3_d{M3};

    nda::tensor::contract(1.0, M1_d, "ik", M2_d, "kj", 1.0, M3_d, "ij");

    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});
  }
  {
    nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
    nda::array<value_t, 3, Layout> M2{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}};
    nda::array<value_t, 3, Layout> M3(2, 2, 2);
    nda::array<value_t, 2, Layout> M4(2, 2);
    nda::array<value_t, 1, Layout> M5(2);
    M3() = 0;
    M4() = 0;
    M5() = 0;
    nda::cuarray<value_t, 3, Layout> M1_d{M1}, M2_d{M2}, M3_d{M3};
    nda::cuarray<value_t, 2, Layout> M4_d{M4};
    nda::cuarray<value_t, 1, Layout> M5_d{M5};

    // ijk,ikl->ijl
    nda::tensor::contract(1.0, M1_d, "ijk", M2_d, "ikl", 0.0, M3_d, "ijl");
    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{4, 6}, {12, 22}}, {{92, 110}, {132, 158}}});

    // ikj,kli->lij
    nda::tensor::contract(1.0, M1_d, "ikj", M2_d, "kli", 0.0, M3_d, "lij");
    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{16, 24}, {68, 80}}, {{24, 40}, {108, 128}}});

    // ijk,klj->li
    nda::tensor::contract(1.0, M1_d, "ijk", M2_d, "klj", 0.0, M4_d, "li");
    M4 = M4_d;
    EXPECT_ARRAY_NEAR(M4, nda::array<value_t, 2>{{42, 122}, {66, 210}});

    // ijk,kij->i
    nda::tensor::contract(1.0, M1_d, "ijk", M2_d, "kij", 0.0, M5_d, "i");
    M5 = M5_d;
    EXPECT_ARRAY_NEAR(M5, nda::array<value_t, 1>{42, 210});

    // ik,jk->ij
    nda::tensor::contract(1.0, M1_d(_, 0, _), "ik", M2_d(0, _, _), "jk", 0.0, M3_d(0, _, _), "ij");
    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3(0, _, _), nda::array<value_t, 2>{{2, 6}, {10, 46}});
  }

  // mixed layouts
  { // ik,kj->ij
    nda::matrix<value_t, Layout> M1{{0, 1}, {1, 2}}, M3{{1, 0}, {0, 1}};
    nda::matrix<value_t, other_layout> M2{{1, 1}, {1, 1}};
    nda::cumatrix<value_t, Layout> M1_d{M1}, M3_d{M3};
    nda::cumatrix<value_t, other_layout> M2_d{M2};
    nda::tensor::contract(1.0, M1_d, "ik", M2_d, "kj", 1.0, M3_d, "ij");

    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{2, 1}, {3, 4}});
  }

  {
    nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
    nda::array<value_t, 3, other_layout> M2{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}};
    nda::array<value_t, 3, Layout> M3(2, 2, 2);
    nda::array<value_t, 2, Layout> M4(2, 2);
    nda::array<value_t, 1, Layout> M5(2);
    M3() = 0;
    M4() = 0;
    M5() = 0;
    nda::cuarray<value_t, 3, Layout> M1_d{M1}, M3_d{M3};
    nda::cuarray<value_t, 3, other_layout> M2_d{M2};
    nda::cuarray<value_t, 2, Layout> M4_d{M4};
    nda::cuarray<value_t, 1, Layout> M5_d{M5};

    // ijk,ikl->ijl
    nda::tensor::contract(1.0, M1_d, "ijk", M2_d, "ikl", 0.0, M3_d, "ijl");
    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{4, 6}, {12, 22}}, {{92, 110}, {132, 158}}});

    // ikj,kli->lij
    nda::tensor::contract(1.0, M1_d, "ikj", M2_d, "kli", 0.0, M3_d, "lij");
    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{16, 24}, {68, 80}}, {{24, 40}, {108, 128}}});

    // ijk,klj->li
    nda::tensor::contract(1.0, M1_d, "ijk", M2_d, "klj", 0.0, M4_d, "li");
    M4 = M4_d;
    EXPECT_ARRAY_NEAR(M4, nda::array<value_t, 2>{{42, 122}, {66, 210}});

    // ijk,kij->i
    nda::tensor::contract(1.0, M1_d, "ijk", M2_d, "kij", 0.0, M5_d, "i");
    M5 = M5_d;
    EXPECT_ARRAY_NEAR(M5, nda::array<value_t, 1>{42, 210});
  }
}

TEST(TENSOR, contract) { test_contract<double, C_layout>(); }     //NOLINT
TEST(TENSOR, contractF) { test_contract<double, F_layout>(); }    //NOLINT
TEST(TENSOR, zcontract) { test_contract<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zcontractF) { test_contract<dcomplex, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_outer_product_contract() {

  using other_layout = std::conditional_t<std::is_same_v<Layout, C_layout>, F_layout, C_layout>;
  { // i,j->ij
    nda::matrix<value_t, Layout> M3{{1, 0}, {0, 1}};
    nda::array<value_t, 1, Layout> M1{{value_t{1}, value_t{2}}}, M2{{value_t{3}, value_t{4}}};

    nda::cumatrix<value_t, Layout> M3_d{M3};
    nda::cuarray<value_t, 1, Layout> M1_d{M1}, M2_d{M2};

    nda::tensor::contract(1.0, M1_d, "i", M2_d, "j", 1.0, M3_d, "ij");

    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{4, 4}, {6, 9}});
  }

  { // i,ij->ij
    nda::matrix<value_t, Layout> M2{{1, 2}, {3, 4}}, M3{{1, 0}, {0, 1}};
    nda::array<value_t, 1, Layout> M1{{value_t{2}, value_t{3}}};

    nda::cumatrix<value_t, Layout> M2_d{M2}, M3_d{M3};
    nda::cuarray<value_t, 1, Layout> M1_d{M1};

    nda::tensor::contract(1.0, M1_d, "i", M2_d, "ij", 1.0, M3_d, "ij");

    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{3, 4}, {9, 13}});
  }

  { // i,j->ij
    nda::matrix<value_t, other_layout> M3{{1, 0}, {0, 1}};
    nda::array<value_t, 1, Layout> M1{{value_t{1}, value_t{2}}}, M2{{value_t{3}, value_t{4}}};

    nda::cumatrix<value_t, other_layout> M3_d{M3};
    nda::cuarray<value_t, 1, Layout> M1_d{M1}, M2_d{M2};

    nda::tensor::contract(1.0, M1_d, "i", M2_d, "j", 1.0, M3_d, "ij");

    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{4, 4}, {6, 9}});
  }

  { // i,ij->ij
    nda::matrix<value_t, other_layout> M2{{1, 2}, {3, 4}};
    nda::matrix<value_t, Layout> M3{{1, 0}, {0, 1}};
    nda::array<value_t, 1, Layout> M1{{value_t{2}, value_t{3}}};

    nda::cumatrix<value_t, other_layout> M2_d{M2};
    nda::cumatrix<value_t, Layout> M3_d{M3};
    nda::cuarray<value_t, 1, Layout> M1_d{M1};

    nda::tensor::contract(1.0, M1_d, "i", M2_d, "ij", 1.0, M3_d, "ij");

    M3 = M3_d;
    EXPECT_ARRAY_NEAR(M3, nda::matrix<value_t>{{3, 4}, {9, 13}});
  }
}

TEST(TENSOR, outer_product_contract) { test_outer_product_contract<double, C_layout>(); }                 //NOLINT
TEST(TENSOR, outer_product_contractF) { test_outer_product_contract<double, F_layout>(); }                //NOLINT
TEST(TENSOR, zouter_product_contract) { test_outer_product_contract<std::complex<double>, C_layout>(); }  //NOLINT
TEST(TENSOR, zouter_product_contractF) { test_outer_product_contract<std::complex<double>, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_add() {
  nda::range::all_t _ = {};
  nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
  nda::array<value_t, 3, Layout> M2{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}};
  nda::array<value_t, 3, Layout> M3(2, 2, 2);
  M3() = 0;
  nda::cuarray<value_t, 3, Layout> M1_d{M1}, M2_d{M2}, M3_d{M3};

  nda::tensor::add(2.0, M1_d, "ijk", 1.0, M2_d, "ijk");
  M2 = M2_d;
  EXPECT_ARRAY_NEAR(M2, nda::array<value_t, 3>{{{0, 4}, {8, 12}}, {{16, 20}, {24, 28}}});

  nda::tensor::add(0.0, M1_d, "ijk", 3.0, M2_d(_, _, _), "ijk");
  M2 = M2_d;
  EXPECT_ARRAY_NEAR(M2, nda::array<value_t, 3>{{{0, 12}, {24, 36}}, {{48, 60}, {72, 84}}});

  nda::tensor::add(2.0, M1_d, "ijk", 0.0, M2_d, "ijk");
  M2 = M2_d;
  EXPECT_ARRAY_NEAR(M2, nda::array<value_t, 3>{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}});

  nda::tensor::add(5.0, M1_d, "kij", 7.0, M2_d, "ijk");
  M2 = M2_d;
  EXPECT_ARRAY_NEAR(M2, nda::array<value_t, 3>{{{0, 34}, {33, 67}}, {{66, 100}, {99, 133}}});

  nda::tensor::add(2.0, M1_d, "ijk", 0.0, M2_d, "ijk"); // to reset to original M2
  nda::tensor::add(5.0, M1_d, "ijk", 7.0, M2_d, "ijk", M3_d, "ijk");
  M3 = M3_d;
  EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{0, 19}, {38, 57}}, {{76, 95}, {114, 133}}});

  nda::tensor::add(5.0, M1_d, "kij", 7.0, M2_d, "ijk", M3_d, "ijk");
  M3 = M3_d;
  EXPECT_ARRAY_NEAR(M3, nda::array<value_t, 3>{{{0, 34}, {33, 67}}, {{66, 100}, {99, 133}}});

  nda::tensor::add(5.0, M1_d(_, 0, _), "ji", 7.0, M2_d(0, _, _), "ij", M3_d(0, _, _), "ij");
  M3 = M3_d;
  EXPECT_ARRAY_NEAR(M3(0, _, _), nda::array<value_t, 2>{{0, 34}, {33, 67}});

  // out of place transposition through add
  nda::tensor::add(1.0, M1_d, "kij", 0.0, M2_d, "ijk");
  M2 = M2_d;
  EXPECT_ARRAY_NEAR(M2, nda::array<value_t, 3>{{{0, 4}, {1, 5}}, {{2, 6}, {3, 7}}});
}

TEST(TENSOR, add) { test_add<double, C_layout>(); }     //NOLINT
TEST(TENSOR, addF) { test_add<double, F_layout>(); }    //NOLINT
TEST(TENSOR, zadd) { test_add<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zaddF) { test_add<dcomplex, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_elementwise() {
  using nda::tensor::op::MUL;
  using nda::tensor::op::SUM;
  nda::range::all_t _ = {};
  nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
  nda::array<value_t, 3, Layout> M2{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}};
  nda::cuarray<value_t, 3, Layout> M1_d{M1}, M2_d{M2};

  nda::tensor::elementwise(M1_d, "ijk", M2_d, "ijk", SUM);
  EXPECT_ARRAY_NEAR(to_host(M2_d), M1);

  nda::tensor::elementwise(1.0, M1_d, 1.0, M2_d, SUM);
  EXPECT_ARRAY_NEAR(to_host(M2_d), M2);

  nda::tensor::elementwise(2.0, M1_d, 1.0, M2_d, SUM);
  EXPECT_ARRAY_NEAR(to_host(M2_d), nda::array<value_t, 3>{{{0, 4}, {8, 12}}, {{16, 20}, {24, 28}}});

  nda::tensor::elementwise(0.0, M1_d, 3.0, M2_d(_, _, _), SUM);
  EXPECT_ARRAY_NEAR(to_host(M2_d), nda::array<value_t, 3>{{{0, 12}, {24, 36}}, {{48, 60}, {72, 84}}});

  nda::tensor::elementwise(2.0, M1_d, "ijk", 0.0, M2_d, "ijk", SUM);
  EXPECT_ARRAY_NEAR(to_host(M2_d), nda::array<value_t, 3>{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}});

  nda::tensor::elementwise(5.0, M1_d, "kij", 7.0, M2_d, "ijk", SUM);
  EXPECT_ARRAY_NEAR(to_host(M2_d), nda::array<value_t, 3>{{{0, 34}, {33, 67}}, {{66, 100}, {99, 133}}});

  nda::tensor::elementwise(1.0, M1_d, 1.0, M2_d, SUM);
  EXPECT_ARRAY_NEAR(to_host(M2_d), nda::array<value_t, 3>{{{0, 35}, {35, 70}}, {{70, 105}, {105, 140}}});

  nda::tensor::elementwise(2.0, M1_d, "ijk", 0.0, M2_d, "ijk", SUM); // to reset to original M2
  EXPECT_ARRAY_NEAR(to_host(M2_d), M2);

  nda::tensor::elementwise(2.0, M1_d(0, _, _), "jk", 0.0, M2_d, "ijk", SUM);
  EXPECT_ARRAY_NEAR(to_host(M2_d), nda::array<value_t, 3>{{{0, 2}, {4, 6}}, {{0, 2}, {4, 6}}});

  nda::tensor::elementwise(2.0, M1_d, 1.0, M2_d, MUL);
  EXPECT_ARRAY_NEAR(to_host(M2_d), nda::array<value_t, 3>{{{0, 4}, {16, 36}}, {{0, 20}, {48, 84}}});
}

TEST(TENSOR, elementwise) { test_elementwise<double, C_layout>(); }     //NOLINT
TEST(TENSOR, elementwiseF) { test_elementwise<double, F_layout>(); }    //NOLINT
TEST(TENSOR, zelementwise) { test_elementwise<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zelementwiseF) { test_elementwise<dcomplex, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_elementwise_trinary() {
  using nda::tensor::op::MUL;
  using nda::tensor::op::SUM;
  nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
  nda::array<value_t, 2, Layout> M2{{{value_t(0), value_t(3)}, {value_t(6), value_t(9)}}};
  nda::array<value_t, 3, Layout> M3{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}};

  {
    nda::cuarray<value_t, 2, Layout> M2_d{M2};
    nda::cuarray<value_t, 3, Layout> M1_d{M1}, M3_d{M3};
    nda::cuarray<value_t, 3, Layout> T1{M1}, T2{M3}; 

    nda::tensor::elementwise_trinary(1.0, M1_d, "ijk", 1.0, M2_d, "jk", 1.0, M3_d, "ijk", SUM, SUM);
    nda::tensor::elementwise(1.0, M2_d, "jk", 1.0, T1, "ijk", SUM); 
    nda::tensor::elementwise(1.0, T1, "ijk", 1.0, T2, "ijk", SUM); 
    EXPECT_ARRAY_NEAR(to_host(T2), to_host(M3_d));
  }

  { 
    nda::cuarray<value_t, 2, Layout> M2_d{M2};
    nda::cuarray<value_t, 3, Layout> M1_d{M1}, M3_d{M3};
    nda::cuarray<value_t, 3, Layout> T1{M1}, T2{M3}; 

    nda::tensor::elementwise_trinary(1.0, M1_d, "jik", 1.0, M2_d, "jk", 1.0, M3_d, "kij", SUM, SUM);  
    nda::tensor::elementwise(1.0, M2_d, "jk", 1.0, T1, "jik", SUM);
    nda::tensor::elementwise(1.0, T1, "jik", 1.0, T2, "kij", SUM);
    EXPECT_ARRAY_NEAR(to_host(T2), to_host(M3_d));
  }

}

TEST(TENSOR, elementwise_trinary) { test_elementwise_trinary<double, C_layout>(); }     //NOLINT
TEST(TENSOR, elementwiseF_trinary) { test_elementwise_trinary<double, F_layout>(); }    //NOLINT
TEST(TENSOR, zelementwise_trinary) { test_elementwise_trinary<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zelementwiseF_trinary) { test_elementwise_trinary<dcomplex, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_assign() {
  nda::range::all_t _ = {};
  {
    nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
    nda::cuarray<value_t, 3, Layout> M1_d{M1};
    nda::cuarray<value_t, 3, Layout> M2_d{M1};
    M2_d() = 0.0;

    nda::tensor::assign(M1_d, M2_d);
    EXPECT_ARRAY_NEAR(nda::to_host(M2_d), M1);

    nda::tensor::assign(2, M1_d, M2_d);
    EXPECT_ARRAY_NEAR(nda::to_host(M2_d), nda::array<value_t, 3>{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}});

    EXPECT_ARRAY_NEAR(nda::to_host(M2_d), nda::array<value_t, 3>{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}});

    nda::tensor::assign(M1_d, "ikj", M2_d, "ijk");
    EXPECT_ARRAY_NEAR(nda::to_host(M2_d), nda::array<value_t, 3>{{{0, 2}, {1, 3}}, {{4, 6}, {5, 7}}});
  }
  { // some complicated case...
    using rg = nda::range;
    nda::array<value_t, 5, Layout> M1(4, 5, 4, 4, 7);
    M1() = 0;
    nda::cuarray<value_t, 5, Layout> M1_d{M1};
    nda::cuarray<value_t, 5, Layout> M2_d{M1};
    nda::tensor::set(2, M1_d(rg(0, 4, 2), rg(0, 4, 2), _, rg(0, 3), rg(0, 5, 3)));
    nda::tensor::assign(2.5, M1_d, "kjlim", M2_d, "ijklm");
    M1      = M1_d;
    auto M2 = nda::to_host(M2_d);
    for (int i = 0; i < M2.extent(0); ++i)
      for (int j = 0; j < M2.extent(1); ++j)
        for (int k = 0; k < M2.extent(2); ++k)
          for (int l = 0; l < M2.extent(3); ++l)
            for (int m = 0; m < M2.extent(4); ++m) EXPECT_EQ(M2(i, j, k, l, m), 2.5 * M1(k, j, l, i, m));
  }
}

TEST(TENSOR, assign) { test_assign<double, C_layout>(); }     //NOLINT
TEST(TENSOR, assignF) { test_assign<double, F_layout>(); }    //NOLINT
TEST(TENSOR, zassign) { test_assign<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zassignF) { test_assign<dcomplex, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_set() {
  nda::range::all_t _ = {};
  {
    nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
    nda::cuarray<value_t, 3, Layout> M1_d{M1};
    nda::tensor::set(2, M1_d);
    M1 = M1_d;
    EXPECT_ARRAY_NEAR(M1, nda::array<value_t, 3>{{{2, 2}, {2, 2}}, {{2, 2}, {2, 2}}});
  }
  { // some complicated case...
    using rg = nda::range;
    nda::array<value_t, 5, Layout> M1(4, 5, 4, 4, 7);
    M1() = 0;
    nda::cuarray<value_t, 5, Layout> M1_d{M1};
    nda::tensor::set(2, M1_d(rg(0, 4, 2), rg(0, 4, 2), _, rg(0, 3), rg(0, 5, 3)));
    M1 = M1_d;
    EXPECT_EQ(192, std::accumulate(M1.data(), M1.data() + M1.size(), int(0), [](auto const &a, auto &v) { return a + int(std::abs(v)); }));
    EXPECT_EQ(96, std::count_if(M1.data(), M1.data() + M1.size(), [](auto &v) { return std::abs(v) > 1.0e-6; }));
  }
}

TEST(TENSOR, set) { test_set<double, C_layout>(); }     //NOLINT
TEST(TENSOR, setF) { test_set<double, F_layout>(); }    //NOLINT
TEST(TENSOR, zset) { test_set<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zsetF) { test_set<dcomplex, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_scale() {
  nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
  nda::cuarray<value_t, 3, Layout> M1_d{M1};
  nda::tensor::scale(2, M1_d);
  M1 = M1_d;
  EXPECT_ARRAY_NEAR(M1, nda::array<value_t, 3>{{{0, 2}, {4, 6}}, {{8, 10}, {12, 14}}});
}

TEST(TENSOR, scale) { test_scale<double, C_layout>(); }     //NOLINT
TEST(TENSOR, scaleF) { test_scale<double, F_layout>(); }    //NOLINT
TEST(TENSOR, zscale) { test_scale<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zscaleF) { test_scale<dcomplex, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_dot() {
  nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
  nda::cuarray<value_t, 3, Layout> M1_d{M1};
  EXPECT_NEAR(std::abs(nda::tensor::dot(M1_d, "ijk", M1_d, "ijk")), double{140}, 1.e-12);
  EXPECT_NEAR(std::abs(nda::tensor::dot(M1_d, "ijk", M1_d, "jik")), double{132}, 1.e-12);
  EXPECT_NEAR(std::abs(nda::tensor::dot(M1_d, "ikj", M1_d, "kji")), double{126}, 1.e-12);
}

TEST(TENSOR, dot) { test_dot<double, C_layout>(); }     //NOLINT
TEST(TENSOR, dotF) { test_dot<double, F_layout>(); }    //NOLINT
TEST(TENSOR, dotz) { test_dot<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zdotF) { test_dot<dcomplex, F_layout>(); } //NOLINT

template <typename value_t, typename Layout>
void test_reduce() {
  nda::array<value_t, 3, Layout> M1{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
  nda::cuarray<value_t, 3, Layout> M1_d{M1};
  EXPECT_NEAR(std::abs(nda::tensor::reduce(M1_d, nda::tensor::op::SUM)), double{28}, 1.e-12);
  if constexpr (not nda::is_complex_v<value_t>) {
    EXPECT_NEAR(std::abs(nda::tensor::reduce(M1_d, nda::tensor::op::MAX)), double{7}, 1.e-12);
    EXPECT_NEAR(std::abs(nda::tensor::reduce(M1_d, nda::tensor::op::MIN)), double{0}, 1.e-12);
  }
}

TEST(TENSOR, reduce) { test_reduce<double, C_layout>(); }     //NOLINT
TEST(TENSOR, reduceF) { test_reduce<double, F_layout>(); }    //NOLINT
TEST(TENSOR, reducez) { test_reduce<dcomplex, C_layout>(); }  //NOLINT
TEST(TENSOR, zreduceF) { test_reduce<dcomplex, F_layout>(); } //NOLINT
