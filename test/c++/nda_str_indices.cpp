// Copyright (c) 2020 Simons Foundation
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
// Authors: Olivier Parcollet

#include "./test_common.hpp"

using str_t = std::string;

// ==============================================================

TEST(Construct, NoString) { //NOLINT
  nda::array<long, 2, nda::C_layout_str> a(3, 2);

  nda::array<nda::array<std::string, 1>, 1> ind = a.indexmap().get_string_indices();

  EXPECT_EQ(ind.shape(), (std::array<long, 1>{2}));

  for (int i = 0; i < a.rank; ++i)
    for (int j = 0; j < a.shape()[i]; ++j) EXPECT_EQ(ind(i)(j), std::to_string(j));
}
// ==============================================================

TEST(Construct, FromStringArray) { //NOLINT

  nda::array<nda::array<std::string, 1>, 1> ind{{"a", "b", "c"}, {"A", "B"}};

  nda::array<long, 2, nda::C_layout_str> b(ind);
  EXPECT_EQ(b.shape(), (std::array<long, 2>{3, 2}));
  EXPECT_EQ(ind, b.indexmap().get_string_indices()());
}
// ==============================================================

TEST(Construct, FromStringArray2) { //NOLINT

  nda::array<nda::array<std::string, 1>, 1> ind{{"a", "b", "c"}, {"A", "B"}};

  using lay_t = nda::array<long, 2, nda::C_layout_str>::layout_t;
  nda::array<long, 2, nda::C_layout_str> b(lay_t{ind});
  EXPECT_EQ(b.shape(), (std::array<long, 2>{3, 2}));
  EXPECT_EQ(ind, b.indexmap().get_string_indices()());
}

// ==============================================================
// a test with one array with string

class StrIndicesA : public ::testing::Test {
  public:
  nda::array<long, 2, nda::C_layout_str> a;

  protected:
  virtual void SetUp() override {

    nda::array<nda::array<std::string, 1>, 1> ind{{"a", "b", "c"}, {"A", "B"}};

    using lay_t = nda::array<long, 2, nda::C_layout_str>::layout_t;

    lay_t lay{ind};
    a = nda::array<long, 2, nda::C_layout_str>(lay);

    for (int i = 0; i < a.shape()[0]; ++i)
      for (int j = 0; j < a.shape()[1]; ++j) a(i, j) = i + 10 * j;
  }
  void TearDown() override {}
};

// ==============================================================

TEST_F(StrIndicesA, CrossConstruct) { //NOLINT

  nda::array<long, 2> AA(a);

  EXPECT_EQ(a.shape(), AA.shape());

  for (int i = 0; i < a.shape()[0]; ++i)
    for (int j = 0; j < a.shape()[1]; ++j) EXPECT_EQ(AA(i, j), a(i, j));
}

// ==============================================================

TEST_F(StrIndicesA, CrossConstructView) { //NOLINT

  nda::array_view<long, 2> v(a);

  EXPECT_EQ(a.shape(), v.shape());

  for (int i = 0; i < a.shape()[0]; ++i)
    for (int j = 0; j < a.shape()[1]; ++j) EXPECT_EQ(v(i, j), a(i, j));
}

// ==============================================================

TEST_F(StrIndicesA, call) { //NOLINT

  auto ind = a.indexmap().get_string_indices();

  EXPECT_EQ(a(str_t{"a"}, str_t{"A"}), a(0, 0));

  for (int i = 0; i < a.shape()[0]; ++i)
    for (int j = 0; j < a.shape()[1]; ++j) EXPECT_EQ(a(ind[0][i], ind[1][j]), a(i, j));

  EXPECT_EQ(a("a", "A"), a(0, 0));

  EXPECT_THROW(a("z", "A"), nda::runtime_error); //NOLINT
}

// ==============================================================

TEST_F(StrIndicesA, Equal) { //NOLINT

  EXPECT_TRUE(a == a);
  auto b = a;
  EXPECT_TRUE(a == b);

  b(0, 0) += 3;
  EXPECT_FALSE(a == b);
}

// ==============================================================

TEST_F(StrIndicesA, sliceall) { //NOLINT
  auto v = a(_, "A");
  nda::array_view<long, 2> view_no_str(a);
  auto vcheck = view_no_str(_, 0);
  nda::array<nda::array<std::string, 1>, 1> ind{{"a", "b", "c"}};

  EXPECT_EQ_ARRAY(v, vcheck);
  EXPECT_EQ(v.indexmap().get_string_indices(), ind);
}

// ==============================================================

TEST_F(StrIndicesA, slice) { //NOLINT
  auto v = a(nda::range(0, 2), "A");
  nda::array_view<long, 2> view_no_str(a);
  auto vcheck = view_no_str(nda::range(0, 2), 0);
  nda::array<nda::array<std::string, 1>, 1> ind{{"a", "b"}};

  EXPECT_EQ_ARRAY(v, vcheck);
  EXPECT_EQ(v.indexmap().get_string_indices(), ind);
}

// ==============================================================

TEST_F(StrIndicesA, transpose) { //NOLINT

  auto a2 = nda::transposed_view<1, 0>(a);

  nda::array<nda::array<std::string, 1>, 1> ind2{{"A", "B"}, {"a", "b", "c"}};
  EXPECT_EQ(a2.indexmap().get_string_indices(), ind2);

  auto v1 = nda::transposed_view<1, 0>(nda::array_view<long, 2>{a});
  auto v2 = nda::array_view<long, 2, F_layout>{a2};

  EXPECT_EQ_ARRAY(v1, v2);
}
