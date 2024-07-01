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

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <array>
#include <string>

// Test fixture for testing arrays with string indices.
class NDAStringIndices : public ::testing::Test {
  protected:
  using lay_t = nda::array<long, 2, nda::C_layout_str>::layout_t;
  nda::array<long, 2, nda::C_layout_str> A;
  nda::array<nda::array<std::string, 1>, 1> str_idxs;

  void SetUp() override {
    str_idxs = nda::array<nda::array<std::string, 1>, 1>{{"a", "b", "c"}, {"A", "B"}};
    lay_t lay{str_idxs};
    A = nda::array<long, 2, nda::C_layout_str>(lay);
    for (int i = 0; i < A.shape()[0]; ++i)
      for (int j = 0; j < A.shape()[1]; ++j) A(i, j) = i + 10 * j;
  }
};

TEST_F(NDAStringIndices, ConstructArrayWithoutStringIndices) {
  nda::array<long, 2, nda::C_layout_str> B(3, 2);
  nda::array<nda::array<std::string, 1>, 1> idxs = B.indexmap().get_string_indices();
  EXPECT_EQ(idxs.shape(), (std::array<long, 1>{2}));
  for (int i = 0; i < B.rank; ++i)
    for (int j = 0; j < B.shape()[i]; ++j) EXPECT_EQ(idxs(i)(j), std::to_string(j));
}

TEST_F(NDAStringIndices, ConstructArrayWithStringIndices) {
  EXPECT_EQ(A.shape(), (std::array<long, 2>{3, 2}));
  EXPECT_EQ(str_idxs, A.indexmap().get_string_indices()());
}

TEST_F(NDAStringIndices, ConstructArrayWithStringIndexLayout) {
  nda::array<long, 2, nda::C_layout_str> B(lay_t{str_idxs});
  EXPECT_EQ(B.shape(), (std::array<long, 2>{3, 2}));
  EXPECT_EQ(str_idxs, B.indexmap().get_string_indices()());
}

TEST_F(NDAStringIndices, CrossConstructArray) {
  nda::array<long, 2> B(A);
  EXPECT_EQ(A.shape(), B.shape());
  for (int i = 0; i < A.shape()[0]; ++i)
    for (int j = 0; j < A.shape()[1]; ++j) EXPECT_EQ(B(i, j), A(i, j));
}

TEST_F(NDAStringIndices, CrossConstructView) {
  nda::array_view<long, 2> B_v(A);
  EXPECT_EQ(A.shape(), B_v.shape());
  for (int i = 0; i < A.shape()[0]; ++i)
    for (int j = 0; j < A.shape()[1]; ++j) EXPECT_EQ(B_v(i, j), A(i, j));
}

TEST_F(NDAStringIndices, AccessArray) {
  EXPECT_EQ(A(std::string{"a"}, std::string{"A"}), A(0, 0));

  for (int i = 0; i < A.shape()[0]; ++i)
    for (int j = 0; j < A.shape()[1]; ++j) EXPECT_EQ(A(str_idxs[0][i], str_idxs[1][j]), A(i, j));

  EXPECT_EQ(A("a", "A"), A(0, 0));

  EXPECT_THROW(A("z", "A"), nda::runtime_error);
}

TEST_F(NDAStringIndices, EqualToOperator) {
  EXPECT_TRUE(A == A);
  auto B = A;
  EXPECT_TRUE(A == B);
  B(0, 0) += 3;
  EXPECT_FALSE(A == B);
}

TEST_F(NDAStringIndices, SliceRangeAll) {
  auto slice = A(nda::range::all, "A");
  nda::array_view<long, 2> A_v(A);
  auto exp_v = A_v(nda::range::all, 0);
  nda::array<nda::array<std::string, 1>, 1> idxs{{"a", "b", "c"}};
  EXPECT_EQ_ARRAY(slice, exp_v);
  EXPECT_EQ(slice.indexmap().get_string_indices(), idxs);
}

TEST_F(NDAStringIndices, SliceRange) {
  auto slice = A(nda::range(0, 2), "A");
  nda::array_view<long, 2> A_v(A);
  auto exp_v = A_v(nda::range(0, 2), 0);
  nda::array<nda::array<std::string, 1>, 1> idxs{{"a", "b"}};
  EXPECT_EQ_ARRAY(slice, exp_v);
  EXPECT_EQ(slice.indexmap().get_string_indices(), idxs);
}

TEST_F(NDAStringIndices, TransposedView) {
  auto A_t = nda::transposed_view<1, 0>(A);

  nda::array<nda::array<std::string, 1>, 1> idxs_t{{"A", "B"}, {"a", "b", "c"}};
  EXPECT_EQ(A_t.indexmap().get_string_indices(), idxs_t);

  auto A_v1 = nda::transposed_view<1, 0>(nda::array_view<long, 2>{A});
  auto A_v2 = nda::array_view<long, 2, nda::F_layout>{A_t};
  EXPECT_EQ_ARRAY(A_v1, A_v2);
}
