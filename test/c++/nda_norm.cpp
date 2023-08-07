// Copyright (c) 2021 Simons Foundation
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
// Authors: Nils Wentzell

#include "./test_common.hpp"
#include <nda/linalg.hpp>

// ==============================================================

TEST(Norm, Zeros) { //NOLINT
  const int N = 100;
  auto v      = nda::zeros<double>(N);

  EXPECT_EQ(norm(v), norm(v, 2.0));
  EXPECT_EQ(norm(v, 0.0), 0.0);
  EXPECT_EQ(norm(v, 1.0), 0.0);
  EXPECT_EQ(norm(v, 2.0), 0.0);
  EXPECT_EQ(norm(v, std::numeric_limits<double>::infinity()), 0.0);
  EXPECT_EQ(norm(v, -std::numeric_limits<double>::infinity()), 0.0);
  EXPECT_EQ(norm(v, 1.5), 0.0);
}

// ==============================================================

TEST(Norm, Ones) { //NOLINT
  const int N = 100;
  auto v      = nda::ones<double>(N);

  EXPECT_EQ(norm(v), norm(v, 2.0));
  EXPECT_EQ(norm(v, 0.0), N);
  EXPECT_EQ(norm(v, 1.0), N);
  EXPECT_EQ(norm(v, 2.0), sqrt(N));
  EXPECT_EQ(norm(v, std::numeric_limits<double>::infinity()), 1);
  EXPECT_EQ(norm(v, -std::numeric_limits<double>::infinity()), 1);
  EXPECT_EQ(norm(v, 1.5), std::pow(double(N), 1.0 / 1.5));
}

// ==============================================================

bool check_norm_p(auto &v, double p) { return norm(v, p) == std::pow(sum(pow(abs(v), p)), 1.0 / p); };

TEST(Norm, Rand) { //NOLINT
  const int N = 100;
  auto v      = nda::rand<double>(N);

  EXPECT_EQ(norm(v), norm(v, 2.0));
  EXPECT_EQ(norm(v, 0.0), N);
  EXPECT_EQ(norm(v, 1.0), sum(v));
  EXPECT_EQ(norm(v, 2.0), sqrt(std::real(nda::blas::dotc(v, v))));
  EXPECT_EQ(norm(v, std::numeric_limits<double>::infinity()), max_element(v));
  EXPECT_EQ(norm(v, -std::numeric_limits<double>::infinity()), min_element(v));

  EXPECT_TRUE((check_norm_p(v, -1.5)));
  EXPECT_TRUE((check_norm_p(v, -1.0)));
  EXPECT_TRUE((check_norm_p(v, 1.5)));
}

// ==============================================================

TEST(Norm, Example) { //NOLINT
  auto vdbl = nda::array<double, 1>{-0.5, 0.0, 1.0, 2.5};

  auto run_checks = [](auto const &v) {
    EXPECT_EQ(norm(v), norm(v, 2.0));
    EXPECT_EQ(norm(v, 0.0), 3);
    EXPECT_EQ(norm(v, 1.0), 4);
    EXPECT_NEAR(norm(v, 2.0), sqrt(7.5), 1e-15);

    EXPECT_TRUE((check_norm_p(v, -1.5)));
    EXPECT_TRUE((check_norm_p(v, -1.0)));
    EXPECT_TRUE((check_norm_p(v, 1.5)));
  };
  run_checks(vdbl);
  run_checks(1i * vdbl);
  run_checks((1 + 1i) / sqrt(2) * vdbl);

  EXPECT_EQ(norm(vdbl, std::numeric_limits<double>::infinity()), 2.5);
  EXPECT_EQ(norm(vdbl, -std::numeric_limits<double>::infinity()), 0.0);
}
