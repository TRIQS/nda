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

#include <gtest/gtest.h> // NOLINT
#include <cmath>
#include <limits>
#include <iostream>

#ifndef NDA_DEBUG
#define NDA_DEBUG
#define NDA_ENFORCE_BOUNDCHECK
#endif

#include <nda/nda.hpp>
#include <nda/gtest_tools.hpp>

using namespace std::complex_literals;

namespace clef= nda::clef;

// for test only. Backward compat
namespace nda { 
  template <int Rank>
  using shape_t = std::array<long, Rank>;
}

// variables for the test
auto _ = nda::range::all;
nda::ellipsis ___;

using nda::range;
using dcomplex = std::complex<double>;
using nda::C_layout;
using nda::F_layout;
using nda::matrix;
using nda::matrix_view;

#define MAKE_MAIN_MPI                                                                                                                                \
  int main(int argc, char **argv) {                                                                                                                  \
    ::mpi::environment env(argc, argv);                                                                                                              \
    ::testing::InitGoogleTest(&argc, argv);                                                                                                          \
    return RUN_ALL_TESTS();                                                                                                                          \
  }

#define MAKE_MAIN                                                                                                                                    \
  int main(int argc, char **argv) {                                                                                                                  \
    ::testing::InitGoogleTest(&argc, argv);                                                                                                          \
    return RUN_ALL_TESTS();                                                                                                                          \
  }

#define EXPECT_PRINT(X, Y)                                                                                                                           \
  {                                                                                                                                                  \
    std::stringstream ss;                                                                                                                            \
    ss << Y;                                                                                                                                         \
    EXPECT_EQ(X, ss.str());                                                                                                                          \
  }
