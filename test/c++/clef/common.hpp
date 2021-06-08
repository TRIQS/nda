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

#include <nda/clef/clef.hpp>
#include <nda/clef/io.hpp>
#include <sstream>
#include <string>

#define EXPECT_PRINT(X, Y)                                                                                                                           \
  {                                                                                                                                                  \
    std::stringstream ss;                                                                                                                            \
    ss << Y;                                                                                                                                         \
    EXPECT_EQ(X, ss.str());                                                                                                                          \
  }

using namespace std::complex_literals;

namespace clef= nda::clef;

template <typename T>
std::string to_string(T const &x) {
  std::stringstream fs;
  fs << x;
  return fs.str();
}

struct F1 {
  int v = 0;
  F1(int v_) : v(v_) {}
  F1(F1 const &) = delete; // non copyable
  F1(F1 &&x) : v(x.v) { std::cerr << "Moving F1 " << v << std::endl; }

  int operator()(int x) const { return 10 * x; }

  CLEF_IMPLEMENT_LAZY_CALL(F1)

  template < typename RHS, typename Tag, typename PhList>
  friend void clef_auto_assign(F1 &x, RHS && rhs, Tag, PhList phl) {
    auto f = clef::make_function(std::forward<RHS>(rhs), phl);
    x.v = f(x.v);
  }

  friend std::ostream &operator<<(std::ostream &out, F1 const &) { return out << "F1"; }
};

struct F2 {

  double v;
  F2() { v = 0; }

  double operator()(double x, double y) const { return 10 * x + y; }

  CLEF_IMPLEMENT_LAZY_CALL(F2)

  template < typename RHS, typename Tag, typename PhList>
  friend void clef_auto_assign(F2 const &, RHS && rhs, Tag, PhList phl) {
    auto f = clef::make_function(std::forward<RHS>(rhs), phl);
    std::cerr << " called F2 clef_auto_assign " << f(10, 20) << std::endl;
  }

  friend std::ostream &operator<<(std::ostream &out, F2 const &) { return out << "F2"; }
};

using namespace clef;

clef::placeholder<1> x_;
clef::placeholder<2> y_;
clef::placeholder<3> z_;
