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

#include "./common.hpp"

double foo(double x) { return x / 2; }
int foo(int x) { return x / 2; }

double bar(double x, double y) { return x + y; }

namespace nda::clef {
  using ::bar;
  using ::foo;

  template <typename T>
  T inc(T const &x) requires(!clef::is_any_lazy<T>) {
    return x + 1;
  }

  CLEF_MAKE_FNT_LAZY(bar)
  CLEF_MAKE_FNT_LAZY(inc)
  CLEF_MAKE_FNT_LAZY(foo)
} // namespace nda::clef

#define _TEST_3(EXPR) EXPECT_EQ(eval(EXPR, x_ = 2), [&](int x_) { return EXPR; }(2));

TEST(clef, mathfunction) {

  //_TEST_3(cos(x_));
  //_TEST_3(cos(2 * x_ + 1));
  //_TEST_3(abs(2 * x_ - 1));
  _TEST_3(foo(2 * x_ + 1));
  _TEST_3(foo(2 * x_ + 1));
  _TEST_3(inc(2 * x_ + 1));

  EXPECT_EQ(eval(bar(2 * x_ + 1, x_ - 1), x_ = 2), 6);
}
