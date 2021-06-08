// Copyright (c) 2019 Simons Foundation
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

#include "./common.hpp"

struct F7 {
  double v;
  F7(double v_) : v(v_) {}
  double operator()(int i1, int, int, int, int, int, int) const { return 10 * i1; }

  CLEF_IMPLEMENT_LAZY_CALL(F7)

  template <typename RHS, typename Tag, typename PhList>
  friend void clef_auto_assign(F7 &x, RHS && rhs, Tag, PhList phl) {
    auto f = nda::clef::make_function(std::forward<RHS>(rhs), phl);
    x.v++;
    std::cerr << " called clef_auto_assign " << f(1, 2, 3, 4, 5, 6, 7) << std::endl;
  }
  friend std::ostream &operator<<(std::ostream &out, F7 const &) { return out << "F7"; }
};

clef::placeholder<1> x1_;
clef::placeholder<2> x2_;
clef::placeholder<3> x3_;
clef::placeholder<4> x4_;
clef::placeholder<5> x5_;
clef::placeholder<6> x6_;
clef::placeholder<7> x7_;
clef::placeholder<8> x8_;

TEST(Clef, F7) {

  F7 f(7), g(8), h(7);

  auto str = to_string(eval(f(x1_, x2_, x3_, x4_, x5_, x6_, x7_), x_ = 1, y_ = 2));
  EXPECT_EQ(str, "lambda(1, 2, _3, _4, _5, _6, _7)");

  // Check compilation speed...
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + x2_ + x3_ + x4_ + x5_ + x6_ + x7_;
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + 2 * x2_ + x3_ + x4_ + x5_ + x6_ + x7_;
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + 2 * x2_ + 4 * x3_ + x4_ + x5_ + x6_ + x7_
        + g(x1_, x2_, x3_, x4_, x5_, x6_, x7_) * h(x1_, x2_, x3_, x4_, x5_, x6_, x7_);
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + 2 * x2_ + 4 * x3_ + 8 * x4_ + x5_ + x6_ + x7_
        + g(x1_, x2_, x3_, x4_, x5_, x6_, x7_) * h(x1_, x2_, x3_, x4_, x5_, x6_, x7_);

  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + x2_ - x3_ + x4_ + x5_ + x6_ + x7_;
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + 2 * x2_ + x3_ + x4_ - x5_ + x6_ + x7_;
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + 2 * x2_ + 4 * x3_ + x4_ - x5_ + x6_ + x7_
        + g(x1_, x2_, x3_, x4_, x5_, x6_, x7_) * h(x1_, x2_, x3_, x4_, x5_, x6_, x7_);
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << g(x1_, x2_, x3_, x4_, x5_, x6_, x7_) / h(x1_, x2_, x3_, x4_, x5_, x6_, x7_)
        + g(x1_, x2_, x3_, x4_, x5_, x6_, x7_) * h(x1_, x2_, x3_, x4_, x5_, x6_, x7_);

  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + x2_ - x3_ + x4_ + x5_ + x6_ - x7_;
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + 2 * x2_ + x3_ + x4_ - x5_ + x6_ - x7_;
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << x1_ + 2 * x2_ + 4 * x3_ + x4_ - x5_ + x6_ + x7_
        - g(x1_, x2_, x3_, x4_, x5_, x6_, x7_) * h(x1_, x2_, x3_, x4_, x5_, x6_, x7_);
  f(x1_, x2_, x3_, x4_, x5_, x6_, x7_) << g(x1_, x2_, x3_, x4_, x5_, x6_, x7_) / h(x1_, x2_, x3_, x4_, x5_, x6_, x7_)
        - g(x1_, x2_, x3_, x4_, x5_, x6_, x7_) * h(x1_, x2_, x3_, x4_, x5_, x6_, x7_);
}
