// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2023 Simons Foundation
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

#include "./test_common.hpp"

#include <nda/basic_functions.hpp>
#include <nda/clef.hpp>
#include <nda/stdutil/array.hpp>

#include <array>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace nda;

// Test fixture for CLEF tests.
struct CLEF : ::testing::Test {
  protected:
  clef::placeholder<0> x0_;
  clef::placeholder<1> x1_;
  clef::placeholder<2> x2_;
  clef::placeholder<3> x3_;
  clef::placeholder<4> x4_;
  clef::placeholder<5> x5_;
};

// Custom class to test various CLEF features.
struct foo {
  // various constructors and assignment operators
  foo() { std::cout << "foo()" << std::endl; }
  foo(int x) : x(x) { std::cout << "foo(int)" << std::endl; }
  foo(foo const &) = delete;
  foo(foo &&f) noexcept : x(f.x) { std::cout << "foo(foo &&)" << std::endl; }
  foo &operator=(foo const &) = delete;
  foo &operator=(foo &&f) noexcept {
    x = f.x;
    std::cout << "foo &operator=(foo &&)" << std::endl;
    return *this;
  }

  // print methods
  void print() const & { std::cout << "foo::print() const &" << std::endl; }
  void print() const && { std::cout << "foo::print() const &&" << std::endl; }

  // call operator
  int operator()(int i) const { return x + i; }

  // plus operator
  foo operator+(foo const &f) const { return x + f.x; }

  // lazy method
  [[nodiscard]] int lazy(int i) const { return x * i; }
  CLEF_IMPLEMENT_LAZY_METHOD(foo, lazy)

  int x{100};
};

// Custom class to test auto assignments.
struct bar {
  // function call operator
  int &operator()(int i) { return arr[i]; }
  const int &operator()(int i) const { return arr[i]; }

  // make function call operator lazy
  CLEF_IMPLEMENT_LAZY_CALL(bar)

  // clef_auto_assign overload
  template <typename F>
  friend void clef_auto_assign(bar &b, F &&f) {
    for (int i = 0; i < b.arr.size(); ++i) { b.arr[i] = std::forward<F>(f)(i); }
  }

  // subscript operator
  int &operator[](int i) { return arr[i]; }
  const int &operator[](int i) const { return arr[i]; }

  // clef_auto_assign_subscript overload
  template <typename F>
  friend void clef_auto_assign_subscript(bar &b, F &&f) {
    for (int i = 0; i < b.arr.size(); ++i) { b.arr[i] = std::forward<F>(f)(i); }
  }

  std::array<int, 5> arr{};
};

// Lazy functions.
int lazy_f1(int x) { return x * x; }
double lazy_f1(double x) { return x + x; }
CLEF_MAKE_FNT_LAZY(lazy_f1)

int lazy_f2(int x, int y) { return x * y; }
double lazy_f2(double x, double y) { return x + y; }
CLEF_MAKE_FNT_LAZY(lazy_f2)

TEST_F(CLEF, PlaceholderValuePair) {
  // check correct storage of types
  auto p1 = (x0_ = 10);
  static_assert(std::is_same_v<decltype(p1.rhs), int>);

  auto p2 = (x1_ = std::array<int, 3>{1, 2, 3});
  static_assert(std::is_same_v<decltype(p2.rhs), std::array<int, 3>>);

  auto p3 = (x2_ = x0_ / 1);
  static_assert(std::is_same_v<decltype(p3.rhs), decltype(x0_ / 1)>);

  auto vec = std::vector{1, 2, 3};
  auto p4  = (x0_ = vec);
  static_assert(std::is_same_v<decltype(p4.rhs), std::vector<int> &>);

  // check lazyness
  using namespace clef::literals;
  static_assert(clef::is_lazy<decltype(x0_)>);
  static_assert(clef::is_lazy<decltype(i_)>);
  static_assert(clef::is_lazy<decltype(1 + x0_)>);

  // function call operator
  auto f1  = [](int x) { return x * x; };
  auto ex1 = x0_(10);
  EXPECT_EQ(clef::eval(ex1, x0_ = f1), 100);

  // subscript operator
  auto arr = std::array{1, 2, 3};
  auto ex2 = x0_[1];
  EXPECT_EQ(clef::eval(ex2, x0_ = arr), 2);
}

TEST_F(CLEF, LazyExpressions) {
  auto ex1 = 5 * x0_;
  EXPECT_EQ(clef::eval(ex1, x0_ = 1, x1_ = 2), 5 * 1);
  auto ex1_p0 = clef::eval(ex1, x0_ = x0_ + x1_);
  EXPECT_EQ(clef::eval(ex1_p0, x0_ = 1, x1_ = 2), 5 * (1 + 2));
  EXPECT_EQ(clef::eval(clef::eval(ex1_p0, x0_ = 1), x1_ = 2), 5 * (1 + 2));
  EXPECT_EQ(clef::eval(clef::eval(ex1_p0, x1_ = 2), x0_ = 1), 5 * (1 + 2));

  auto ex2 = x0_ + 2 * x1_;
  EXPECT_EQ(clef::eval(ex2, x0_ = 1, x1_ = 2), 1 + 2 * 2);
  auto ex2_p0 = clef::eval(ex2, x0_ = x0_ + x1_);
  EXPECT_EQ(clef::eval(ex2_p0, x0_ = 1, x1_ = 2), 1 + 2 + 2 * 2);

  auto ex3 = x0_ + x1_ * 2;
  EXPECT_EQ(clef::eval(ex3, x0_ = 1, x1_ = 2), 1 + 2 * 2);
  auto ex3_p1 = clef::eval(ex3, x1_ = x0_ + x1_);
  EXPECT_EQ(clef::eval(ex3_p1, x0_ = 1, x1_ = 2), 1 + (1 + 2) * 2);

  auto ex4 = x0_ + 2 * x1_ + x0_;
  EXPECT_EQ(clef::eval(ex4, x0_ = 1, x1_ = 2), 1 + 2 * 2 + 1);
  auto ex4_p0 = clef::eval(ex4, x0_ = x0_ + x1_);
  EXPECT_EQ(clef::eval(ex4_p0, x0_ = 1, x1_ = 2), 1 + 2 + 2 * 2 + 1 + 2);

  auto ex5 = x0_ / 2.0 + 2 * x1_;
  EXPECT_EQ(clef::eval(ex5, x0_ = 1, x1_ = 2), 1 / 2.0 + 2 * 2);
  auto ex5_p0 = clef::eval(ex5, x0_ = x0_ + x1_);
  EXPECT_DOUBLE_EQ(clef::eval(ex5_p0, x0_ = 1, x1_ = 2), (1 + 2) / 2.0 + 2 * 2);
}

TEST_F(CLEF, LazyFunctionsAndExpressions) {
  // various lazy function calls with a single argument
  EXPECT_EQ(clef::eval(lazy_f1(x0_), x0_ = 3), 9);
  EXPECT_EQ(clef::eval(lazy_f1(x0_), x0_ = 3.0), 6.0);
  EXPECT_EQ(clef::eval(clef::eval(lazy_f1(2 * x0_), x0_ = 3)), 36);
  EXPECT_EQ(clef::eval(clef::eval(lazy_f1(2.0 * x0_), x0_ = 3)), 12);
  EXPECT_EQ(clef::eval(lazy_f1(x0_ + x1_), x0_ = 3, x1_ = 3), 36);
  EXPECT_EQ(clef::eval(clef::eval(lazy_f1(x0_ + x1_), x0_ = 3), x1_ = 3), 36);
  EXPECT_EQ(clef::eval(lazy_f1(x0_ + x1_), x0_ = 3.0, x1_ = 3), 12);
  EXPECT_EQ(clef::eval(clef::eval(lazy_f1(x0_ + x1_), x0_ = 3.0), x1_ = 3), 12);
  EXPECT_EQ(clef::eval(lazy_f1(lazy_f2(x0_, x1_)), x0_ = 4, x1_ = 3), 144);
  EXPECT_EQ(clef::eval(clef::eval(lazy_f1(lazy_f2(x0_ + x2_, x1_)), x2_ = 2), x0_ = 2, x1_ = 3), 144);
  EXPECT_EQ(clef::eval(lazy_f1(x0_) + 2 * x1_, x0_ = 1, x1_ = 2), lazy_f1(1) + 2 * 2);
  EXPECT_EQ(clef::eval(clef::eval(lazy_f1(x0_) + 2 * x1_, x0_ = x0_ + x1_), x0_ = 1, x1_ = 2), lazy_f1(1 + 2) + 2 * 2);
  EXPECT_DOUBLE_EQ(clef::eval(1 / lazy_f1(x0_) + 2 * x1_, x0_ = 1.0, x1_ = 2), 1 / lazy_f1(1.0) + 2 * 2);
  EXPECT_DOUBLE_EQ(clef::eval(clef::eval(1 / lazy_f1(x0_) + 2 * x1_, x0_ = x0_ + x1_), x0_ = 1.0, x1_ = 2), 1 / lazy_f1(1.0 + 2) + 2 * 2);

  // various lazy function calls with multiple arguments
  EXPECT_EQ(clef::eval(lazy_f2(x0_, x1_), x0_ = 2, x1_ = 3), 6);
  EXPECT_EQ(clef::eval(lazy_f2(x0_, 2), x0_ = 3), 6);
  EXPECT_EQ(clef::eval(lazy_f2(x0_, x1_), x0_ = 2.0, x1_ = 3.0), 5.0);
  EXPECT_EQ(clef::eval(lazy_f2(x0_ / x2_, x1_), x0_ = 2.0, x1_ = 3.0, x2_ = 4.0), 3.5);
  EXPECT_EQ(clef::eval(lazy_f2(x0_ / x2_ + x3_, x1_), x3_ = 2, x1_ = 3, x2_ = 4, x0_ = 4), 9);
  EXPECT_EQ(clef::eval(clef::eval(clef::eval(lazy_f2(x0_ / x2_ + x3_, x1_), x3_ = 2), x1_ = 3), x2_ = 4, x0_ = 4), 9);
  EXPECT_EQ(clef::eval(lazy_f2(lazy_f1(x0_), lazy_f1(x0_)), x0_ = 3), 81);
  EXPECT_EQ(clef::eval(lazy_f2(lazy_f1(x0_), lazy_f1(x1_ * 2.0 - x2_)), x0_ = -3.0, x1_ = 1, x2_ = 5), -12);
}

TEST_F(CLEF, TerminalExpressions) {
  auto arr = std::array{1, 2, 3};

  // temporary in terminal expression
  auto ex1 = clef::make_expr(std::array{1, 2, 3});
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(ex1.childs)>, decltype(arr)>);
  EXPECT_EQ(clef::eval(ex1), arr);

  // reference in terminal expression
  auto ex2 = clef::make_expr(arr);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(ex2.childs)>, std::reference_wrapper<decltype(arr)>>);
  EXPECT_EQ(clef::eval(ex2), arr);

  // clone in terminal expression
  auto ex3 = clef::make_expr_from_clone(arr);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(ex3.childs)>, decltype(arr)>);
  EXPECT_EQ(clef::eval(ex3), arr);
}

TEST_F(CLEF, SubscriptExpressions) {
  auto vec       = std::vector{1, 2, 3};
  const auto idx = 1;

  // placeholder subscript operator
  auto ex1 = x0_[idx];
  EXPECT_EQ(clef::eval(ex1, x0_ = vec), vec[idx]);
  EXPECT_EQ(clef::eval(ex1, x0_ = std::vector{1, 2, 3}), vec[idx]);

  // placeholder subscript operator with lazy index
  auto ex2 = x0_[x1_];
  EXPECT_EQ(clef::eval(ex2, x0_ = vec, x1_ = idx), vec[idx]);
  // partial evaluation of subscriptable object
  auto ex2_p0 = clef::eval(ex2, x0_ = vec);
  EXPECT_EQ(clef::eval(ex2_p0, x1_ = idx), vec[idx]);
  // partial evaluation of index
  auto ex2_p1 = clef::eval(ex2, x1_ = idx);
  EXPECT_EQ(clef::eval(ex2_p1, x0_ = vec), vec[idx]);

  // make expression with subscript operator
  auto ex3 = clef::make_expr(vec)[x0_ + x1_];
  EXPECT_EQ(clef::eval(ex3, x0_ = idx + 1, x1_ = -1), vec[idx]);
  // partial evaluations
  auto ex3_p0 = clef::eval(ex3, x0_ = idx + 1);
  EXPECT_EQ(clef::eval(ex3_p0, x1_ = -1), vec[idx]);
  auto ex3_p1 = clef::eval(ex3, x1_ = -1);
  EXPECT_EQ(clef::eval(ex3_p1, x0_ = idx + 1), vec[idx]);

  // make subscript expression
  auto ex4 = clef::make_expr_subscript(vec, x0_ / x1_);
  EXPECT_EQ(clef::eval(ex4, x0_ = idx * 4, x1_ = 4), vec[idx]);
  auto ex4_p0 = clef::eval(ex4, x0_ = idx * 4);
  // partial evaluations
  EXPECT_EQ(clef::eval(ex4_p0, x1_ = 4), vec[idx]);
  auto ex4_p1 = clef::eval(ex4, x1_ = 4);
  EXPECT_EQ(clef::eval(ex4_p1, x0_ = idx * 4), vec[idx]);

  // subscript operator of binary expression (not working and probably should not work anyway)
  // auto arr1 = std::array{1, 2, 3};
  // auto arr2 = std::array{4, 5, 6};
  // auto ex5  = (x0_ + x1_)[idx];
  // EXPECT_EQ(clef::eval(ex5, x0_ = arr1, x1_ = arr2), 7);

  // print to stdout
  std::cout << ex1 << std::endl;
}

TEST_F(CLEF, FunctionCallExpressions) {
  auto sq        = [](auto x) { return x * x; };
  auto add       = [](auto x, auto y) { return x + y; };
  const auto arg = 2;

  // placeholder function call
  auto ex1 = x0_(arg);
  EXPECT_EQ(clef::eval(ex1, x0_ = sq), sq(arg));
  EXPECT_EQ(clef::eval(ex1, x0_ = [](auto x) { return x * x; }), sq(arg));

  // placeholder function call with lazy argument
  auto ex2 = x0_(-x1_);
  EXPECT_EQ(clef::eval(ex2, x0_ = sq, x1_ = arg), sq(arg));
  // partial evaluation of function
  auto ex2_p0 = clef::eval(ex2, x0_ = sq);
  EXPECT_EQ(clef::eval(ex2_p0, x1_ = arg), sq(arg));
  // partial evaluation of argument
  auto ex2_p1 = clef::eval(ex2, x1_ = arg);
  EXPECT_EQ(clef::eval(ex2_p1, x0_ = sq), sq(arg));

  // make nested expression with function call operator and subscript operator
  auto arr = std::array{1, 2, 3};
  auto ex3 = clef::make_expr(sq)(clef::make_expr(arr)[+x0_]);
  EXPECT_EQ(clef::eval(ex3, x0_ = 1), sq(2));

  // make function call expression
  auto ex4 = clef::make_expr_call(add, x0_ * x1_, x2_);
  EXPECT_EQ(clef::eval(ex4, x0_ = 3, x1_ = 2, x2_ = 1), add(3 * 2, 1));
  // partial evaluation
  auto ex4_p0_p2 = clef::eval(ex4, x0_ = 3, x2_ = 1);
  EXPECT_EQ(clef::eval(ex4_p0_p2, x1_ = 2), add(3 * 2, 1));

  // function call operator of binary expression
  auto ex5 = (x0_ + x1_)(x2_);
  EXPECT_EQ(clef::eval(ex5, x0_ = foo(2), x1_ = foo(3), x2_ = 1), 6);
  // partial evaluation
  auto ex5_p0_p2 = clef::eval(ex5, x0_ = foo(2), x2_ = 1);
  EXPECT_EQ(clef::eval(ex5_p0_p2, x1_ = foo(3)), 6);

  // print to stdout
  std::cout << ex2_p1 << std::endl;
}

TEST_F(CLEF, IfElseExpressions) {
  EXPECT_EQ(clef::eval(clef::if_else(true, 2 * x0_, x1_), x0_ = 1, x1_ = 3), 2);
  EXPECT_EQ(clef::eval(clef::if_else(false, 2 * x0_, x1_), x0_ = 1, x1_ = 3), 3);
  EXPECT_EQ(clef::eval(clef::if_else(x0_ > x1_, 2 * x0_, x1_), x0_ = 1, x1_ = 3), 3);

  // print to stdout
  std::cout << clef::if_else(x0_ * x3_ - x4_, x1_, x2_) << std::endl;
}

TEST_F(CLEF, LogicalExpressions) {
  auto ex1 = x0_ < x1_;
  EXPECT_EQ(clef::eval(ex1, x0_ = 1, x1_ = 2), true);

  auto ex2 = x0_ > x1_;
  EXPECT_EQ(clef::eval(ex2, x0_ = 1, x1_ = 2), false);

  auto ex3 = x0_ + x1_ <= x2_;
  EXPECT_EQ(clef::eval(ex3, x0_ = 1, x1_ = 2, x2_ = 3), true);
  // partial evaluation
  auto ex3_p1_p2 = clef::eval(ex3, x1_ = 2, x2_ = 3);
  EXPECT_EQ(clef::eval(ex3_p1_p2, x0_ = 1), true);

  auto ex4 = x0_ >= x1_ - x2_;
  EXPECT_EQ(clef::eval(ex4, x0_ = 1, x1_ = 2, x2_ = 3), true);

  auto ex5 = x0_[x1_] == x2_(x3_);
  EXPECT_EQ(clef::eval(ex5, x0_ = std::vector{1, 2, 3}, x1_ = 1, x2_ = foo(1), x3_ = 1), true);

  auto ex6 = !(x0_[x1_] == x2_(x3_));
  EXPECT_EQ(clef::eval(ex6, x0_ = std::vector{1, 2, 3}, x1_ = 1, x2_ = foo(1), x3_ = 1), false);

  // print to stdout
  std::cout << ex4 << std::endl;
}

TEST_F(CLEF, MakeFunction) {
  auto ex1 = x0_ - x1_;

  // standard case
  auto f1 = clef::make_function(ex1, x0_, x1_);
  EXPECT_EQ(f1(3, 2), 1);
  EXPECT_EQ(clef::make_function(ex1, x0_, x1_)(2, 3), -1);

  // switch arugment positions
  auto f1_sw = clef::make_function(ex1, x1_, x0_);
  EXPECT_EQ(f1_sw(3, 2), -1);

  // partial case with temporary arguments
  auto f1_p0     = clef::make_function(ex1, x0_);
  auto f1_p0_tmp = clef::make_function(f1_p0(3), x1_);
  EXPECT_EQ(f1_p0_tmp(2), 1);
  EXPECT_EQ(clef::eval(clef::make_function(ex1, x0_), x1_ = 2)(3), 1);

  // partial case with reference arguments
  auto x0        = 3;
  auto x1        = 2;
  auto f1_p1     = clef::make_function(ex1, x1_);
  auto f1_p1_tmp = clef::make_function(f1_p1(x1), x0_);
  EXPECT_EQ(f1_p1_tmp(x0), 1);

  // given placeholder not in expression
  auto ex2     = x0_ + 3;
  auto f2_lazy = clef::make_function(ex2, x1_);
  static_assert(clef::is_lazy<decltype(f2_lazy)>);
  auto f2 = clef::make_function(ex2, x0_);
  static_assert(!clef::is_lazy<decltype(f2)>);
  EXPECT_EQ(f2(2), 5);

  // print to stdout
  std::cout << f1 << std::endl;
}

TEST_F(CLEF, Literals) {
  using namespace clef::literals;

  // some arbitrary expressions
  auto ex1 = !(i_[j_] == k_(l_));
  EXPECT_EQ(clef::eval(ex1, i_ = std::vector{1, 2, 3}, j_ = 1, k_ = foo(1), l_ = 1), false);

  // make function from expression
  auto f1 = clef::make_function(i_ + j_, i_, j_);
  EXPECT_EQ(f1(3, 2), 5);
  static_assert(!clef::is_lazy<decltype(f1)>);
}

TEST_F(CLEF, AutoAssign) {
  // auto assign to a vector
  std::vector<int> vec(5);
  clef::make_expr(vec)[x0_] << x0_ * x0_;
  EXPECT_EQ(vec, (std::vector{0, 1, 4, 9, 16}));

  // auto assign to a vector of vectors
  std::vector<std::vector<int>> vec2(5, std::vector<int>(5));
  clef::make_expr_subscript(vec2, x0_)[x1_] << x0_ * vec2.size() + x1_;
  for (int i = 0; i < vec2.size(); ++i) {
    for (int j = 0; j < vec2[0].size(); ++j) { EXPECT_EQ(vec2[i][j], i * vec2.size() + j); }
  }

  // auto assign to a vector using another vector
  std::vector<int> vec3{1, 2, 3}, vec4(3, 0);
  clef::make_expr(vec4)[x0_] << x0_ + clef::make_expr(vec3)[x0_];
  EXPECT_EQ(vec4[0], 1);
  EXPECT_EQ(vec4[1], 3);
  EXPECT_EQ(vec4[2], 5);

  // auto assign to custom type via standard subscript operator
  bar b1;
  clef::make_expr(b1)[x0_] << x0_ * x0_;
  for (int i = 0; i < b1.arr.size(); ++i) { EXPECT_EQ(b1.arr[i], i * i); }

  // auto assign to custom type via lazy function call operator
  bar b2;
  b2(x0_) << x0_ + x0_;
  for (int i = 0; i < b2.arr.size(); ++i) { EXPECT_EQ(b2.arr[i], i + i); }

  // misuse of auto assign (should not compile)
  // bar b3;
  // b3(x0_ + x1_) << x0_ * x1_;
}

TEST_F(CLEF, LazyMethod) {
  foo f1(10);
  EXPECT_EQ(clef::eval(f1.lazy(x0_), x0_ = 2), f1.lazy(2));
  auto ex1 = f1.lazy(x0_);
  f1.x     = 20;
  EXPECT_EQ(clef::eval(ex1, x0_ = 2), f1.lazy(2));
}

TEST_F(CLEF, AvoidCopies) {
  // full evaluation
  auto ex1  = x0_ + x1_;
  auto res1 = nda::clef::eval(ex1, x0_ = foo(2), x1_ = foo(10));
  EXPECT_EQ(res1.x, 12);

  // partial evaluation
  auto ex2  = nda::clef::eval(ex1, x0_ = foo(3));
  auto res2 = nda::clef::eval(ex2, x1_ = foo(2));
  EXPECT_EQ(res2.x, 5);
}

TEST_F(CLEF, MathFunctions) {
  // cosine and sine
  auto ex1 = clef::cos(x0_) * clef::cos(x0_) + clef::sin(x1_) * clef::sin(x1_);
  EXPECT_EQ(clef::eval(ex1, x0_ = 0.5, x1_ = 0.5), 1.0);
  auto ex1_p0 = clef::eval(ex1, x0_ = 0.9);
  EXPECT_EQ(clef::eval(ex1_p0, x1_ = 0.9), 1.0);

  // cosh and sinh
  auto ex2 = clef::cosh(x0_) * clef::cosh(x0_) - clef::sinh(x1_) * clef::sinh(x1_);
  EXPECT_EQ(clef::eval(ex2, x0_ = 5.7, x1_ = 5.7), 1.0);
  auto ex2_p1 = clef::eval(ex2, x1_ = 5.7);
  EXPECT_EQ(clef::eval(ex2_p1, x0_ = 5.7), 1.0);

  // complex conjugate
  auto ex3 = clef::sqrt(x0_ * clef::conj(x0_));
  auto ex4 = clef::abs(x0_);
  EXPECT_EQ(clef::eval(ex3, x0_ = std::complex{1.0, 2.0}), clef::eval(ex4, x0_ = std::complex{1.0, 2.0}));
}

TEST_F(CLEF, SumExpressionOverDomain) {
  auto dom1 = std::vector{1, 2, 3};
  auto dom2 = std::vector{4, 5, 6};

  // sum over a 1-dimensional domain
  auto ex1 = x0_ * x0_;
  EXPECT_EQ(clef::sum(ex1, x0_ = dom1), 14);
  EXPECT_EQ(clef::sum(ex1, x0_ = std::vector{1, 2, 3}), 14);

  // sum over a 2-dimensional domain with the same 1-dimensional domain
  auto ex2 = x0_ + x1_;
  EXPECT_EQ(clef::sum(ex2, x0_ = dom1, x1_ = dom1), 2 + 2 * 3 + 3 * 4 + 2 * 5 + 6);
  EXPECT_EQ(clef::sum(ex2, x0_ = std::vector{1, 2, 3}, x1_ = std::vector{1, 2, 3}), 2 + 2 * 3 + 3 * 4 + 2 * 5 + 6);

  // sum over a 2-dimensional domain with different domains
  auto ex3 = x0_ + x1_;
  EXPECT_EQ(clef::sum(ex3, x0_ = dom1, x1_ = dom2), 5 + 2 * 6 + 3 * 7 + 2 * 8 + 9);
  EXPECT_EQ(dom1.size(), 3);
  EXPECT_EQ(clef::sum(ex3, x0_ = std::vector{1, 2, 3}, x1_ = std::vector{4, 5, 6}), 5 + 2 * 6 + 3 * 7 + 2 * 8 + 9);
}
