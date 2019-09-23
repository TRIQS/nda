#include "./common.hpp"

// must remove shadow here, the _TEST macro use it heavily on purpose
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

#define _TEST_1a(EXPR)                                                                                                                               \
  EXPECT_EQ(eval(EXPR, x_ = 1, y_ = 2), [&](int x_, int) { return EXPR; }(1, 2));                                                                    \
  EXPECT_EQ(eval(eval(EXPR, x_ = x_ + y_), x_ = 1, y_ = 2), [&](int x_, int) { return EXPR; }(3, 2));

#define _TEST_1(EXPR)                                                                                                                                \
  EXPECT_EQ(eval(EXPR, x_ = 1, y_ = 2), [&](int x_, int y_) { return EXPR; }(1, 2));                                                                 \
  EXPECT_EQ(eval(eval(EXPR, x_ = x_ + y_), x_ = 1, y_ = 2), [&](int x_, int y_) { return EXPR; }(3, 2));

//#pragma GCC diagnostic pop

TEST(clef, eval) {
  F1 f(7);

  _TEST_1a(5 * x_);
  _TEST_1a(f(x_));

  _TEST_1(x_ + 2 * y_);
  _TEST_1(x_ + 2 * y_ + x_);
  _TEST_1(x_ / 2.0 + 2 * y_);
  _TEST_1(f(x_) + 2 * y_);
  _TEST_1(1 / f(x_) + 2 * y_);
}

// -----------------------

TEST(clef, makefunction) {

  auto expr  = x_ + 2 * y_;
  auto myf   = make_function(expr, x_, y_);
  auto myf_r = make_function(expr, y_, x_);

  EXPECT_EQ(myf(2, 5), 12);
  EXPECT_EQ(myf(5, 2), 9);
  EXPECT_EQ(myf_r(2, 5), 9);
  EXPECT_EQ(myf_r(5, 2), 12);
}

// -----------------------

TEST(clef, makefunctionparametric) {
  auto expr = 2 * x_ + 1;
  auto r    = make_function(expr, x_);
  auto r2   = x_ >> expr;
  EXPECT_EQ(r(3), 7);
  EXPECT_EQ(r2(3), 7);

  EXPECT_EQ((eval(make_function(x_ + 2 * y_, x_), y_ = 2)(3)), 7);
  EXPECT_EQ((make_function(x_ + 2, x_)(3)), 5);
}

// -----------------------

TEST(clef, autoassign) {
  F1 f(7);
  EXPECT_EQ(f(2), 20);
  EXPECT_EQ(f.v, 7);

  f(x_) << 8 * x_;
  EXPECT_EQ(f.v, 8 * 7);

  // f(x_ + y_) << 8*x_ ; // SHOULD NOT COMPILE
}

// -----------------------

TEST(clef, F2) {
  double x = 1, y = 2;
  F2 ff;
  EXPECT_EQ(eval(ff(x_, y_) + 2 * y_, x_ = x, y_ = y), ff(x, y) + 2 * y);
  EXPECT_EQ(eval(ff(x_, 2), x_ = x), 12);
}

// -----------------------

TEST(clef, elseif) {
  EXPECT_EQ(eval(if_else(true, 2 * x_, y_), x_ = 1, y_ = 3), 2);
  EXPECT_EQ(eval(if_else(false, 2 * x_, y_), x_ = 1, y_ = 3), 3);
  EXPECT_EQ(eval(if_else(x_ > y_, 2 * x_, y_), x_ = 1, y_ = 3), 3);
  EXPECT_PRINT(std::string{"(_1 < _2)"}, (x_ < y_));
}

struct Obj {
  double v;                  // put something in it
  Obj(double v_) : v(v_) {}  // constructor
  Obj(Obj const &) = delete; // a non copyable object, to illustrate that we do NOT copy...

  // a method
  double my_method(double d) const { return 2 * d; }

  // CLEF overload
  CLEF_IMPLEMENT_LAZY_METHOD(Obj, my_method)

  // Just to print itself nicely in the expressions
  friend std::ostream &operator<<(std::ostream &out, Obj const &) { return out << "Obj"; }
};

// -----------------------
TEST(clef, lazymethod) {
  Obj f(7);
  EXPECT_EQ(eval(f.my_method(x_) + 2 * x_, x_ = 1), 4);
  EXPECT_EQ(eval(f.my_method(y_) + 2 * x_, x_ = 3, y_ = 1), 8);
  std::cerr << "Clef expression     : " << f.my_method(y_) + 2 * x_ << std::endl;
  std::cerr << "Partial evaluation  : " << eval(f.my_method(y_) + 2 * x_, y_ = 1) << std::endl;
}

