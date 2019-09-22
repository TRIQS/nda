#include "./common.hpp"

double foo(double x) { return x / 2; }
int foo(int x) { return x / 2; }

double bar(double x, double y) { return x + y; }

namespace clef {
    using ::bar;
    using ::foo;

    template <typename T> typename std::enable_if<!clef::is_any_lazy<T>::value, T>::type inc(T const &x) { return x + 1; }

    CLEF_MAKE_FNT_LAZY(bar);
    CLEF_MAKE_FNT_LAZY(inc);
    CLEF_MAKE_FNT_LAZY(foo);
} 

#define _TEST_3(EXPR)                                                                                                                                \
  EXPECT_EQ(eval(EXPR, x_ = 2), [&](int x_) { return EXPR; }(2));                                                                 \

TEST(clef, mathfunction) { 

  //_TEST_3(cos(x_));
  //_TEST_3(cos(2 * x_ + 1));
  //_TEST_3(abs(2 * x_ - 1));
  _TEST_3(foo(2 * x_ + 1));
  _TEST_3(foo(2 * x_ + 1));
  _TEST_3(inc(2 * x_ + 1));
  
  EXPECT_EQ(eval(bar(2 * x_ + 1, x_ - 1), x_ = 2), 6);
}
