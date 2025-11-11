#include <gtest/gtest.h>
#include <nda/gtest_tools.hpp>
#include <type_traits>
#include <cmath>
#include <array>
#include <complex>
#include <algorithm>
#include <bit>
#include <nda/simd/mock_simd.hpp>
#include <nda/simd/simd.hpp>
using namespace nda;

template <typename T>
constexpr void expect_eq(const T a, const T b) {
  using U = std::decay_t<decltype(a)>;
  if constexpr (std::is_same_v<U, float>) {
    EXPECT_FLOAT_EQ(a, b);
  } else if constexpr (std::is_same_v<U, double>) {
    EXPECT_DOUBLE_EQ(a, b);
  } else if constexpr (std::is_same_v<U, std::complex<float>>) {
    EXPECT_FLOAT_EQ(a.real(), b.real());
    EXPECT_FLOAT_EQ(a.imag(), b.imag());
  } else if constexpr (std::is_same_v<U, std::complex<double>>) {
    EXPECT_DOUBLE_EQ(a.real(), b.real());
    EXPECT_DOUBLE_EQ(a.imag(), b.imag());
  } else {
    EXPECT_EQ(a, b);
  }
}

template <typename T, size_t R, typename Layout, char Algebra>
void load_and_store() {

  std::array<size_t, R> shape;
  shape.fill(32);

  using array_t = basic_array<T, R, Layout, Algebra, heap<>>;
  array_t A     = rand(shape);
  using idx_t   = std::array<long, R>;

  // --- A.load/store tests ---
  idx_t idx1{};
  auto a              = std::apply([&](auto... args) { return A.load(simd::vectorize, args...); }, idx1);
  constexpr auto size = decltype(a)::size;

  for (size_t i = 0; i < size; ++i) {
    std::apply([&](auto... args) { expect_eq(a.get(i), A(args...)); }, idx1);
    ++idx1[A.stride_order()[R - 1]];
  }

  idx_t idx2{};
  idx2[A.stride_order()[R - 1]] += size;
  auto b = std::apply([&](auto... args) { return A.load(simd::vectorize, args...); }, idx2);
  for (size_t i = 0; i < size; ++i) {
    std::apply([&](auto... args) { expect_eq(b.get(i), A(args...)); }, idx2);
    ++idx2[A.stride_order()[R - 1]];
  }

  idx_t idx3{};
  auto c = a + b;
  std::apply([&](auto... args) { A.store(c, args...); }, idx3);
  for (size_t i = 0; i < size; ++i) {
    std::apply([&](auto... args) { expect_eq(c.get(i), A(args...)); }, idx3);
    ++idx3[A.stride_order()[R - 1]];
  }

  // --- Expression tests: B + C, B - C, B * C, B / C ---
  array_t B = rand(shape);
  array_t C = rand(shape);
  for (auto &x : C) {
    using value_t = std::decay_t<decltype(x)>;
    if constexpr (std::is_arithmetic_v<value_t>) {
      if (x == value_t(0)) x = value_t(1);
    } else if constexpr (is_complex_v<value_t>) {
      if (x == value_t(0, 0)) x = value_t(1, 2);
    }
  }

  // Expression 1: B + C
  auto e1 = B + C;
  {
    idx_t idx{};
    auto v = std::apply([&](auto... args) { return e1.load(simd::vectorize, args...); }, idx);
    for (size_t i = 0; i < size; ++i) {
      std::apply([&](auto... args) { expect_eq(v.get(i), e1(args...)); }, idx);
      ++idx[B.stride_order()[R - 1]];
    }
  }

  // Expression 2: B - C
  auto e2 = B - C;
  {
    idx_t idx{};
    auto v = std::apply([&](auto... args) { return e2.load(simd::vectorize, args...); }, idx);
    for (size_t i = 0; i < size; ++i) {
      std::apply([&](auto... args) { expect_eq(v.get(i), e2(args...)); }, idx);
      ++idx[B.stride_order()[R - 1]];
    }
  }

  if constexpr (Algebra == 'A') {
    // Expression 3: B * C
    auto e3 = B * C;
    {
      idx_t idx{};
      auto v = std::apply([&](auto... args) { return e3.load(simd::vectorize, args...); }, idx);
      for (size_t i = 0; i < size; ++i) {
        std::apply([&](auto... args) { expect_eq(v.get(i), e3(args...)); }, idx);
        ++idx[B.stride_order()[R - 1]];
      }
    }
    // Expression 4: B / C
    auto e4 = B / C;
    {
      idx_t idx{};
      auto v = std::apply([&](auto... args) { return e4.load(simd::vectorize, args...); }, idx);
      for (size_t i = 0; i < size; ++i) {
        std::apply([&](auto... args) { expect_eq(v.get(i), e4(args...)); }, idx);
        ++idx[B.stride_order()[R - 1]];
      }
    }
  }
  if constexpr (Algebra == 'M') {
    auto e5 = B + 5;
    {
      idx_t idx{};
      auto v = std::apply([&](auto... args) { return e1.load(simd::vectorize, args...); }, idx);
      for (size_t i = 0; i < size; ++i) {
        std::apply([&](auto... args) { expect_eq(v.get(i), e1(args...)); }, idx);
        ++idx[B.stride_order()[R - 1]];
      }
    }
  }
}

template <typename T>
struct scalar_f {
  template <typename... Args>
  T operator()(const Args &...args) const {
    return (args + ...) + T{1};
  }
};

template <typename T>
struct emulated_f : simd::mock_simd<emulated_f<T>, T> {
  template <typename... Args>
  T operator()(const Args &...args) const {
    return (args + ...) + T{1};
  }
};

template <typename T, size_t Rank, typename Layout>
void mock_simd_test() {
  std::array<size_t, Rank> shape;
  shape.fill(32);

  using array_t = array<T, Rank, Layout>;
  using idx_t   = std::array<long, Rank>;

  array_t A = rand(shape);
  array_t B = rand(shape);
  array_t C = rand(shape);

  // Generate expression results
  auto r1 = map(emulated_f<T>{})(A);
  auto r2 = map(emulated_f<T>{})(A, B);
  auto r3 = map(emulated_f<T>{})(A, B, C);

  auto s1 = map(scalar_f<T>{})(A);
  auto s2 = map(scalar_f<T>{})(A, B);
  auto s3 = map(scalar_f<T>{})(A, B, C);

  auto check = [&](const auto &r, const auto &s) {
    idx_t idx{};
    auto v              = std::apply([&](auto... args) { return r.load(simd::vectorize, args...); }, idx);
    constexpr auto size = decltype(v)::size;

    for (size_t i = 0; i < size; ++i) {
      std::apply([&](auto... args) { expect_eq(v.get(i), s(args...)); }, idx);
      ++idx[A.stride_order()[Rank - 1]];
    }
  };
  check(r1, s1);
  check(r2, s2);
  check(r3, s3);
  EXPECT_ARRAY_NEAR(r1, s1);
  EXPECT_ARRAY_NEAR(r2, s2);
  EXPECT_ARRAY_NEAR(r3, s3);
}

TEST(NDA, LoadAndStore) {
  // Macro to test all ranks (1 to 3) for a given type, layout, and algebra
#define TEST_ALGEBRA_A(type, layout)                                                                                                                 \
  load_and_store<type, 1, layout, 'A'>();                                                                                                            \
  load_and_store<type, 2, layout, 'A'>();                                                                                                            \
  load_and_store<type, 3, layout, 'A'>();

#define TEST_ALGEBRA_M(type, layout) load_and_store<type, 2, layout, 'M'>(); // Only rank 2 for matrix algebra

#define TEST_ALGEBRA_V(type, layout) load_and_store<type, 1, layout, 'V'>(); // Only rank 1 for vector algebra

  // Macro to test all relevant algebra types (A, M, V) for a given type
#define TEST_ALL_ALGEBRAS(type)                                                                                                                      \
  TEST_ALGEBRA_A(type, C_layout)                                                                                                                     \
  TEST_ALGEBRA_A(type, F_layout)                                                                                                                     \
  TEST_ALGEBRA_M(type, C_layout)                                                                                                                     \
  TEST_ALGEBRA_M(type, F_layout)                                                                                                                     \
  TEST_ALGEBRA_V(type, C_layout)                                                                                                                     \
  TEST_ALGEBRA_V(type, F_layout)

  // Call tests for all relevant types
  TEST_ALL_ALGEBRAS(int32_t)
  TEST_ALL_ALGEBRAS(int64_t)
  TEST_ALL_ALGEBRAS(uint32_t)
  TEST_ALL_ALGEBRAS(uint64_t)
  TEST_ALL_ALGEBRAS(float)
  TEST_ALL_ALGEBRAS(double)
  TEST_ALL_ALGEBRAS(std::complex<float>)
  TEST_ALL_ALGEBRAS(std::complex<double>)

#undef TEST_ALGEBRA_A
#undef TEST_ALGEBRA_M
#undef TEST_ALGEBRA_V
#undef TEST_ALL_ALGEBRAS
}

TEST(NDA, MOCK_SIMD) {
  // Run mock_simd_test for all ranks for a given type and layout
#define TEST_ALL_RANKS(T, L)                                                                                                                         \
  mock_simd_test<T, 1, L>();                                                                                                                         \
  mock_simd_test<T, 2, L>();                                                                                                                         \
  mock_simd_test<T, 3, L>();

  // Run for both C and F layouts
#define TEST_LAYOUTS(T)                                                                                                                              \
  TEST_ALL_RANKS(T, C_layout)                                                                                                                        \
  TEST_ALL_RANKS(T, F_layout)

  TEST_LAYOUTS(int32_t)
  TEST_LAYOUTS(int64_t)
  TEST_LAYOUTS(uint32_t)
  TEST_LAYOUTS(uint64_t)
  TEST_LAYOUTS(float)
  TEST_LAYOUTS(double)
  TEST_LAYOUTS(std::complex<float>)
  TEST_LAYOUTS(std::complex<double>)

#undef TEST_ALL_RANKS
#undef TEST_LAYOUTS
}

TEST(NDA, EXPR_TREE_MODEL) {
  using namespace nda::simd;
  std::array<size_t, 2> shape;
  shape.fill(32);
  using array_t = array<float, 2>;

  array_t A = rand(shape);
  array_t B = rand(shape);
  array_t C = rand(shape);

  static_assert(simd_gain<float>() == 0);
  static_assert(simd_gain<array_t>() == 1);

  auto e1 = A + B + C;
  static_assert(simd_gain<decltype(e1)>() == 5);

  auto e2 = e1 + A;
  static_assert(simd_gain<decltype(e2)>() == 7);

  auto e3 = nda::map([](auto const &x) { return x * x; })(e2);
  static_assert(simd_gain<decltype(e3)>() == 8);

  auto e4 = -e3;
  static_assert(simd_gain<decltype(e4)>() == 9);

  auto e5 = e4 + 5.0f;
  static_assert(simd_gain<decltype(e5)>() == 10);

  auto e6 = 5.0f * e5;
  static_assert(simd_gain<decltype(e6)>() == 11);

  auto e7 = e6 + e6;
  static_assert(simd_gain<decltype(e7)>() == 23);

  auto e8 = nda::map([](auto const &x, auto const &y) { return x - y; })(e7, e3);
  static_assert(simd_gain<decltype(e8)>() == 32);

  auto e9 = log(e8);
  static_assert(simd_gain<decltype(e9)>() == 33);

  static_assert(std::is_same_v<dispatch_policy_t<float>, scalar_t>);
  static_assert(std::is_same_v<dispatch_policy_t<array_t>, vectorize_t>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e1)>, vectorize_t>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e2)>, vectorize_t>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e3)>, std::conditional_t<simd_cost_model<decltype(e3)>::emulate(), emulate_t, scalar_t>>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e4)>, std::conditional_t<simd_cost_model<decltype(e4)>::emulate(), emulate_t, scalar_t>>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e5)>, std::conditional_t<simd_cost_model<decltype(e5)>::emulate(), emulate_t, scalar_t>>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e6)>, std::conditional_t<simd_cost_model<decltype(e6)>::emulate(), emulate_t, scalar_t>>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e7)>, std::conditional_t<simd_cost_model<decltype(e7)>::emulate(), emulate_t, scalar_t>>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e8)>, std::conditional_t<simd_cost_model<decltype(e8)>::emulate(), emulate_t, scalar_t>>);
  static_assert(std::is_same_v<dispatch_policy_t<decltype(e9)>, std::conditional_t<simd_cost_model<decltype(e9)>::emulate(), emulate_t, scalar_t>>);
}
