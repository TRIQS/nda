// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/nda.hpp>

#include <complex>
#include <cstdint>
#include <limits>
#include <random>

using namespace nda::rng;

namespace {

  template <typename T, typename E>
  void check_range() {
    auto rng = random_number_generator<E>{42};
    auto a   = nda::array<T, 2>(20, 30);
    if constexpr (nda::is_complex_v<T>) {
      rng.fill(a, rectangle{{3, -2}, {1.5, 0.5}});
      for (auto x : a) {
        EXPECT_TRUE(x.real() >= -2 and x.real() < 3);
        EXPECT_TRUE(x.imag() >= 0.5 and x.imag() < 1.5);
      }
    } else {
      rng.fill(a, interval{T(10), T(2)});
      EXPECT_GE(nda::min_element(a), T(2));
      if constexpr (std::integral<T>) {
        EXPECT_LE(nda::max_element(a), T(10));
      } else {
        EXPECT_LT(nda::max_element(a), T(10));
      }
    }
  }

  template <typename E>
  void check_all_types() {
    check_range<int, E>();
    check_range<unsigned, E>();
    check_range<long, E>();
    check_range<float, E>();
    check_range<double, E>();
    check_range<std::complex<float>, E>();
    check_range<std::complex<double>, E>();
  }

} // namespace

TEST(NDA, RandomTypesAndEngines) {
  check_all_types<std::mt19937_64>();
  check_all_types<std::mt19937>();     // 32 bits per call
  check_all_types<std::ranlux48>();    // 48 bits per call
  check_all_types<std::minstd_rand>(); // range not a power of two
}

TEST(NDA, RandomReproducible) {
  // the same values with every standard library, through every path
  auto const expected = nda::array<double, 1>{0x1.82a3befaddcbcp-1, 0x1.472f1f73724ap-1, 0x1.81192cfe1cbcfp-1, 0x1.171621fc50d68p-3};
  global_rng().seed(42);
  EXPECT_EQ(nda::rand<double>(4), expected);
  auto rng = random_number_generator{42};
  EXPECT_EQ(rng.random<double>(), expected(0));
  EXPECT_EQ(rng.uniform<double>(interval{-1.0, 1.0}), 2 * expected(1) - 1);
}

TEST(NDA, RandomEdgeCases) {
  auto rng = random_number_generator{42};
  EXPECT_EQ(rng.uniform<int>(interval{3, 3}), 3);
  EXPECT_EQ(rng.uniform<double>(interval{2.0, 2.0}), 2.0);
  auto const full = std::numeric_limits<std::uint64_t>::max();
  EXPECT_LE(rng.uniform<std::uint64_t>(interval{std::uint64_t{0}, full}), full);
  auto const coin = rng.random<int>();
  EXPECT_TRUE(coin == 0 or coin == 1);
  EXPECT_THROW((uniform_distribution<double>{interval{-1e308, 1e308}}), nda::runtime_error);
}

TEST(NDA, RandomFillWithDistribution) {
  auto a = nda::array<float, 1>(10);
  auto b = nda::array<float, 1>(10);
  random_number_generator{1}.fill<uniform_distribution>(a, interval{2, 10});
  random_number_generator{1}.fill(b, uniform_distribution<float>{interval{2, 10}});
  EXPECT_EQ(a, b);

  auto n = nda::array<double, 1>(1000);
  random_number_generator{1}.fill<std::normal_distribution>(n, 5.0, 0.1);
}

TEST(NDA, RandomFillLayoutsAndViews) {
  // arrays are filled in memory order: a Fortran-layout matrix holds the transpose of a C-layout one
  auto C = nda::matrix<double>(4, 5);
  auto F = nda::matrix<double, nda::F_layout>(5, 4);
  random_number_generator{3}.fill(C, interval{0.0, 1.0});
  random_number_generator{3}.fill(F, interval{0.0, 1.0});
  EXPECT_EQ(nda::transpose(C), F);

  // a temporary strided view is filled and the rest of the matrix is left alone
  auto m = nda::matrix<int>::zeros(5, 5);
  random_number_generator{3}.fill<uniform_distribution>(m(nda::range::all, 0), interval{1, 2});
  for (auto x : m(nda::range::all, 0)) EXPECT_TRUE(x == 1 or x == 2);
  m(nda::range::all, 0) = 0;
  EXPECT_EQ(m, nda::matrix<int>::zeros(5, 5));
}
