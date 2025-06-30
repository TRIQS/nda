#include <gtest/gtest.h>
#include <nda/gtest_tools.hpp>
#include <type_traits>
#include <cmath>
#include <array>
#include <complex>
#include <algorithm>
#include <bit>
#include <nda/simd/mock_simd.hpp>

#include <xsimd/xsimd.hpp>
// Scalar-only functor (no SIMD support)
struct log_scalar {
  float operator()(float x) const {
    return std::log(x);
  }
};

// Emulated SIMD functor using mock_simd
struct log_emulated : nda::simd::mock_simd<log_emulated, float> {
  float operator()(float x) const {
    return std::log(x);
  }
};
TEST(NDA, OurSIMD) {
  nda::array<float,1> A(std::array{100});
  for (int i = 0 ; i < 100 ; ++i) {
    A[i] = i+1;
  }
  using simd_t = nda::native_simd<float>;
  nda::native_simd<int> a(1);
  auto result1 = nda::map(log_scalar{})(A);  // Scalar evaluation
  auto result2 = nda::map(log_emulated{})(A);    // Evaluates vectorized with mock_simd fallback
  std::cout << result2.load(0) << std::endl;
  std::cout << result2.load(0) + xsimd::batch_cast<float>(a) << std::endl;

}