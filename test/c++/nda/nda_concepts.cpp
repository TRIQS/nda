#include "./test_common.hpp"

// ==============================================================

// just compile time

TEST(NDA, Concept) { // NOLINT

  static_assert(!std::is_pod<nda::array<long, 2>>::value, "POD pb");
  static_assert(nda::is_scalar_for_v<int, matrix<std::complex<double>>> == 1, "oops");

#if __cplusplus > 201703L

  using nda::Array;
  using nda::ArrayOfRank;

  static_assert(Array<nda::array<int, 2>>, "INTERNAL");
  static_assert(ArrayOfRank<nda::array<int, 2>, 2>, "INTERNAL");
  static_assert(not ArrayOfRank<nda::array<int, 2>, 1>, "INTERNAL");
  static_assert(not ArrayOfRank<nda::array<int, 2>, 3>, "INTERNAL");

  static_assert(Array<nda::array_view<int, 2>>, "INTERNAL");
  static_assert(ArrayOfRank<nda::array_view<int, 2>, 2>, "INTERNAL");
  static_assert(not ArrayOfRank<nda::array_view<int, 2>, 1>, "INTERNAL");
  static_assert(not ArrayOfRank<nda::array_view<int, 2>, 3>, "INTERNAL");

#endif
}
