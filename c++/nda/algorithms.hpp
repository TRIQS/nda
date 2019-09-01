#pragma once

namespace nda {

  // FIXME ? For some generic code ? needed ? Cf TRIQS and comment
  //inline double max_element(double x) { return x; }
  //inline std::complex<double> max_element(std::complex<double> x) { return x; }

  /**
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The maximum element of A
   */
  template <typename A>
  auto max_element(A const &a) REQUIRES(is_ndarray_v<A>) {
    return fold([](auto const &a, auto const &b) {
      using std::max;
      return max(a, b);
    }, a, get_first_element(a));
  }

  /**
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The minimum element of A
   */
  template <typename A>
  auto min_element(A const &a) REQUIRES(is_ndarray_v<A>) {
    return fold([](auto const &a, auto const &b) {
      using std::min;
      return min(a, b);
    }, a, get_first_element(a));
  }

  /**
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The sum of all elements of a 
   */

  template <typename A>
  auto sum(A const &a) REQUIRES(is_ndarray_v<A>) {
    return fold(std::plus<>{}, a);
  }

} // namespace nda

