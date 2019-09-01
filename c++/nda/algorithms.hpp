#pragma once

namespace nda {

  // --------------- Computation of the matrix norm ------------------------

  //template <typename T>
  ////require( is_real_or_complex<T>)
  //double frobenius_norm(matrix<T> const &a) {
    //return std::sqrt(fold([](double r, T const &x) -> double {
      //auto ab = std::abs(x);
      //return r + ab * ab;
    //}, a, double(0)));
  //}

  // --------------- Check if is finite ------------------------

  /// Returns true iif at least one element of the array is true
  template <typename A>
  bool any(A const &a) REQUIRES(is_ndarray_v<A>) {
    static_assert(std::is_same_v<get_value_t<A>, bool>, "OOPS");
    return fold([](bool r, auto const &x) -> bool { return r or bool(x); }, a, false);
  }

  /// Returns true iif all elements of the array are true
  template <typename A>
  bool all(A const &a) REQUIRES(is_ndarray_v<A>) {
    static_assert(std::is_same_v<get_value_t<A>, bool>, "OOPS");
    return fold([](bool r, auto const &x) -> bool { return r and bool(x); }, a, true);
  }

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
    return fold(
       [](auto const &a, auto const &b) {
         using std::max;
         return max(a, b);
       },
       a, get_first_element(a));
  }

  /**
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The minimum element of A
   */
  template <typename A>
  auto min_element(A const &a) REQUIRES(is_ndarray_v<A>) {
    return fold(
       [](auto const &a, auto const &b) {
         using std::min;
         return min(a, b);
       },
       a, get_first_element(a));
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
