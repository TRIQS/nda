#pragma once

namespace nda {

  // FIXME : CHECK ORDER or the LOOP !
  // --------------- fold  ------------------------
  /**
   * @tparam A
   * @tparam F is a function f(x, r)
   * @tparam R
   * @param f
   * @param a
   * @param r
   *
   * fold computes f(f(r, a(0,0)), a(0,1), ...)  etc
   */
  template <typename A, typename F, typename R = get_value_t<A>>
  auto fold(F f, A const &a, R r = R{}) REQUIRES(is_ndarray_v<A>) {
    decltype(f(r, get_value_t<A>{})) r2 = r;
    // to take into account that f may be double,double -> double, while one passes 0 (an int...)
    // R = int, R2= double in such case, and the result will be a double, or narrowing will occur
    nda::for_each(a.shape(), [&a, &r2, &f](auto &&... args) { r2 = f(r2, a(args...)); });
    return r2;
  }

  // --------------- applications of fold -----------------------

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

  /**
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The maximum element of A
   */
  template <typename A>
  auto max_element(A const &a) REQUIRES(is_ndarray_v<A>) {
    return fold(
       [](auto const &x, auto const &y) {
         using std::max;
         return max(x, y);
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
       [](auto const &x, auto const &y) {
         using std::min;
         return min(x, y);
       },
       a, get_first_element(a));
  }

  // FIXME in matrix functions ?

  // --------------- Computation of the matrix norm ------------------------

  /**
   * @tparam A Anything modeling NdArray
   * @param a The object of type A
   * @return The minimum element of A
   */

  //template <typename T>
  ////require( is_real_or_complex<T>)
  //double frobenius_norm(matrix<T> const &a) {
  //return std::sqrt(fold([](double r, T const &x) -> double {
  //auto ab = std::abs(x);
  //return r + ab * ab;
  //}, a, double(0)));
  //}

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
