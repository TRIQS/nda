/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include "./functional/map.hpp"
namespace nda {

  // not for libc++ (already defined)
#if !defined(_LIBCPP_VERSION)
  // complex conjugation for integers
  inline std::complex<double> conj(int x) { return x; }
  inline std::complex<double> conj(long x) { return x; }
  inline std::complex<double> conj(long long x) { return x; }
  inline std::complex<double> conj(double x) { return x; }
#endif

  inline bool isnan(std::complex<double> const &z) { return std::isnan(z.real()) or std::isnan(z.imag()); }
  inline bool any(bool x) { return x; } // for generic codes   FIXME ????

#define MAP_IT_IMPL(FNT, CONDITION)                                                                                                                  \
  template <typename A> decltype(auto) FNT(A &&a) REQUIRES(CONDITION) {                                                                              \
    return map(                                                                                                                                      \
       [](auto const &x) {                                                                                                                           \
         using std::FNT;                                                                                                                             \
         return FNT(a);                                                                                                                              \
       },                                                                                                                                            \
       std::forward<A>(a));                                                                                                                          \
  }

#define MAP_IT_ARRAY_VECTOR_ONLY(FNT) MAP_IT_IMPL(FNT, ImmutableArray<A>::value || ImmutableVector<A>)
#define MAP_IT(FNT) MAP_IT_IMPL(FNT, ImmutableCuboidArray<A>)

  MAP_IT(abs);
  MAP_IT(imag);
  MAP_IT(floor);
  MAP_IT(conj);
  MAP_IT(isnan);

  MAP_IT_ARRAY_VECTOR_ONLY(sqrt);
  MAP_IT_ARRAY_VECTOR_ONLY(log);
  MAP_IT_ARRAY_VECTOR_ONLY(exp);
  MAP_IT_ARRAY_VECTOR_ONLY(cos);
  MAP_IT_ARRAY_VECTOR_ONLY(sin);
  MAP_IT_ARRAY_VECTOR_ONLY(tan);
  MAP_IT_ARRAY_VECTOR_ONLY(cosh);
  MAP_IT_ARRAY_VECTOR_ONLY(sinh);
  MAP_IT_ARRAY_VECTOR_ONLY(tanh);
  MAP_IT_ARRAY_VECTOR_ONLY(acos);
  MAP_IT_ARRAY_VECTOR_ONLY(asin);
  MAP_IT_ARRAY_VECTOR_ONLY(atan);

#undef MAP_IT_IMPL
#undef MAP_IT_ARRAY_VECTOR_ONLY
#undef MAP_IT

  /// Power function
  template <typename A> decltype(auto) pow(A &&a, int n) REQUIRES(ImmutableCuboidArray<A>) {
    return map(
       [n](auto const &x) {
         using std::pow;
         return pow(a, n);
       },
       std::forward<A>(a));
  }

  // --------------- Computation of the matrix norm ------------------------

  template <typename T>
  //require( is_real_or_complex<T>)
  double frobenius_norm(matrix<T> const &a) {
    return std::sqrt(fold([](double r, T const &x) -> double {
      auto ab = std::abs(x);
      return r + ab * ab;
    })(a, double(0)));
  }

  // --------------- all and any

  /// Returns true iif at least one element of the array is true
  template <typename A> bool any(A const &a) REQUIRES(ImmutableCuboidArray<A>) {
    return fold([](bool r, auto const &x) -> bool { return r or bool(x); })(a, false);
  }

  /// Returns true iif all elements of the array are true
  template <typename A> bool all(A const &a) REQUIRES(ImmutableCuboidArray<A>) {
    return fold([](bool r, auto const &x) -> bool { return r and bool(x); })(a, true);
  }
} // namespace nda
