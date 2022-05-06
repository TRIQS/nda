// Copyright (c) 2019-2021 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include "./map.hpp"

namespace nda {

  // ---------------  a few additional functions --------------

  template <typename T>
  auto real(T t) requires(nda::is_scalar_v<T>) {
    if constexpr (is_complex_v<T>) {
      return std::real(t);
    } else {
      return t;
    }
  }

  template <typename T>
  auto conj(T t) requires(nda::is_scalar_v<T>) {
    if constexpr (is_complex_v<T>) {
      return std::conj(t);
    } else {
      return t;
    }
  }

  ///
  inline double abs2(double x) { return x * x; }

  ///
  inline double abs2(std::complex<double> x) { return (conj(x) * x).real(); }

  ///
  inline bool isnan(std::complex<double> const &z) { return std::isnan(z.real()) or std::isnan(z.imag()); }

  /// FIXME ??
  //  inline bool any(bool x) { return x; } // for generic codes

  /// pow for integer
  template <typename T>
  T pow(T x, int n) requires(std::is_integral_v<T>) {
    T r = 1;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
  }

  // ---------------  The Mapped function using map--------------

  // can not use a macro or I can not write the doc !

  /// Map pow on Ndarray
  template <Array A>
  auto pow(A &&a, int n) {
    return nda::map([n](auto const &x) {
      using std::pow;
      return pow(x, n);
    })(std::forward<A>(a));
  }

} // namespace nda
