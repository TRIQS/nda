// Copyright (c) 2019-2020 Simons Foundation
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

#ifndef STD_ADDONS_COMPLEX_H
#define STD_ADDONS_COMPLEX_H

#include <complex>

// Add doc manually
namespace std { // has to be in the right namespace for ADL !

  // clang-format off
  template <typename T> std::complex<T> operator+(std::complex<T> const &a, long b) { return a + T(b); }
  template <typename T> std::complex<T> operator+(long a, std::complex<T> const &b) { return T(a) + b; }
  template <typename T> std::complex<T> operator-(std::complex<T> const &a, long b) { return a - T(b); }
  template <typename T> std::complex<T> operator-(long a, std::complex<T> const &b) { return T(a) - b; }
  template <typename T> std::complex<T> operator*(std::complex<T> const &a, long b) { return a * T(b); }
  template <typename T> std::complex<T> operator*(long a, std::complex<T> const &b) { return T(a) * b; }
  template <typename T> std::complex<T> operator/(std::complex<T> const &a, long b) { return a / T(b); }
  template <typename T> std::complex<T> operator/(long a, std::complex<T> const &b) { return T(a) / b; }
  // clang-format on 

} // namespace std
#endif
