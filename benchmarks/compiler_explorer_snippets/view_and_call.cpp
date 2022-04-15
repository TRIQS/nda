// Copyright (c) 2020 Simons Foundation
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

// clang 10 :  -std=c++20 -O3 -DNDEBUG -march=skylake
// gcc 9  :    -std=c++17 -O3 -DNDEBUG -march=skylake

#include <nda/nda.hpp>

auto _ = nda::range::all;

template <int R>
using Vs = nda::basic_array_view<
   double, R, nda::C_stride_layout, 'A',
   nda::default_accessor, nda::borrowed<>>;

template <int R>
using V = nda::basic_array_view<
   double, R, nda::C_layout, 'A',
   nda::default_accessor, nda::borrowed<>>;

double f1(V<4> v, int i, int j, int k, int l) {
  return v(k, _, l, _)(i, j);
}

double f2(V<4> v, int i, int j, int k, int l) {
  return v(k, i, l, j);
}

double g0(V<2> v, int i, int j) {
  return v(i, j);
}
double g0b(Vs<2> v, int i, int j) {
  return v(i, j);
}

double g1(V<2> v, int i, int j) {
  return v(_, _)(i, j);
}

double g2(V<2> v, int i, int j) {
  return v(i, _)(j);
}
