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
//
// Authors: Olivier Parcollet, Nils Wentzell

#include <nda/nda.hpp>
#include <nda/print.hpp>

struct member_access {
  nda::array<long, 1> arr                    = {1, 2, 3};
  nda::array<nda::array<long, 1>, 1> arr_arr = {{1, 2, 3}, {1, 2}};
};

nda::array<long, 1> make_arr(long n) {
  auto res = nda::array<long, 1>{std::array{n}};
  for (int i = 0; i < n; ++i) { res[i] = i; }
  return res;
}

nda::array<long, 2> make_arr(long n1, long n2) {
  auto res = nda::array<long, 2>{std::array{n1, n2}};
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n2; ++j) { res(i, j) = j + i * n2; }
  }
  return res;
}

nda::array<nda::array<long, 1>, 1> make_arr_arr(long n1, long n2) {
  auto res = nda::array<nda::array<long, 1>, 1>{std::array{n1}};
  for (int i = 0; i < n1; ++i) {
    res(i) = nda::array<long, 1>{std::array{n2}};
    for (int j = 0; j < n2; ++j) { res(i)(j) = j + i * n2; }
  }
  return res;
}

// =================== C2PY ===================

long size_arr(nda::array<long, 1> const &a) {
  std::cerr << "size_arr R 1" << std::endl;
  return a.size();
}
long size_arr(nda::array<long, 2> const &a) {
  std::cerr << "size_arr R 2" << std::endl;
  return a.size();
}

long size_arr_v(nda::array_view<long, 1> a) { return a.size(); }
long size_arr_v(nda::array_view<long, 2> a) { return a.size(); }

long size_arr_cv(nda::array_const_view<long, 1> a) { return a.size(); }
long size_arr_cv(nda::array_const_view<long, 2> a) { return a.size(); }

long size_arr_arr(nda::array<nda::array<long, 1>, 1> a) { return a.size(); }
long size_arr_arr_v(nda::array<nda::array_view<long, 1>, 1> a) { return a.size(); }
long size_arr_arr_cv(nda::array<nda::array_const_view<long, 1>, 1> a) { return a.size(); }

// =================== C2PY list ===================

template <auto R>
nda::array<long, R> multby2(nda::array<long, R> const &a) {
  return 2 * a;
}

nda::array<double, 1> multby2_d(nda::array<double, 1> const &a) { return 2 * a; }

// =================== PY2C + C2PY ===================

//auto ret_arr(nda::array<long, 1> const & a) { return a; }
//auto ret_arr(nda::array<long, 2> const & a) { return a; }

//auto ret_arr_v(nda::array_view<long, 1> a) { return a; }
//auto ret_arr_v(nda::array_view<long, 2> a) { return a; }

//auto ret_arr_cv(nda::array_const_view<long, 1> a) { return a; }
//auto ret_arr_cv(nda::array_const_view<long, 2> a) { return a; }

//auto ret_arr_arr(nda::array<nda::array<long, 1>, 1> a) { return a; }
//auto ret_arr_arr_v(nda::array<nda::array_view<long, 1>, 1> a) { return a; }
//auto ret_arr_arr_cv(nda::array<nda::array_const_view<long, 1>, 1> a) { return a; }
