// Copyright (c) 2020-2021 Simons Foundation
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

#include "stdutil/array.hpp"
#include "concepts.hpp"

namespace nda {

  /// A pair shape + lambda --> an immutable array
  template <int R, typename F>
  class array_adapter {
    static_assert(CallableWithLongs<F, R>, "Lambda should be callable with R integers");

    std::array<long, R> myshape;
    F f;

    public:
    template <typename Int>
    array_adapter(std::array<Int, R> const &shape, F f) : myshape(stdutil::make_std_array<long>(shape)), f(f) {}

    std::array<long, R> const &shape() const { return myshape; }
    [[nodiscard]] long size() const { return stdutil::product(myshape); }

    template <typename... Long>
    auto operator()(long i, Long... is) const {
      static_assert((std::is_convertible_v<Long, long> and ...), "Arguments must be convertible to long");
      return f(i, is...);
    }
  }; // namespace nda

  // CTAD
  template <auto R, typename Int, typename F>
  array_adapter(std::array<Int, R>, F) -> array_adapter<R, F>;

  template <int R, typename F>
  inline constexpr char get_algebra<array_adapter<R, F>> = 'A';

} // namespace nda
