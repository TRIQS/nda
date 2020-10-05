// Copyright (c) 2018-2020 Simons Foundation
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

#pragma once
#include <ostream>

#include <itertools/itertools.hpp>

namespace nda {

  // Elevate range implementation from itertools
  using itertools::range;

  /// Ellipsis can be provided in place of [[range]], as in python. The type `ellipsis` is similar to [[range_all]] except that it is implicitly repeated to as much as necessary.
  struct ellipsis : range::all_t {};

  inline std::ostream &operator<<(std::ostream &os, range::all_t) noexcept { return os << "_"; }
  inline std::ostream &operator<<(std::ostream &os, ellipsis) noexcept { return os << "___"; }

  // impl details : detects ellipsis in a argument pack
  template <typename... T>
  constexpr bool ellipsis_is_present = ((std::is_same_v<T, ellipsis> ? 1 : 0) + ... + 0); // +0 because it can be empty

} // namespace nda
