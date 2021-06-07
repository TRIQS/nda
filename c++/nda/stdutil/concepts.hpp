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

#pragma once

#include <version>
#include <concepts>

// libcpp below 13 has incomplete <concepts>
#if defined(_LIBCPP_VERSION) and _LIBCPP_VERSION < 13000

namespace std {
  template <class T>
  concept integral = std::is_integral_v<T>;
} // namespace std

#endif
