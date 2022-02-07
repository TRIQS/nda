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
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include "macros.hpp"

namespace nda {

  // same as std:: ...

  struct default_accessor {

    template <typename T>
    struct accessor {
      using element_type = T;
      using pointer      = T *;
      using reference    = T &;
      FORCEINLINE static reference access(pointer p, std::ptrdiff_t i) noexcept {
        EXPECTS(p != nullptr);
        return p[i];
      }
      FORCEINLINE static T *offset(pointer p, std::ptrdiff_t i) noexcept { return p + i; }
    };
  };

  struct no_alias_accessor {

    template <typename T>
    struct accessor {
      using element_type = T;
      using pointer      = T *__restrict;
      using reference    = T &;
      FORCEINLINE static reference access(pointer p, std::ptrdiff_t i) noexcept { return p[i]; }
      FORCEINLINE static T *offset(pointer p, std::ptrdiff_t i) noexcept { return p + i; }
    };
  };
  // atomic ?

} // namespace nda
