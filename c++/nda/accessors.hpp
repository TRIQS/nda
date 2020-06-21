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
