#pragma once

namespace nda {

  // same as std:: ...

  struct default_accessor {

    template <typename T>
    struct accessor {
      using element_type = T;
      using pointer      = T *;
      using reference    = T &;
      reference access(pointer p, std::ptrdiff_t i) const noexcept { return p[i]; }
      pointer offset(pointer p, std::ptrdiff_t i) const noexcept { return p + i; }
    };
  };

  struct no_alias_accessor {

    template <typename T>
    struct accessor {
      using element_type = T;
      using pointer      = T *__restrict;
      using reference    = T &;
      reference access(pointer p, std::ptrdiff_t i) const noexcept { return p[i]; }
      pointer offset(pointer p, std::ptrdiff_t i) const noexcept { return p + i; }
    };
  };
  // atomic ?

} // namespace nda
