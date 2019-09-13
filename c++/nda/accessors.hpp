#pragma once

namespace nda::access {

  // same as std:: ...

  template <typename T>
  struct basic {
    using element_type = T;
    using pointer      = T *;
    using reference    = T &;
    reference access(pointer p, std::ptrdiff_t i) const noexcept { return p[i]; }
    reference offset(pointer p, std::ptrdiff_t i) const noexcept { return p + i; }
  };

  template <typename T>
  struct no_alias {
    using element_type = T;
    using pointer      = T *__restrict;
    using reference    = T &;
    reference access(pointer p, std::ptrdiff_t i) const noexcept { return p[i]; }
    reference offset(pointer p, std::ptrdiff_t i) const noexcept { return p + i; }
  };

  // atomic ?

} // namespace nda::access
