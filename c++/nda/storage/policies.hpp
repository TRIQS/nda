#pragma once
#include "./handle.hpp"

namespace nda {

  // Policy classes
  struct heap {
    template <typename T>
    using handle = ::nda::mem::handle<T, 'R'>;
  };

  struct stack {
    template <typename T>
    using handle = ::nda::mem::handle<T, '?'>;
  };

  struct shared {
    template <typename T>
    using handle = ::nda::mem::handle<T, 'S'>;
  };

  struct borrowed {
    template <typename T>
    using handle = ::nda::mem::handle<T, 'B'>;
  };

} // namespace nda::mem
