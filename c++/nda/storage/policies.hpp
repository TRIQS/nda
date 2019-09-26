#pragma once
#include "./handle.hpp"

namespace nda {

  // Policy classes
  struct heap {
    template <typename T, size_t StackSize = 0> // StackSize is ignored in this case
    using handle = ::nda::mem::handle_heap<T>;
  };

  struct stack {
    template <typename T, size_t StackSize>
    using handle = ::nda::mem::handle_stack<T, StackSize>;
  };

  struct shared {
    template <typename T>
    using handle = ::nda::mem::handle_shared<T>;
  };

  struct borrowed {
    template <typename T>
    using handle = ::nda::mem::handle_borrowed<T>;
  };

} // namespace nda
