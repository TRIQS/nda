#pragma once
#include "./handle.hpp"

namespace nda {

  // Policy classes
  struct heap {
    template <typename T, size_t StackSize = 0> // StackSize is ignored in this case, but called in basic_array
    using handle = ::nda::mem::handle_heap<T, void>;
  };

  template <typename Allocator>
  struct heap_custom_alloc {
#ifdef _OPENMP
    template<typename T> static constexpr bool always_true = true; // to prevent the static_assert to trigger only when instantiated
    static_assert(false and always_true<Allocator> , "Custom Allocators are not available in OpenMP");
#endif
    template <typename T, size_t StackSize = 0> // StackSize is ignored in this case, but called in basic_array
    using handle = ::nda::mem::handle_heap<T, Allocator>;
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
