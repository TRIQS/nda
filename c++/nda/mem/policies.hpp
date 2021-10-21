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

#pragma once
#include "./handle.hpp"

namespace nda {

  // Policy classes
  struct heap {

#if !defined(NDA_TEST_DEFAULT_ALLOC_SSO) and !defined(NDA_TEST_DEFAULT_ALLOC_MBUCKET)
    // Normal version.
    template <typename T, size_t StackSize = 0> // StackSize is ignored in this case, but called in basic_array
    using handle = ::nda::mem::handle_heap<T, void>;
#else
    // FOR TESTS ONLY : To be able to rerun all tests with the SSO as a default
#ifdef NDA_TEST_DEFAULT_ALLOC_SSO
    template <typename T, size_t StackSize = 0>
    using handle =
       std::conditional_t<std::is_copy_constructible_v<T>, ::nda::mem::handle_sso<T, NDA_TEST_DEFAULT_ALLOC_SSO>, ::nda::mem::handle_heap<T, void>>;
#endif
    // FOR TESTS ONLY : To be able to rerun all tests with another allocator
#ifdef NDA_TEST_DEFAULT_ALLOC_MBUCKET
    using test_buck_alloc_t =
       nda::mem::segregator<8 * NDA_TEST_DEFAULT_ALLOC_MBUCKET, nda::mem::multiple_bucket<8 * NDA_TEST_DEFAULT_ALLOC_MBUCKET>, nda::mem::mallocator>;
    template <typename T, size_t StackSize = 0>
    using handle = ::nda::mem::handle_heap<T, test_buck_alloc_t>;
#endif
#endif
  };

  template <typename Allocator>
  struct heap_custom_alloc {
#ifdef _OPENMP
    template <typename T>
    static constexpr bool always_true = true; // to prevent the static_assert to trigger only when instantiated
    static_assert(false and always_true<Allocator>, "Custom Allocators are not available in OpenMP");
#endif
    template <typename T, size_t StackSize = 0> // StackSize is ignored in this case, but called in basic_array
    using handle = ::nda::mem::handle_heap<T, Allocator>;
  };

  template <size_t SSO_Size>
  struct sso {
    template <typename T, size_t StackSize = 0>
    using handle = ::nda::mem::handle_sso<T, SSO_Size>;
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
