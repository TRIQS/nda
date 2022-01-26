// Copyright (c) 2019-2021 Simons Foundation
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
#include "./handle.hpp"

namespace nda {

  // ----- Policy classes -----

#ifdef NDA_TEST_DEFAULT_ALLOC_MBUCKET // FOR TESTS ONLY: Run all with bucket allocator
  template <typename Allocator = mem::segregator<8 * NDA_TEST_DEFAULT_ALLOC_MBUCKET, mem::multi_bucket<8 * NDA_TEST_DEFAULT_ALLOC_MBUCKET>,
                                                      mem::mallocator>>
#else // Normal case
  template <typename Allocator>
#endif
  struct heap_basic {

    template <typename T>
#ifdef NDA_TEST_DEFAULT_ALLOC_SSO // FOR TESTS ONLY: Run all with sso
    using handle = std::conditional_t<std::is_copy_constructible_v<T>, mem::handle_sso<T, NDA_TEST_DEFAULT_ALLOC_SSO>,
                                      mem::handle_heap<T, Allocator>>;
#else
    using handle = mem::handle_heap<T, Allocator>;
#endif

#ifdef _OPENMP
    static_assert(false and !std::is_void_v<Allocator>, "Custom Allocators are not available in OpenMP");
#endif
  };

  template <mem::AddressSpace AdrSp = mem::Host>
  using heap = heap_basic<mem::mallocator<AdrSp>>;

  template <size_t Size>
  struct sso {
    template <typename T>
    using handle = mem::handle_sso<T, Size>;
  };

  template <size_t Size>
  struct stack {
    template <typename T>
    using handle = mem::handle_stack<T, Size>;
  };

  struct shared {
    template <typename T>
    using handle = mem::handle_shared<T>;
  };

  template <mem::AddressSpace AdrSp = mem::Host>
  struct borrowed {
    template <typename T>
    using handle = mem::handle_borrowed<T, AdrSp>;
  };

} // namespace nda
