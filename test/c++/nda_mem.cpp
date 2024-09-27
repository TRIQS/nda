// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2023 Simons Foundation
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
// Authors: Thomas Hahn

#define NDA_DEBUG_LEAK_CHECK

#include "./test_common.hpp"

#include <nda/mem.hpp>

#include <bitset>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

using namespace nda;

// Test malloc, memcpy, and memset.
template <mem::AddressSpace AdrSp1, mem::AddressSpace AdrSp2>
void check_malloc_memcpy_memset() {
  size_t size = 10;
  int val     = 5;

  // allocate memory
  auto *ptr1 = mem::malloc<AdrSp1>(size);
  auto *ptr2 = mem::malloc<AdrSp2>(size);
  auto *ptr3 = mem::malloc<mem::Host>(size);

  // set the memory
  mem::memset<AdrSp1>(ptr1, val, size);

  // copy the memory
  mem::memcpy<AdrSp2, AdrSp1>(ptr2, ptr1, size);
  mem::memcpy<mem::Host, AdrSp2>(ptr3, ptr2, size);

  // check the memory
  for (size_t i = 0; i < size; ++i) { EXPECT_EQ(static_cast<int>(static_cast<char *>(ptr3)[i]), val); }

  // free the memory
  mem::free<AdrSp1>(ptr1);
  mem::free<AdrSp2>(ptr2);
  mem::free<mem::Host>(ptr3);
}

// Test malloc, memcpy2D, and memset2D.
template <mem::AddressSpace AdrSp1, mem::AddressSpace AdrSp2>
void check_malloc_memcpy2D_memset2D() {
  size_t size = 100;
  int val     = 5;

  // allocate memory on host
  auto *ptr1 = mem::malloc<AdrSp1>(size);
  auto *ptr2 = mem::malloc<AdrSp2>(size);
  auto *ptr3 = mem::malloc<mem::Host>(size);

  // zero out both memories
  mem::memset<AdrSp1>(ptr1, 0, size);
  mem::memset<AdrSp2>(ptr2, 0, size);

  // set a 5x5 matrix to val
  mem::memset2D<AdrSp1>(ptr1, 10, val, 5, 5);

  // copy the memory
  mem::memcpy2D<AdrSp2, AdrSp1>(ptr2, 10, ptr1, 10, 5, 5);
  mem::memcpy<mem::Host, AdrSp2>(ptr3, ptr2, size);

  // check the memory
  for (size_t i = 0; i < 10; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      auto idx = i * 10 + j;
      if (i < 5 && j < 5) {
        EXPECT_EQ(static_cast<int>(static_cast<char *>(ptr3)[idx]), val);
      } else {
        EXPECT_EQ(static_cast<int>(static_cast<char *>(ptr3)[idx]), 0);
      }
    }
  }

  // free the memory
  mem::free<AdrSp1>(ptr1);
  mem::free<AdrSp2>(ptr2);
  mem::free<mem::Host>(ptr3);
}

// Test basic functionalities of various handles.
template <mem::Handle H>
H check_handle() {
  using value_t       = typename H::value_type;
  constexpr auto size = 10;

  // create handle
  H handle(size);

  // set memory in handle
  for (int i = 0; i < handle.size(); ++i) { handle[i] = static_cast<value_t>(i); }

  // check copy operations
  H copy1(handle), copy2{};
  copy2 = handle;
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif // __clang__
  handle = handle; // NOLINT (we want to check self assignment)
#ifdef __clang__
#pragma GCC diagnostic pop
#endif // __clang__
  EXPECT_EQ(handle.size(), size);
  EXPECT_EQ(handle.size(), copy1.size());
  EXPECT_EQ(handle.size(), copy2.size());
  for (int i = 0; i < handle.size(); ++i) {
    EXPECT_EQ(handle[i], static_cast<value_t>(i));
    EXPECT_EQ(copy1[i], static_cast<value_t>(i));
    EXPECT_EQ(copy2[i], static_cast<value_t>(i));
  }

  // check move operations
  H move1(std::move(copy2)), move2{};
  move2 = std::move(copy1);
  EXPECT_EQ(handle.size(), move1.size());
  EXPECT_EQ(handle.size(), move2.size());
  std::swap(handle, handle); // check self swap
  for (int i = 0; i < handle.size(); ++i) {
    EXPECT_EQ(handle[i], static_cast<value_t>(i));
    EXPECT_EQ(move1[i], static_cast<value_t>(i));
    EXPECT_EQ(move2[i], static_cast<value_t>(i));
  }

  // check self move assignment (see https://stackoverflow.com/questions/9322174/move-assignment-operator-and-if-this-rhs)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
  move1 = std::move(move1);
#pragma GCC diagnostic pop

  return handle;
}

// A non-trivial struct for testing.
struct foo {
  foo(int x = 100) : x(x) {}
  foo(const foo &f) : x(f.x) {}  // NOLINT (we want non-trivial copy constructor)
  foo &operator=(const foo &f) { // NOLINT (we want non-trivial copy assignment)
    x = f.x;
    return *this;
  }
  foo(foo &&f) noexcept : x(f.x) {}
  foo &operator=(foo &&f) noexcept {
    x = f.x;
    return *this;
  }
  bool operator==(const foo &f) const { return x == f.x; }
  int x;
};

// A struct that counts the number of current instances.
struct bar {
  static inline int count = 0; // NOLINT (we don't want it to be const)
  bar() { ++count; };
  ~bar() { --count; };
};

TEST(NDA, MemoryAddressSpaces) {
  using namespace mem;

  // test combinations of address spaces
  static_assert(combine<None, None> == None);
  static_assert(combine<Host, Host> == Host);
  static_assert(combine<None, Host> == Host);
  static_assert(combine<Host, None> == Host);

  static_assert(combine<Device, Device> == Device);
  static_assert(combine<None, Device> == Device);
  static_assert(combine<Device, None> == Device);

  static_assert(combine<Device, Unified> == Unified);
  static_assert(combine<Unified, Device> == Unified);
  static_assert(combine<Host, Unified> == Unified);
  static_assert(combine<Unified, Host> == Unified);
}

TEST(NDA, MemoryMallocMemcpyMemset) {
  // host to host
  check_malloc_memcpy_memset<mem::Host, mem::Host>();
  check_malloc_memcpy2D_memset2D<mem::Host, mem::Host>();

#ifdef NDA_HAVE_CUDA
  {
    // host to device
    check_malloc_memcpy_memset<mem::Host, mem::Device>();
    check_malloc_memcpy2D_memset2D<mem::Host, mem::Device>();

    // device to host
    check_malloc_memcpy_memset<mem::Device, mem::Host>();
    check_malloc_memcpy2D_memset2D<mem::Device, mem::Host>();

    // device to device
    check_malloc_memcpy_memset<mem::Device, mem::Device>();
    check_malloc_memcpy2D_memset2D<mem::Device, mem::Device>();
  }
#endif
}

TEST(NDA, MemoryMallocator) {
  const auto size = 100;

  // host
  auto allo1 = mem::mallocator<mem::Host>();
  auto mb1   = allo1.allocate_zero(size);
  EXPECT_EQ(mb1.s, size);
  for (int i = 0; i < size; ++i) EXPECT_EQ(mb1.ptr[i], 0);
  allo1.deallocate(mb1);

  // device
#ifdef NDA_HAVE_CUDA
  {
    auto allo2 = mem::mallocator<mem::Device>();
    auto mb2   = allo2.allocate_zero(size);
    EXPECT_EQ(mb2.s, size);
    auto mb3 = allo1.allocate(size);
    mem::memcpy<mem::Host, mem::Device>(mb3.ptr, mb2.ptr, size);
    for (int i = 0; i < size; ++i) EXPECT_EQ(mb3.ptr[i], 0);
    allo2.deallocate(mb2);
    allo1.deallocate(mb3);
  }
#endif
}

TEST(NDA, MemoryBucketAllocator) {
  // only on host
  constexpr auto chunksize = 8;
  constexpr auto size      = 6;
  std::vector<mem::blk_t> mbs(64);

  // empty bucket allocator
  auto allo = mem::bucket<chunksize>();
  EXPECT_TRUE(allo.empty());
  std::cout << "Empty bucket: " << std::bitset<64>(allo.mask()) << std::endl;

  // fill the bucket
  for (int i = 0; i < 64; ++i) {
    mbs[i] = allo.allocate(size);
    mem::memset<mem::Host>(static_cast<void *>(mbs[i].ptr), i, size);
    EXPECT_EQ(mbs[i].s, size);
  }
  std::cout << "Full bucket: " << std::bitset<64>(allo.mask()) << std::endl;

  // check the bucket
  EXPECT_TRUE(allo.is_full());
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(allo.owns(mbs[i]));
    for (int j = 0; j < size; ++j) { EXPECT_EQ(static_cast<char *>(mbs[i].ptr)[j], i); }
  }

  // free some chunk in the bucket
  const auto n = 20;
  allo.deallocate(mbs[n]);
  EXPECT_FALSE(allo.is_full());
  std::cout << "After deallocation: " << std::bitset<64>(allo.mask()) << std::endl;
  auto mb20 = allo.allocate_zero(size);
  EXPECT_EQ(mb20.ptr + chunksize, mbs[n + 1].ptr);
  std::cout << "After reallocation: " << std::bitset<64>(allo.mask()) << std::endl;

  // some other allocations
  auto mb_outside = mem::mallocator<mem::Host>::allocate(chunksize);
  EXPECT_FALSE(allo.owns(mb_outside));
  mem::mallocator<mem::Host>::deallocate(mb_outside);
}

TEST(NDA, MemoryMultiBucketAllocator) {
  // only on host
  constexpr auto chunksize = 8;
  std::vector<mem::blk_t> bucket1(64), bucket2(64);

  // empty multi-bucket allocator
  auto allo = mem::multi_bucket<chunksize>();
  EXPECT_TRUE(allo.empty());
  EXPECT_EQ(allo.buckets().size(), 1);

  // fill 1st bucket
  for (int i = 0; i < 64; ++i) { bucket1[i] = allo.allocate_zero(chunksize); }
  EXPECT_FALSE(allo.empty());
  EXPECT_EQ(allo.buckets().size(), 1);
  EXPECT_TRUE(allo.buckets()[0].is_full());

  // start 2nd bucket
  bucket2[0] = allo.allocate_zero(chunksize);
  EXPECT_EQ(allo.buckets().size(), 2);
  EXPECT_TRUE(allo.owns(bucket2[0]));
  // The new bucket may be first inside allo.buckets(),
  // as it respects memory ordering. Let's get the indeces
  std::size_t first_bucket_idx = allo.buckets()[1].owns(bucket1[0]);
  std::size_t second_bucket_idx = allo.buckets()[1].owns(bucket2[0]);
  EXPECT_TRUE(allo.buckets()[first_bucket_idx].is_full());
  EXPECT_FALSE(allo.buckets()[first_bucket_idx].owns(bucket2[0]));
  EXPECT_TRUE(allo.buckets()[second_bucket_idx].owns(bucket2[0]));

  // deallocate and reallocate in the 1st bucket
  allo.deallocate(bucket1[20]);
  EXPECT_FALSE(allo.buckets()[first_bucket_idx].is_full());
  auto mb_realloc = allo.allocate_zero(chunksize);
  EXPECT_TRUE(allo.buckets()[first_bucket_idx].owns(mb_realloc));
  EXPECT_TRUE(allo.buckets()[first_bucket_idx].is_full());

  // erase 2nd bucket
  allo.deallocate(bucket2[0]);
  EXPECT_EQ(allo.buckets().size(), 1);

  // some other allocations
  auto mb_outside = mem::mallocator<mem::Host>::allocate(chunksize);
  EXPECT_FALSE(allo.owns(mb_outside));
  mem::mallocator<mem::Host>::deallocate(mb_outside);
}

TEST(NDA, MemoryLeakCheckAllocator) {
  // test only on host
  auto allo = mem::leak_check<mem::mallocator<mem::Host>>();

  // check empty allocator
  EXPECT_TRUE(allo.empty());
  EXPECT_EQ(allo.get_memory_used(), 0);

  // allocate/deallocate and check used memory
  auto mb10 = allo.allocate(10);
  EXPECT_FALSE(allo.empty());
  EXPECT_EQ(allo.get_memory_used(), 10);
  auto mb20 = allo.allocate(20);
  EXPECT_EQ(allo.get_memory_used(), 30);
  allo.deallocate(mb10);
  EXPECT_EQ(allo.get_memory_used(), 20);
  allo.deallocate(mb20);
  EXPECT_EQ(allo.get_memory_used(), 0);
  EXPECT_TRUE(allo.empty());
}

TEST(NDA, MemoryStatsAllocator) {
  // test only on host
  auto allo = mem::stats<mem::mallocator<mem::Host>>();
  std::vector<mem::blk_t> mbs(20);

  // allocate and check stats
  for (auto i = 0ull; i < mbs.size(); ++i) {
    mbs[i] = allo.allocate(1ull << i);
    EXPECT_EQ(allo.histogram()[63 - i], 1);
  }

  // check stats
  allo.print_histogram(std::cout);

  // deallocate
  for (auto mb : mbs) { allo.deallocate(mb); }
}

TEST(NDA, MemoryHandleHeap) {
  using host_trivial_t = mem::handle_heap<int>;
  using host_foo_t     = mem::handle_heap<foo>;

  // test on host
  auto host_trivial = check_handle<host_trivial_t>();
  auto host_foo     = check_handle<host_foo_t>();

  // construct from different handle
  mem::handle_stack<foo, 5> stack_foo{};
  for (int i = 0; i < stack_foo.size(); ++i) { stack_foo[i] = foo(i); }
  auto host_foo2 = host_foo_t(stack_foo);
  EXPECT_EQ(stack_foo.size(), host_foo2.size());
  for (int i = 0; i < host_foo2.size(); ++i) { EXPECT_EQ(host_foo2[i], stack_foo[i]); }

  // test on device
#ifdef NDA_HAVE_CUDA
  {
    using device_trivial_t = mem::handle_heap<int, mem::mallocator<mem::Device>>;

    // construct on the device
    const auto size     = 5;
    auto device_trivial = device_trivial_t(size);
    EXPECT_EQ(device_trivial.size(), size);

    // assign from host
    device_trivial = host_trivial;

    // copy back to host and check
    auto host_trivial2 = host_trivial_t(device_trivial);
    EXPECT_EQ(host_trivial2.size(), host_trivial.size());
    for (int i = 0; i < host_trivial2.size(); ++i) { EXPECT_EQ(host_trivial2[i], host_trivial[i]); }
  }
#endif

  // check correct construction and destruction of non-trivial objects
  {
    mem::handle_heap<bar> h{10};
    EXPECT_EQ(bar::count, 10);
  }
  EXPECT_EQ(bar::count, 0);
}

TEST(NDA, MemoryHandleStack) {
  using handle_trivial_t = mem::handle_stack<int, 10>;
  using handle_foo_t     = mem::handle_stack<foo, 10>;

  // test on host
  check_handle<handle_trivial_t>();
  check_handle<handle_foo_t>();

  // check correct construction and destruction of non-trivial objects
  {
    mem::handle_stack<bar, 10> h{10};
    EXPECT_EQ(bar::count, 10);
  }
  EXPECT_EQ(bar::count, 0);
}

TEST(NDA, MemoryHandleSSO) {
  using handle_small_trivial_t = mem::handle_sso<int, 5>;
  using handle_small_foo_t     = mem::handle_sso<foo, 5>;
  using handle_big_trivial_t   = mem::handle_sso<int, 20>;
  using handle_big_foo_t       = mem::handle_sso<foo, 20>;

  // test on host
  check_handle<handle_small_trivial_t>();
  check_handle<handle_small_foo_t>();
  check_handle<handle_big_trivial_t>();
  check_handle<handle_big_foo_t>();

  // construct from different handle
  mem::handle_stack<foo, 10> stack_foo{};
  for (int i = 0; i < stack_foo.size(); ++i) { stack_foo[i] = foo(i); }
  auto small_foo = handle_small_foo_t(stack_foo);
  auto big_foo   = handle_big_foo_t(stack_foo);
  EXPECT_EQ(stack_foo.size(), small_foo.size());
  EXPECT_EQ(stack_foo.size(), big_foo.size());
  for (int i = 0; i < big_foo.size(); ++i) {
    EXPECT_EQ(small_foo[i], stack_foo[i]);
    EXPECT_EQ(big_foo[i], stack_foo[i]);
  }

  // check correct construction and destruction of non-trivial objects
  {
    mem::handle_sso<bar, 10> h{5};
    EXPECT_EQ(bar::count, 5);
  }
  EXPECT_EQ(bar::count, 0);

  {
    mem::handle_sso<bar, 10> h{20};
    EXPECT_EQ(bar::count, 20);
  }
  EXPECT_EQ(bar::count, 0);
}

TEST(NDA, MemoryHandleShared) {
  // test the reference counting
  mem::handle_heap<int> h{10};

  mem::handle_shared<int> s{h};
  EXPECT_EQ(s.refcount(), 2);

  s = mem::handle_shared<int>{h};
  EXPECT_EQ(s.refcount(), 2);

  mem::handle_shared<int> s2{h};
  s = s2;
  EXPECT_EQ(s.refcount(), 3);
}
