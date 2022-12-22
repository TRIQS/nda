// Copyright (c) 2018-2021 Simons Foundation
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

#include <cstddef>
#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <concepts>

#include "address_space.hpp"
#include "../macros.hpp"

#include "../mem/memcpy.hpp"
#include "../mem/malloc.hpp"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define NDA_USE_ASAN
#endif
#endif

using namespace std::literals;

namespace nda::mem {

  /// The memory block with its size
  struct blk_t {
    char *ptr = nullptr;
    size_t s  = 0;
  };

  // -------------------------  Malloc allocator ----------------------------
  //
  // Allocates simply with malloc
  //
  template <AddressSpace AdrSp = Host>
  class mallocator {
    public:
    mallocator()                   = default;
    mallocator(mallocator const &) = delete;
    mallocator(mallocator &&)      = default;
    mallocator &operator=(mallocator const &) = delete;
    mallocator &operator=(mallocator &&) = default;

    static constexpr auto address_space = AdrSp;

    static blk_t allocate(size_t s) noexcept {
      return {(char *)malloc<AdrSp>(s), s}; // NOLINT
    }
    static blk_t allocate_zero(size_t s) noexcept {
      return {(char *)calloc<AdrSp>(s, sizeof(char)), s}; // NOLINT
    }

    static void deallocate(blk_t b) noexcept {
      free<AdrSp>((void*)b.ptr); // NOLINT
    }
  };

  // -------------------------  Bucket allocator ----------------------------
  //
  //
  template <int ChunkSize>
  class bucket {
    static constexpr int TotalChunkSize = 64 * ChunkSize;
    std::unique_ptr<char[]> _start      = std::make_unique<char[]>(TotalChunkSize); // NOLINT
    char *p                             = _start.get();
    uint64_t flags                      = uint64_t(-1);

    public:
    static constexpr auto address_space = Host;

#ifdef NDA_USE_ASAN
    bucket() { __asan_poison_memory_region(p, TotalChunkSize); }
    ~bucket() { __asan_unpoison_memory_region(p, TotalChunkSize); }
#else
    bucket() = default;
#endif
    bucket(bucket const &) = delete;
    bucket(bucket &&)      = default;
    bucket &operator=(bucket const &) = delete;
    bucket &operator=(bucket &&) = default;

    blk_t allocate(size_t s) noexcept {
      // FIXME not here ! in the handle
      //auto n = round_to_align(s);
      //if (n > ChunkSize) std::abort();
      if (flags == 0) std::abort();
      int pos = __builtin_ctzll(flags);
      flags &= ~(1ull << pos);
      blk_t b{p + pos * ChunkSize, s};
#ifdef NDA_USE_ASAN
      __asan_unpoison_memory_region(b.ptr, ChunkSize);
#endif
      return b;
    }

    blk_t allocate_zero(size_t s) noexcept {
      auto blk = allocate(s);
      std::memset(blk.ptr, 0, s);
      return blk;
    }

    void deallocate(blk_t b) noexcept {
#ifdef NDA_USE_ASAN
      __asan_poison_memory_region(b.ptr, ChunkSize);
#endif
      int pos = (b.ptr - p) / ChunkSize;
      flags |= (1ull << pos);
    }

    [[nodiscard]] bool is_full() const noexcept { return flags == 0; }
    [[nodiscard]] bool empty() const noexcept { return flags == uint64_t(-1); }

    [[nodiscard]] const char *data() const noexcept { return p; }

    [[nodiscard]] bool owns(blk_t b) const noexcept { return b.ptr >= p and b.ptr < p + TotalChunkSize; }
  };

  // -------------------------  Multiple bucket allocator ----------------------------
  //
  //
  template <int ChunkSize>
  class multi_bucket {

    using b_t = bucket<ChunkSize>;
    std::vector<b_t> bu_vec;                // an ordered vector of buckets
    typename std::vector<b_t>::iterator bu; // current bucket in use

    // find the next bucket with some space. Possibly allocating new ones.
    [[gnu::noinline]] void find_non_full_bucket() {
      bu = std::find_if(bu_vec.begin(), bu_vec.end(), [](auto const &b) { return !b.is_full(); });
      if (bu != bu_vec.end()) return;

      // insert a new bucket ordered. Position is defined by data (NB : which is NOT affected by the move)
      b_t b;
      auto insert_position = std::upper_bound(bu_vec.begin(), bu_vec.end(), b, [](auto const &B, auto const &B2) { return B.data() < B2.data(); });
      bu                   = bu_vec.insert(insert_position, std::move(b));

      //for (auto const &bb : bu_vec) TRIQS_PRINT((void *)bb.data());
      //TRIQS_PRINT("---------------");
    }

    public:
    static constexpr auto address_space = Host;

    multi_bucket() : bu_vec(1), bu(bu_vec.begin()) {}
    multi_bucket(multi_bucket const &) = delete;
    multi_bucket(multi_bucket &&)      = delete;
    multi_bucket &operator=(multi_bucket const &) = delete;
    multi_bucket &operator=(multi_bucket &&) = delete;

    blk_t allocate(size_t s) noexcept {
      //[[unlikely]]
      if ((bu == bu_vec.end()) or (bu->is_full())) find_non_full_bucket();
      return bu->allocate(s);
    }

    blk_t allocate_zero(size_t s) noexcept {
      auto blk = allocate(s);
      std::memset(blk.ptr, 0, s);
      return blk;
    }

    void deallocate(blk_t b) noexcept {
      //[[likely]]
      if (bu != bu_vec.end() and bu->owns(b)) {
        bu->deallocate(b);
        return;
      }
      bu = std::lower_bound(bu_vec.begin(), bu_vec.end(), b.ptr, [](auto const &B, auto p) { return B.data() <= p; });
      --bu;
      EXPECTS_WITH_MESSAGE((bu != bu_vec.end()), "Fatal Logic Error in allocator. Not in bucket. \n");
      EXPECTS_WITH_MESSAGE((bu->owns(b)), "Fatal Logic Error in allocator. \n");
      bu->deallocate(b);
      if (!bu->empty()) return;
      if (bu_vec.size() <= 1) return;
      bu_vec.erase(bu);
      bu = bu_vec.end();
    }

    //bool owns(blk_t b) const noexcept { return b.ptr >= d and b.ptr < d + Size; }
  };

  // -------------------------  segregator allocator ----------------------------
  //
  // Dispatch according to size to two allocators
  //
  template <size_t Threshold, Allocator A, Allocator B>
  class segregator {

    A small;
    B big;

    public:
    static_assert(A::address_space == B::address_space);
    static constexpr auto address_space = A::address_space;

    segregator()                   = default;
    segregator(segregator const &) = delete;
    segregator(segregator &&)      = default;
    segregator &operator=(segregator const &) = delete;
    segregator &operator=(segregator &&) = default;

    blk_t allocate(size_t s) noexcept { return s <= Threshold ? small.allocate(s) : big.allocate(s); }
    blk_t allocate_zero(size_t s) noexcept { return s <= Threshold ? small.allocate_zero(s) : big.allocate_zero(s); }

    void deallocate(blk_t b) noexcept { return b.s <= Threshold ? small.deallocate(b) : big.deallocate(b); }
    [[nodiscard]] bool owns(blk_t b) const noexcept { return small.owns(b) or big.owns(b); }
  };

  // -------------------------  dress allocator with leak_checking ----------------------------
  template <Allocator A>
  class leak_check : A {

    long memory_used = 0;

    public:
    static constexpr auto address_space = A::address_space;

    leak_check()                   = default;
    leak_check(leak_check const &) = delete;
    leak_check(leak_check &&)      = default;
    leak_check &operator=(leak_check const &) = delete;
    leak_check &operator=(leak_check &&) = default;

    ~leak_check() {
      if (!empty()) {
#ifndef NDEBUG
        std::cerr << "Allocator : MEMORY LEAK of " << memory_used << " bytes\n";
        std::abort();
#endif
      }
    }

    [[nodiscard]] bool empty() const { return (memory_used == 0); }

    blk_t allocate(size_t s) {
      blk_t b     = A::allocate(s);
      memory_used = memory_used + b.s;
      //      std::cerr<< "Allocating "<< b.s << "Total = "<< memory_used << "\n";
      return b;
    }

    blk_t allocate_zero(size_t s) {
      blk_t b     = A::allocate_zero(s);
      memory_used = memory_used + b.s;
      //      std::cerr<< "Allocating "<< b.s << "Total = "<< memory_used << "\n";
      return b;
    }

    void deallocate(blk_t b) noexcept {
      memory_used -= b.s;
      //    std::cerr<< "Deallocating "<< b.s << "Total = "<< memory_used << "\n";

      if (memory_used < 0) {
#ifndef NDEBUG
        std::cerr << "Allocator : memory_used <0 : " << memory_used << " b.s = " << b.s << " b.ptr = " << (void *)b.ptr;
        std::abort();
#endif
      }
      A::deallocate(b);
    }

    [[nodiscard]] bool owns(blk_t b) const noexcept { return A::owns(b); }

    [[nodiscard]] long get_memory_used() const noexcept { return memory_used; }
  };

  // ------------------------- gather statistics for a generic allocator ----------------------------
  template <Allocator A>
  class stats : A {

    std::vector<uint64_t> hist = std::vector<uint64_t>(65, 0);

    public:
    static constexpr auto address_space = A::address_space;

    ~stats() {
#ifndef NDEBUG
      std::cerr << "Allocation size histogram :\n";
      //auto weight = 1.0 / std::accumulate(hist.begin(), hist.end(), uint64_t(0));
      double lz = 65;
      for (auto c : hist) {
        std::cerr << "[2^" << lz << ", 2^" << lz - 1 << "]: " << c << "\n";
        --lz;
      }
#endif
    }
    stats()              = default;
    stats(stats const &) = delete;
    stats(stats &&)      = default;
    stats &operator=(stats const &) = delete;
    stats &operator=(stats &&) = default;

    blk_t allocate(uint64_t s) {
      ++hist[__builtin_clzl(s)];
      return A::allocate(s);
    }

    blk_t allocate_zero(uint64_t s) {
      ++hist[__builtin_clzl(s)];
      return A::allocate(s);
    }

    void deallocate(blk_t b) noexcept { A::deallocate(b); }

    [[nodiscard]] bool owns(blk_t b) const noexcept { return A::owns(b); }

    auto const &histogram() const noexcept { return hist; }
  };

} // namespace nda::mem
