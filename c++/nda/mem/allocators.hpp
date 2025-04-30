// Copyright (c) 2018--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides custom allocators for the nda library.
 */

#pragma once

#include "./address_space.hpp"
#include "./malloc.hpp"
#include "./memset.hpp"
#include "./memcpy.hpp"
#include "./fill.hpp"
#include "../macros.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define NDA_USE_ASAN
#endif
#endif

namespace nda::mem {

  /**
   * @addtogroup mem_allocators
   * @{
   */

  /// Memory block consisting of a pointer and its size.
  struct blk_t {
    /// Pointer to the memory block.
    char *ptr = nullptr;

    /// Size of the memory block in bytes.
    size_t s = 0;
  };

  /**
   * @brief Custom allocator that uses nda::mem::malloc to allocate memory.
   * @tparam AdrSp nda::mem::AddressSpace in which the memory is allocated.
   */
  template <AddressSpace AdrSp = Host>
  class mallocator {
    public:
    /// Default constructor.
    mallocator() = default;

    /// Deleted copy constructor.
    mallocator(mallocator const &) = delete;

    /// Default move constructor.
    mallocator(mallocator &&) = default;

    /// Deleted copy assignment operator.
    mallocator &operator=(mallocator const &) = delete;

    /// Default move assignment operator.
    mallocator &operator=(mallocator &&) = default;

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = AdrSp;

    /**
     * @brief Allocate memory using nda::mem::malloc.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    static blk_t allocate(size_t s) noexcept { return {(char *)malloc<AdrSp>(s), s}; }

    /**
     * @brief Allocate memory and set it to zero.
     *
     * @details The behavior depends on the address space:
     * - It uses std::calloc for `Host` nda::mem::AddressSpace.
     * - Otherwise it uses nda::mem::malloc and nda::mem::memset.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    static blk_t allocate_zero(size_t s) noexcept {
      if constexpr (AdrSp == mem::Host) {
        return {(char *)std::calloc(s, 1 /* byte */), s}; // NOLINT (C-style cast is fine here)
      } else {
        char *ptr = (char *)malloc<AdrSp>(s);
        memset<AdrSp>(ptr, 0, s);
        return {ptr, s};
      }
    }

    /**
     * @brief Deallocate memory using nda::mem::free.
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    static void deallocate(blk_t b) noexcept { free<AdrSp>((void *)b.ptr); }
  };

  /**
   * @brief Custom allocator that allocates a bucket of memory on the heap consisting of 64 chunks.
   *
   * @details The allocator keeps track of which chunks are free using a bitmask. Once all chunks have been allocated,
   * it will call std::abort on any further allocation requests.
   *
   * @note Only works with `Host` nda::mem::AddressSpace.
   *
   * @tparam ChunkSize Size of the chunks in bytes.
   */
  template <int ChunkSize>
  class bucket {
    // Unique pointer to handle memory allocation for the bucket.
    std::unique_ptr<char[]> _start = std::make_unique<char[]>(TotalChunkSize); // NOLINT (C-style array is fine here)

    // Pointer to the start of the bucket.
    char *p = _start.get();

    // Bitmask to keep track of which chunks are free.
    uint64_t flags = uint64_t(-1);

    public:
    /// Total size of the bucket in bytes.
    static constexpr int TotalChunkSize = 64 * ChunkSize;

    /// Only `Host` nda::mem::AddressSpace is supported for this allocator.
    static constexpr auto address_space = Host;

#ifdef NDA_USE_ASAN
    bucket() { __asan_poison_memory_region(p, TotalChunkSize); }
    ~bucket() { __asan_unpoison_memory_region(p, TotalChunkSize); }
#else
    /// Default constructor.
    bucket() = default;
#endif
    /// Deleted copy constructor.
    bucket(bucket const &) = delete;

    /// Default move constructor.
    bucket(bucket &&) = default;

    /// Deleted copy assignment operator.
    bucket &operator=(bucket const &) = delete;

    /// Default move assignment operator.
    bucket &operator=(bucket &&) = default;

    /**
     * @brief Allocate a chunk of memory in the bucket and update the bitmask.
     *
     * @param s Size in bytes of the returned memory block (has to be < `ChunkSize`).
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate(size_t s) noexcept {
      // check the size and if there is a free chunk, otherwise abort
      if (s > ChunkSize) std::abort();
      if (flags == 0) std::abort();

      // find the first free chunk
      int pos = __builtin_ctzll(flags);

      // update the bitmask and return the memory block
      flags &= ~(1ull << pos);
      blk_t b{p + static_cast<ptrdiff_t>(pos * ChunkSize), s};
#ifdef NDA_USE_ASAN
      __asan_unpoison_memory_region(b.ptr, ChunkSize);
#endif
      return b;
    }

    /**
     * @brief Allocate a chunk of memory in the bucket, set it to zero and update the bitmask.
     *
     * @param s Size in bytes of the returned memory block (has to be < `ChunkSize`).
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate_zero(size_t s) noexcept {
      auto blk = allocate(s);
      std::memset(blk.ptr, 0, s);
      return blk;
    }

    /**
     * @brief Deallocate a chunk of memory from the bucket by simply resetting the bitmask.
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b) noexcept {
#ifdef NDA_USE_ASAN
      __asan_poison_memory_region(b.ptr, ChunkSize);
#endif
      int pos = (b.ptr - p) / ChunkSize;
      flags |= (1ull << pos);
    }

    /**
     * @brief Check if the bucket is full.
     * @return True if there are no free chunks left, i.e. the bitmask is all zeros.
     */
    [[nodiscard]] bool is_full() const noexcept { return flags == 0; }

    /**
     * @brief Check if the bucket is empty.
     * @return True if all chunks are free, i.e. the bitmask is all ones.
     */
    [[nodiscard]] bool empty() const noexcept { return flags == uint64_t(-1); }

    /**
     * @brief Get a pointer to the start of the bucket.
     * @return Pointer to the chunk with the lowest memory address.
     */
    [[nodiscard]] const char *data() const noexcept { return p; }

    /**
     * @brief Get the bitmask of the bucket.
     * @return Bitmask in the form of a `uint64_t`.
     */
    [[nodiscard]] auto mask() const noexcept { return flags; }

    /**
     * @brief Check if a given nda::mem::blk_t memory block is owned by the bucket.
     *
     * @param b nda::mem::blk_t memory block.
     * @return True if the memory block is owned by the bucket.
     */
    [[nodiscard]] bool owns(blk_t b) const noexcept { return b.ptr >= p and b.ptr < p + TotalChunkSize; }
  };

  /**
   * @brief Custom allocator that uses multiple nda::mem::bucket allocators.
   *
   * @details It uses a std::vector of bucket allocators. When all buckets in the vector are full, it simply adds a new
   * one at the end.
   *
   * @note Only works with `Host` nda::mem::AddressSpace.
   *
   * @tparam ChunkSize Size of the chunks in bytes.
   */
  template <int ChunkSize>
  class multi_bucket {
    // Alias for the bucket allocator type.
    using b_t = bucket<ChunkSize>;

    // Vector of nda::mem::bucket allocators (ordered in memory).
    std::vector<b_t> bu_vec;

    // Iterator to the current bucket in use.
    typename std::vector<b_t>::iterator bu;

    // Find a bucket with a free chunk, otherwise create a new one and insert it at the correct position.
    [[gnu::noinline]] void find_non_full_bucket() {
      bu = std::find_if(bu_vec.begin(), bu_vec.end(), [](auto const &b) { return !b.is_full(); });
      if (bu != bu_vec.end()) return;
      b_t b;
      auto pos = std::upper_bound(bu_vec.begin(), bu_vec.end(), b, [](auto const &b1, auto const &b2) { return b1.data() < b2.data(); });
      bu       = bu_vec.insert(pos, std::move(b));
    }

    public:
    /// Only `Host` nda::mem::AddressSpace is supported for this allocator.
    static constexpr auto address_space = Host;

    /// Default constructor.
    multi_bucket() : bu_vec(1), bu(bu_vec.begin()) {}

    /// Deleted copy constructor.
    multi_bucket(multi_bucket const &) = delete;

    /// Default move constructor.
    multi_bucket(multi_bucket &&) = default;

    /// Deleted copy assignment operator.
    multi_bucket &operator=(multi_bucket const &) = delete;

    /// Default move assignment operator.
    multi_bucket &operator=(multi_bucket &&) = default;

    /**
     * @brief Allocate a chunk of memory in the current bucket or find a new one if the current one is full.
     *
     * @param s Size in bytes of the returned memory block (has to be < `ChunkSize`).
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate(size_t s) noexcept {
      if ((bu == bu_vec.end()) or (bu->is_full())) find_non_full_bucket();
      return bu->allocate(s);
    }

    /**
     * @brief Allocate a chunk of memory in the current bucket or find a new one if the current one is full and set it
     * to zero.
     *
     * @param s Size in bytes of the returned memory block (has to be < `ChunkSize`).
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate_zero(size_t s) noexcept {
      auto blk = allocate(s);
      std::memset(blk.ptr, 0, s);
      return blk;
    }

    /**
     * @brief Deallocate a chunk of memory from the bucket to which it belongs.
     *
     * @details If the bucket is empty after deallocation and it is not the only one, it is removed from the vector of
     * buckets.
     *
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b) noexcept {
      // try the current bucket first
      if (bu != bu_vec.end() and bu->owns(b)) {
        bu->deallocate(b);
        return;
      }

      // otherwise, find the owning bucket in the vector and deallocate
      bu = std::lower_bound(bu_vec.begin(), bu_vec.end(), b.ptr, [](auto const &b1, auto p) { return b1.data() <= p; });
      --bu;
      EXPECTS_WITH_MESSAGE((bu != bu_vec.end()), "Error in nda::mem::multi_bucket::deallocate: Owning bucket not found");
      EXPECTS_WITH_MESSAGE((bu->owns(b)), "Error in nda::mem::multi_bucket::deallocate: Owning bucket not found");
      bu->deallocate(b);

      // remove bucket the current bucket if it is empty and not the only one
      if (!bu->empty()) return;
      if (bu_vec.size() <= 1) return;
      bu_vec.erase(bu);
      bu = bu_vec.end();
    }

    /**
     * @brief Check if the current allocator is empty.
     * @return True if there is only one bucket in the vector and it is empty.
     */
    [[nodiscard]] bool empty() const noexcept { return bu_vec.size() == 1 && bu_vec[0].empty(); }

    /**
     * @brief Get the bucket vector.
     * @return std::vector with all the bucket allocators currently in use.
     */
    [[nodiscard]] auto const &buckets() const noexcept { return bu_vec; }

    /**
     * @brief Check if a given nda::mem::blk_t memory block is owned by allocator.
     *
     * @param b nda::mem::blk_t memory block.
     * @return True if the memory block is owned by one of the buckets.
     */
    [[nodiscard]] bool owns(blk_t b) const noexcept {
      bool res = false;
      for (const auto &mb : bu_vec) { res = res || mb.owns(b); }
      return res;
    }
  };

  /**
   * @brief Custom allocator that dispatches memory allocation to one of two allocators based on the size of the memory
   * block to be allocated.
   *
   * @note Only works if both allocators have the same nda::mem::AddressSpace.
   *
   * @tparam Threshold Size in bytes that determines which allocator to use.
   * @tparam A nda::mem::Allocator for small memory blocks.
   * @tparam B nda::mem::Allocator for big memory blocks.
   */
  template <size_t Threshold, Allocator A, Allocator B>
  class segregator {
    // Allocator for small memory blocks.
    A small;

    // Allocator for big memory blocks.
    B big;

    public:
    static_assert(A::address_space == B::address_space);

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = A::address_space;

    /// Default constructor.
    segregator() = default;

    /// Deleted copy constructor.
    segregator(segregator const &) = delete;

    /// Default move constructor.
    segregator(segregator &&) = default;

    /// Deleted copy assignment operator.
    segregator &operator=(segregator const &) = delete;

    /// Default move assignment operator.
    segregator &operator=(segregator &&) = default;

    /**
     * @brief Allocate memory using the small allocator if the size is less than or equal to the `Threshold`, otherwise
     * use the big allocator.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate(size_t s) noexcept { return s <= Threshold ? small.allocate(s) : big.allocate(s); }

    /**
     * @brief Allocate memory and set the memory to zero using the small allocator if the size is less than or equal to
     * the `Threshold`, otherwise use the big allocator.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate_zero(size_t s) noexcept { return s <= Threshold ? small.allocate_zero(s) : big.allocate_zero(s); }

    /**
     * @brief Deallocate memory using the small allocator if the size is less than or equal to the `Threshold`,
     * otherwise use the big allocator.
     *
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b) noexcept { return b.s <= Threshold ? small.deallocate(b) : big.deallocate(b); }

    /**
     * @brief Check if a given nda::mem::blk_t memory block is owned by the allocator.
     *
     * @param b nda::mem::blk_t memory block.
     * @return True if one of the two allocators owns the memory block.
     */
    [[nodiscard]] bool owns(blk_t b) const noexcept { return small.owns(b) or big.owns(b); }
  };

  /**
   * @brief Wrap an allocator to check for memory leaks.
   *
   * @details It simply keeps track of the memory currently being used by the allocator, i.e. the total memory allocated
   * minus the total memory deallocated, which should never be smaller than zero and should be exactly zero when the
   * allocator is destroyed.
   *
   * @tparam A nda::mem::Allocator type to wrap.
   */
  template <Allocator A>
  class leak_check : A {
    // Total memory used by the allocator.
    long memory_used = 0;

    public:
    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = A::address_space;

    /// Default constructor.
    leak_check() = default;

    /// Deleted copy constructor.
    leak_check(leak_check const &) = delete;

    /// Default move constructor.
    leak_check(leak_check &&) = default;

    /// Deleted copy assignment operator.
    leak_check &operator=(leak_check const &) = delete;

    /// Default move assignment operator.
    leak_check &operator=(leak_check &&) = default;

    /**
     * @brief Destructor that checks for memory leaks.
     * @details In debug mode, it aborts the program if there is a memory leak.
     */
    ~leak_check() {
      if (!empty()) {
#ifndef NDEBUG
        std::cerr << "Memory leak in allocator: " << memory_used << " bytes leaked\n";
        std::abort();
#endif
      }
    }

    /**
     * @brief Allocate memory and update the total memory used.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate(size_t s) {
      blk_t b = A::allocate(s);
      memory_used += b.s;
      return b;
    }

    /**
     * @brief Allocate memory, set it to zero and update the total memory used.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate_zero(size_t s) {
      blk_t b = A::allocate_zero(s);
      memory_used += b.s;
      return b;
    }

    /**
     * @brief Deallocate memory and update the total memory used.
     * @details In debug mode, it aborts the program if the total memory used is smaller than zero.
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b) noexcept {
      memory_used -= b.s;
      if (memory_used < 0) {
#ifndef NDEBUG
        std::cerr << "Memory used by allocator < 0: Memory block to be deleted: b.s = " << b.s << ", b.ptr = " << (void *)b.ptr << "\n";
        std::abort();
#endif
      }
      A::deallocate(b);
    }

    /**
     * @brief Check if the base allocator is empty.
     * @return True if no memory is currently being used.
     */
    [[nodiscard]] bool empty() const { return (memory_used == 0); }

    /**
     * @brief Check if a given nda::mem::blk_t memory block is owned by the base allocator.
     *
     * @param b nda::mem::blk_t memory block.
     * @return True if the base allocator owns the memory block.
     */
    [[nodiscard]] bool owns(blk_t b) const noexcept { return A::owns(b); }

    /**
     * @brief Get the total memory used by the base allocator.
     * @return The size of the memory which has been allocated and not yet deallocated.
     */
    [[nodiscard]] long get_memory_used() const noexcept { return memory_used; }
  };

  /**
   * @brief Wrap an allocator to gather statistics about memory allocation.
   *
   * @details It gathers a histogram of the different allocation sizes. The histogram is a std::vector of size 65, where
   * element \f$ i \in \{0,...,63\} \f$ contains the number of allocations with a size in the range
   * \f$ [2^{64-i-1}, 2^{64-i}) \f$ and the last element contains the number of allocations of size zero.
   *
   * @tparam A nda::mem::Allocator type to wrap.
   */
  template <Allocator A>
  class stats : A {
    // Histogram of the allocation sizes.
    std::vector<uint64_t> hist = std::vector<uint64_t>(65, 0);

    public:
    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = A::address_space;

    /// Default constructor.
    stats() = default;

    /// Deleted copy constructor.
    stats(stats const &) = delete;

    /// Default move constructor.
    stats(stats &&) = default;

    /// Deleted copy assignment operator.
    stats &operator=(stats const &) = delete;

    /// Default move assignment operator.
    stats &operator=(stats &&) = default;

    /// Destructor that outputs the statistics about the memory allocation in debug mode.
    ~stats() {
#ifndef NDEBUG
      print_histogram(std::cerr);
#endif
    }

    /**
     * @brief Allocate memory and update the histogram.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate(uint64_t s) {
      // __builtin_clzl returns the number of leading zeros
      ++hist[__builtin_clzl(s)];
      return A::allocate(s);
    }

    /**
     * @brief Allocate memory, set it to zero and update the histogram.
     *
     * @param s Size in bytes of the memory to allocate.
     * @return nda::mem::blk_t memory block.
     */
    blk_t allocate_zero(uint64_t s) {
      // __builtin_clzl returns the number of leading zeros
      ++hist[__builtin_clzl(s)];
      return A::allocate_zero(s);
    }

    /**
     * @brief Deallocate memory.
     * @param b nda::mem::blk_t memory block to deallocate.
     */
    void deallocate(blk_t b) noexcept { A::deallocate(b); }

    /**
     * @brief Check if a given nda::mem::blk_t memory block is owned by the base allocator.
     *
     * @param b nda::mem::blk_t memory block.
     * @return True if the base allocator owns the memory block.
     */
    [[nodiscard]] bool owns(blk_t b) const noexcept { return A::owns(b); }

    /**
     * @brief Get the histogram of the allocation sizes.
     * @return std::vector of size 65 with the number of allocations in each size range.
     */
    [[nodiscard]] auto const &histogram() const noexcept { return hist; }

    /**
     * @brief Print the histogram to a std::ostream.
     * @param os std::ostream object to print to.
     */
    void print_histogram(std::ostream &os) const {
      os << "Allocation size histogram :\n";
      os << "[0, 2^0): " << hist.back() << "\n";
      for (int i = 0; i < 64; ++i) { os << "[2^" << i << ", 2^" << i + 1 << "): " << hist[63 - i] << "\n"; }
    }
  };

  /** @} */

} // namespace nda::mem
