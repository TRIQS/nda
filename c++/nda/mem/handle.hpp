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
#include <limits>
#include <complex>
#include <type_traits>
#include <cstring>
#include "./allocators.hpp"

namespace nda::mem {

  // -------  Handle and OwningHandle Concept ----------

  /// Concept of a handle on a block of memory
  template <typename H, typename T = typename H::value_type>
  concept Handle = requires(H const &h) {
    { typename H::value_type{} } -> std::same_as<T>;
    { h.is_null() } noexcept -> std::same_as<bool>;
    { h.data() } noexcept -> std::same_as<T *>;
    { H::address_space } -> std::same_as<AddressSpace const &>;
  };

  /// Concept of a handle that owns a block of memory
  template <typename H, typename T = typename H::value_type>
  concept OwningHandle = Handle<H, T> and requires(H const &h) {
    requires(not std::is_const_v<typename H::value_type>);
    { h.size() } noexcept -> std::same_as<long>;
  };

  //
  static constexpr bool init_dcmplx = true; // initialize dcomplex to 0 globally

  // -------------- is_complex ----------------

  template <typename T>
  static constexpr bool is_complex_v = false;
  template <typename T>
  static constexpr bool is_complex_v<std::complex<T>> = true;

  // -------------- Utilities ---------------------------

  // To have aligned objects, use aligner<T, alignment> instead of T in constructor and get
  template <typename T, int Al>
  struct alignas(Al) aligner {
    T x;
    T &get() noexcept { return x; }
    T const &get() const noexcept { return x; }
  };

  // ------------------  tag and var for constructors -------------------------------------

  struct do_not_initialize_t {};
  inline static constexpr do_not_initialize_t do_not_initialize{};

  struct init_zero_t {};
  inline static constexpr init_zero_t init_zero{};

  // -------------- handle ---------------------------

  // The block of memory for the arrays
  // Heap (owns the memory on the heap)
  // Shared (shared memory ownership)
  // Borrowed (no memory ownership)
  // Stack  (on stack)
  // Forward declaration
  template <typename T, AddressSpace AdrSp = Host>
  struct handle_shared;

  // ------------------  HEAP -------------------------------------

  template <typename T, Allocator alloc_t = mallocator<>>
  struct handle_heap {
    private:
    T *_data     = nullptr; // Pointer to the start of the memory block
    size_t _size = 0;       // Size of the memory block. Invariant: size > 0 iif data != 0

#ifndef NDA_DEBUG_LEAK_CHECK
    static inline alloc_t allocator;
#else
    static inline leak_check<alloc_t> allocator;
#endif

    // In case we need to share the memory
    mutable std::shared_ptr<void> sptr;

    using blk_T_t = std::pair<T *, size_t>;

    // code for destructor
    static void destruct(blk_T_t b) noexcept {

      static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");
      auto [data, size] = b;

      if (data == nullptr) return;

      // If needed, call the T destructors
      if constexpr (!std::is_trivial_v<T>) {
        for (size_t i = 0; i < size; ++i) data[i].~T();
      }

      // Deallocate the memory block
      allocator.deallocate({(char *)data, size * sizeof(T)});
    }

    // a deleter for the data in the sptr
    static void deleter(void *p) noexcept { destruct(*((blk_T_t *)p)); }

    public:
    std::shared_ptr<void> get_sptr() const {
      if (not sptr) sptr.reset(new blk_T_t{_data, _size}, deleter);
      return sptr;
    }

    using value_type = T;
    static constexpr auto address_space = alloc_t::address_space;

    ~handle_heap() noexcept {
      // if the data is not in the shared_ptr, we delete it, otherwise the shared_ptr will take care of it
      if (not sptr and not(is_null())) destruct({_data, _size});
    }

    handle_heap() = default;

    handle_heap(handle_heap &&x) noexcept {
      _data   = x._data;
      _size   = x._size;
      sptr    = std::move(x.sptr);
      x._data = nullptr;
      x._size = 0;
    }

    handle_heap &operator=(handle_heap &&x) noexcept {
      if (not sptr and not(is_null())) destruct({_data, _size});
      _data   = x._data;
      _size   = x._size;
      sptr    = std::move(x.sptr);
      x._data = nullptr;
      x._size = 0;
      return *this;
    }

    handle_heap &operator=(handle_heap const &x) {
      *this = handle_heap{x};
      return *this;
    }

    template <Allocator alloc_t_other>
    handle_heap &operator=(handle_heap<T, alloc_t_other> const &x) {
      *this = handle_heap{x};
      return *this;
    }

    // Set up a memory block of the correct size without initializing it
    handle_heap(long size, do_not_initialize_t) {
      if (size == 0) return;                                               // no size -> null handle
      auto b = allocator.allocate(size * sizeof(T)); //, alignof(T));
      ASSERT(b.ptr != nullptr);
      _data = (T *)b.ptr;
      _size = size;
    }

    // Set up a memory block of the correct size without initializing it
    handle_heap(long size, init_zero_t) {
      if (size == 0) return;                                                    // no size -> null handle
      auto b = allocator.allocate_zero(size * sizeof(T)); //, alignof(T));
      ASSERT(b.ptr != nullptr);
      _data = (T *)b.ptr;
      _size = size;
    }

    // Construct a new block of memory of given size and init if needed.
    handle_heap(long size) {
      if (size == 0) return; // no size -> null handle

      blk_t b;
      if constexpr (is_complex_v<T> && init_dcmplx)
        b = allocator.allocate_zero(size * sizeof(T));
      else
        b = allocator.allocate(size * sizeof(T));

      ASSERT(b.ptr != nullptr);
      _data = (T *)b.ptr;
      _size = size;

      // Call placement new except for complex types
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < size; ++i) new (_data + i) T();
      }
    }

    explicit handle_heap(handle_heap const &x) : handle_heap(x.size(), do_not_initialize) {
      if (is_null()) return; // nothing to do for null handle
      if constexpr (std::is_trivially_copyable_v<T>) {
	std::memcpy(_data, x.data(), x.size() * sizeof(T));
      } else {
	for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]); // placement new
      }
    }

    // Copy data from another owning handle
    template <OwningHandle<value_type> H>
    explicit handle_heap(H const &x) : handle_heap(x.size(), do_not_initialize) {
      if (is_null()) return; // nothing to do for null handle
      if constexpr (not std::is_trivially_copyable_v<T>) {
        static_assert(address_space == H::address_space,
                      "Constructing from handle of different address space requires trivially copyable value_type");
        for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]); // placement new
      } else {
        memcpy<address_space, H::address_space>((void *)_data, (void *)x.data(), _size * sizeof(T));
      }
    }

    T &operator[](long i) noexcept { return _data[i]; }
    T const &operator[](long i) const noexcept { return _data[i]; }

    bool is_null() const noexcept {
#ifdef NDA_DEBUG
      // Check the Invariants in Debug Mode
      EXPECTS((_data == nullptr) == (_size == 0));
#endif
      return _data == nullptr;
    }

    // A const-handle does not entail T const data
    T *data() const noexcept { return _data; }

    long size() const noexcept { return _size; }
  };

  // ------------------  Stack -------------------------------------

  template <typename T, size_t Size>
  // struct alignas(alignof(T)) handle_stack {
  struct handle_stack {
    static_assert(std::is_copy_constructible_v<T>,
                  "nda::mem::handle_sso requires the value_type to be copy constructible, or it can not even move (it is on stack)");
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");

    private:
    std::array<char, sizeof(T) * Size> buffer; //

    public:
    using value_type = T;
    static constexpr auto address_space = Host;

    //
    T *data() const noexcept { return (T *)buffer.data(); }

    T &operator[](long i) noexcept { return data()[i]; }
    T const &operator[](long i) const noexcept { return data()[i]; }

    handle_stack() = default;

    ~handle_stack() noexcept {
      if constexpr (!std::is_trivial_v<T>) {
        for (size_t i = 0; i < Size; ++i) data()[i].~T();
      }
    }

    handle_stack(handle_stack &&x) noexcept { operator=(x); } // no move makes a copy, we are on stack

    handle_stack &operator=(handle_stack &&x) noexcept {
      operator=(x);
      return *this;
    }

    handle_stack(long /*size*/) : handle_stack{} {
      // Call placement new except for complex types
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < Size; ++i) new (data() + i) T();
      }
    }

    handle_stack(long /*size*/, do_not_initialize_t) {}

    // Set up a memory block of the correct size without initializing it
    handle_stack(long /*size*/, init_zero_t) {
      static_assert(std::is_scalar_v<T> or is_complex_v<T>, "Internal Error");
      for (size_t i = 0; i < Size; ++i) data()[i] = 0;
    }

    handle_stack &operator=(handle_stack const &x) {
      for (size_t i = 0; i < Size; ++i) new (data() + i) T(x[i]); // placement new
      return *this;
    }

    // Construct by making a clone of the data
    handle_stack(handle_stack const &x) noexcept { // if an exception occurs in T construction, so be it, we terminate
      operator=(x);
    }

    static constexpr bool is_null() noexcept { return false; }
    static constexpr long size() noexcept { return Size; }
  };

  // ------------------  SSO -------------------------------------

  template <typename T, size_t Size>
  //struct alignas(alignof(T)) handle_sso {
  struct handle_sso {
    static_assert(Size > 0, "Size =0 makes no sense here");
    static_assert(std::is_copy_constructible_v<T>,
                  "nda::mem::handle_sso requires the value_type to be copy constructible, or it can not even move (it is on stack)");
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");

    private:
    std::array<char, sizeof(T) * Size> buffer; //

    T *_data     = nullptr; // Pointer to the start of the memory block
    size_t _size = 0;       // Size of the memory block. Invariant: size > 0 iif data != 0

    public:
    using value_type = T;
    static constexpr auto address_space = Host;

    bool on_heap() const { return _size > Size; }

    bool is_null() const noexcept {
#ifdef NDA_DEBUG
      EXPECTS((_data == nullptr) == (_size == 0));
#endif
      return _data == nullptr;
    }

    T *data() const noexcept { return _data; }

    T &operator[](long i) noexcept { return _data[i]; }
    T const &operator[](long i) const noexcept { return _data[i]; }

    long size() const noexcept { return _size; }

    // Caution! We need to provide a user-defined constructor (over =default)
    // to avoid value initialization of the buffer
    handle_sso(){};

    private:
    void clean() noexcept {
      if (is_null()) return;
      if constexpr (!std::is_trivial_v<T>) {
        for (size_t i = 0; i < _size; ++i) data()[i].~T();
        // STACK	for (size_t i = 0; i < Size; ++i) data()[i].~T();
      }
      if (on_heap()) mallocator<>::deallocate({(char *)_data, _size * sizeof(T)});
      _data = nullptr;
    }

    public:
    ~handle_sso() noexcept { clean(); }

    handle_sso(handle_sso &&x) noexcept {
      _size = x._size;
      if (on_heap()) { // heap path
        _data = x._data;
      } else {            // stack path. We MUST copy
        if (_size != 0) { // if crucial to maintain invariant
          _data = (T *)buffer.data();
          for (size_t i = 0; i < _size; ++i) new (data() + i) T(x[i]);
        }
        //for (size_t i = 0; i < Size; ++i) new (data() + i) T(x[i]);
      }
      x._data = nullptr; // steal data
      x._size = 0;       // maintain invariant
    }

    handle_sso &operator=(handle_sso &&x) noexcept {
      clean();
      _size = x._size;
      if (on_heap()) {
        _data = x._data;
      } else {
        if (_size != 0) { // if crucial to maintain invariant
          _data = (T *)buffer.data();
          for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]); // placement new
        }
      }
      x._data = nullptr; // steal data
      x._size = 0;       // maintain invariant
      return *this;
    }

    // Copy data from another owning handle
    template <OwningHandle<value_type> H>
    explicit handle_sso(H const &x) : handle_sso(x.size(), do_not_initialize) {
      if (is_null()) return; // nothing to do for null handle
      if constexpr (std::is_trivially_copyable_v<T>) {
        static_assert(address_space == H::address_space,
                      "Constructing from handle of different address space requires trivially copyable value_type");
        for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]); // placement new
      } else {
        memcpy<address_space, H::address_space>((void *)_data, (void *)x.data(), _size * sizeof(T));
      }
    }

    // Set up a memory block of the correct size without initializing it
    handle_sso(long size, do_not_initialize_t) {
      if (size == 0) return; // no size -> null handle
      _size = size;
      if (not on_heap()) {
        _data = (T *)buffer.data();
      } else {
        blk_t b;
        b = mallocator<>::allocate(size * sizeof(T));
        ASSERT(b.ptr != nullptr);
        _data = (T *)b.ptr;
      }
    }

    // Same but call calloc with malloc ...
    handle_sso(long size, init_zero_t) {
      static_assert(std::is_scalar_v<T> or is_complex_v<T>, "Internal Error");
      if (size == 0) return; // no size -> null handle
      _size = size;
      if (not on_heap()) {
        _data = (T *)buffer.data();
        for (size_t i = 0; i < _size; ++i) data()[i] = 0;
      } else {
        auto b = mallocator<>::allocate_zero(size * sizeof(T)); //, alignof(T));
        ASSERT(b.ptr != nullptr);
        _data = (T *)b.ptr;
      }
    }

    // Construct a new block of memory of given size and init if needed.
    handle_sso(long size) {
      if (size == 0) return; // no size -> null handle
      _size = size;          // NOt needed in the Stack path if type is trivial ??

      if (not on_heap()) {
        _data = (T *)buffer.data();
      } else {
        blk_t b;
        if constexpr (is_complex_v<T> && init_dcmplx)
          b = mallocator<>::allocate_zero(size * sizeof(T));
        else
          b = mallocator<>::allocate(size * sizeof(T));
        ASSERT(b.ptr != nullptr);
        _data = (T *)b.ptr;
      }

      // Call placement new except for complex types
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < size; ++i) new (_data + i) T();
      }
    }

    handle_sso &operator=(handle_sso const &x) noexcept {
      clean();
      _size = x._size;
      if (_size == 0) return *this;
      if (on_heap()) {
        blk_t b;
        b = mallocator<>::allocate(_size * sizeof(T));
        ASSERT(b.ptr != nullptr);
        _data = (T *)b.ptr;
      } else {
        _data = (T *)buffer.data();
      }
      for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]);

      return *this;
    }
  };

  // ------------------  Shared -------------------------------------

  template <typename T, AddressSpace AdrSp>
  struct handle_shared {
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");

    private:
    T *_data     = nullptr; // Pointer to the start of the memory block
    size_t _size = 0;       // Size of the memory block. Invariant: size > 0 iif data != 0

    using blk_t = std::pair<T *, size_t>;
    std::shared_ptr<void> sptr;

    public:
    using value_type = T;
    static constexpr auto address_space = AdrSp;

    handle_shared() = default;

    // Construct from foreign library shared object
    handle_shared(T *data, size_t size, void *foreign_handle, void (*foreign_decref)(void *)) noexcept
       : _data(data), _size(size), sptr{foreign_handle, foreign_decref} {}

    // Cross construction from a regular handle. MALLOC CASE ONLY. FIXME : why ?
    template <Allocator alloc_t>
    handle_shared(handle_heap<T, alloc_t> const &x) noexcept requires(alloc_t::address_space == address_space) : _data(x.data()), _size(x.size()) {
      if (not x.is_null()) sptr = x.get_sptr();
    }

    T &operator[](long i) noexcept { return _data[i]; }
    T const &operator[](long i) const noexcept { return _data[i]; }

    [[nodiscard]] bool is_null() const noexcept {
#ifdef NDA_DEBUG
      // Check the Invariants in Debug Mode
      EXPECTS((_data == nullptr) == (_size == 0));
#endif
      return _data == nullptr;
    }

    [[nodiscard]] long refcount() const noexcept { return sptr.use_count(); }

    // A constant handle does not entail T const data
    [[nodiscard]] T *data() const noexcept { return _data; }

    [[nodiscard]] long size() const noexcept { return _size; }
  };

  // ------------------  Borrowed -------------------------------------

  template <typename T, AddressSpace AdrSp = Host>
  struct handle_borrowed {
    private:
    using T0                       = std::remove_const_t<T>;
    handle_heap<T0> const *_parent = nullptr; // Parent, Required for regular->shared promotion in Python Converter
    T *_data                       = nullptr; // Pointer to the start of the memory block

    public:
    using value_type = T;
    static constexpr auto address_space = AdrSp;

    handle_borrowed() = default;
    handle_borrowed(handle_borrowed const &x) = default;

    handle_borrowed(T *ptr) noexcept : _data(ptr) {}

    template <Handle H>
    requires(address_space == H::address_space and (std::is_const_v<value_type> or !std::is_const_v<typename H::value_type>)
             and std::is_same_v<const value_type, const typename H::value_type>) handle_borrowed(H const &x, long offset = 0)
    noexcept : _data(x.data() + offset) {
      if constexpr (std::is_same_v<H, handle_heap<T0>>) _parent = &x;
    }

    T &operator[](long i) noexcept { return _data[i]; }
    T const &operator[](long i) const noexcept { return _data[i]; }

    // warnings supp
    handle_borrowed &operator=(handle_borrowed const &) = default;
    handle_borrowed &operator=(handle_borrowed &&) = default;

    [[nodiscard]] bool is_null() const noexcept { return _data == nullptr; }

    [[nodiscard]] handle_heap<T0> const *parent() const { return _parent; }

    // A const-handle does not entail T const data
    [[nodiscard]] T *data() const noexcept { return _data; }
  };

} // namespace nda::mem
