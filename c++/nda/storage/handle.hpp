// Copyright (c) 2018-2020 Simons Foundation
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
#include <limits>
#include <complex>
#include <type_traits>
#include <cstring>
#include "./blk.hpp"
#include "./allocators.hpp"

namespace nda::mem {

  //
  static constexpr bool init_dcmplx = true; // initialize dcomplex to 0 globally

  // -------------- is_complex ----------------

  template <typename T>
  static constexpr bool is_complex_v = false;
  template <typename T>
  static constexpr bool is_complex_v<std::complex<T>> = true;

  // -------------- Allocation -----------

  template <typename Allocator>
  struct allocator_singleton {

#ifndef NDA_DEBUG_LEAK_CHECK
    static inline Allocator allocator;
#else
    static inline allocators::leak_check<Allocator> allocator;
#endif

    static allocators::blk_t allocate(size_t size) { return allocator.allocate(size); }
    static allocators::blk_t allocate_zero(size_t size) { return allocator.allocate_zero(size); }
    static void deallocate(allocators::blk_t b) { allocator.deallocate(b); }
  };

#ifndef NDA_DEBUG_LEAK_CHECK

  // the default mallocator is special : it has no state and a special calloc
  // use void : it is the default case, and simplify error messages in 99.999% of cases
  template <>
  struct allocator_singleton<void> {
    static allocators::blk_t allocate(size_t size) { return allocators::mallocator::allocate(size); }
    static allocators::blk_t allocate_zero(size_t size) { return allocators::mallocator::allocate_zero(size); }
    static void deallocate(allocators::blk_t b) { allocators::mallocator::deallocate(b); }
  };
#else
  template <>
  struct allocator_singleton<void> : allocator_singleton<allocators::leak_check<allocators::mallocator>> {};

#endif

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
  // clang-format off
  template <typename T, typename Allocator> struct handle_heap; 
  template <typename T> struct handle_shared; 
  template <typename T> struct handle_borrowed; 

  template <typename T, size_t Size> struct handle_stack;
  template <typename T, size_t Size> struct handle_sso;
  // clang-format on

  // ------------------  HEAP -------------------------------------

  template <typename T, typename Allocator>
  struct handle_heap {
    private:
    T *_data     = nullptr; // Pointer to the start of the memory block
    size_t _size = 0;       // Size of the memory block. Invariant: size > 0 iif data != 0

    // In case we need to share the memory
    mutable std::shared_ptr<void> sptr;

    using blk_t = std::pair<T *, size_t>;

    // code for destructor
    static void destruct(blk_t b) noexcept {

      static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");
      auto [data, size] = b;

      if (data == nullptr) return;

      // If needed, call the T destructors
      if constexpr (!std::is_trivial_v<T>) {
        for (size_t i = 0; i < size; ++i) data[i].~T();
      }

      // Deallocate the memory block
      allocator_singleton<Allocator>::deallocate({(char *)data, size * sizeof(T)});
    }

    // a deleter for the data in the sptr
    static void deleter(void *p) noexcept { destruct(*((blk_t *)p)); }

    public:
    std::shared_ptr<void> get_sptr() const {
      if (not sptr) sptr.reset(new blk_t{_data, _size}, deleter);
      return sptr;
    }

    using value_type = T;

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

    // Set up a memory block of the correct size without initializing it
    handle_heap(long size, do_not_initialize_t) {
      if (size == 0) return;                                               // no size -> null handle
      auto b = allocator_singleton<Allocator>::allocate(size * sizeof(T)); //, alignof(T));
      ASSERT(b.ptr != nullptr);
      _data = (T *)b.ptr;
      _size = size;
    }

    // Set up a memory block of the correct size without initializing it
    handle_heap(long size, init_zero_t) {
      static_assert(std::is_scalar_v<T> or is_complex_v<T>, "Internal Error");
      if (size == 0) return;                                                    // no size -> null handle
      auto b = allocator_singleton<Allocator>::allocate_zero(size * sizeof(T)); //, alignof(T));
      ASSERT(b.ptr != nullptr);
      _data = (T *)b.ptr;
      _size = size;
    }

    // Construct a new block of memory of given size and init if needed.
    handle_heap(long size) {
      if (size == 0) return; // no size -> null handle

      allocators::blk_t b;
      if constexpr (is_complex_v<T> && init_dcmplx)
        b = allocator_singleton<Allocator>::allocate_zero(size * sizeof(T));
      else
        b = allocator_singleton<Allocator>::allocate(size * sizeof(T));

      ASSERT(b.ptr != nullptr);
      _data = (T *)b.ptr;
      _size = size;

      // Call placement new except for complex types
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < size; ++i) new (_data + i) T();
      }
    }

    // Construct by making a clone of the data
    handle_heap(handle_heap const &x) : handle_heap(x.size(), do_not_initialize) {
      if (is_null()) return; // nothing to do for null handle
      if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(_data, x.data(), x.size() * sizeof(T));
      } else {
        for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]); // placement new
      }
    }

    // Construct by making a clone of the data. same code
    handle_heap(handle_shared<T> const &x) : handle_heap(x.size(), do_not_initialize) {
      if (is_null()) return; // nothing to do for null handle
      if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(_data, x.data(), x.size() * sizeof(T));
      } else {
        for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]); // placement new
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

    handle_sso() = default;

    private:
    void clean() noexcept {
      if (is_null()) return;
      if constexpr (!std::is_trivial_v<T>) {
        for (size_t i = 0; i < _size; ++i) data()[i].~T();
        // STACK	for (size_t i = 0; i < Size; ++i) data()[i].~T();
      }
      if (on_heap()) allocators::mallocator::deallocate({(char *)_data, _size * sizeof(T)});
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

    // Set up a memory block of the correct size without initializing it
    handle_sso(long size, do_not_initialize_t) {
      if (size == 0) return; // no size -> null handle
      _size = size;
      if (not on_heap()) {
        _data = (T *)buffer.data();
      } else {
        allocators::blk_t b;
        b = allocators::mallocator::allocate(size * sizeof(T));
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
        auto b = allocators::mallocator::allocate_zero(size * sizeof(T)); //, alignof(T));
        ASSERT(b.ptr != nullptr);
        _data = (T *)b.ptr;
      }
    }

    // Copy data
    handle_sso(handle_sso const &x) : handle_sso(x.size(), do_not_initialize) {
      for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]); // placement new
    }

    handle_sso &operator=(handle_sso const &x) noexcept {
      clean();
      _size = x._size;
      if (_size == 0) return *this;
      if (on_heap()) {
        allocators::blk_t b;
        b = allocators::mallocator::allocate(_size * sizeof(T));
        ASSERT(b.ptr != nullptr);
        _data = (T *)b.ptr;
      } else {
        _data = (T *)buffer.data();
      }
      for (size_t i = 0; i < _size; ++i) new (_data + i) T(x[i]);

      return *this;
    }

    // Construct a new block of memory of given size and init if needed.
    handle_sso(long size) {
      if (size == 0) return; // no size -> null handle
      _size = size;          // NOt needed in the Stack path if type is trivial ??

      if (not on_heap()) {
        _data = (T *)buffer.data();
        _size = size;
      } else {
        allocators::blk_t b;
        if constexpr (is_complex_v<T> && init_dcmplx)
          b = allocators::mallocator::allocate_zero(size * sizeof(T));
        else
          b = allocators::mallocator::allocate(size * sizeof(T));
        ASSERT(b.ptr != nullptr);
        _data = (T *)b.ptr;
        _size = size;
      }

      // Call placement new except for complex types
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < size; ++i) new (_data + i) T();
      }
    }
  };

  // ------------------  Shared -------------------------------------

  template <typename T>
  struct handle_shared {
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");

    private:
    T *_data     = nullptr; // Pointer to the start of the memory block
    size_t _size = 0;       // Size of the memory block. Invariant: size > 0 iif data != 0

    using blk_t = std::pair<T *, size_t>;
    std::shared_ptr<void> sptr;

    public:
    using value_type = T;

    handle_shared() = default;

    // Construct from foreign library shared object
    handle_shared(T *data, size_t size, void *foreign_handle, void (*foreign_decref)(void *)) noexcept
       : _data(data), _size(size), sptr{foreign_handle, foreign_decref} {}

    // Cross construction from a regular handle. MALLOC CASE ONLY. FIXME : why ?
    handle_shared(handle_heap<T, void> const &x) noexcept : _data(x.data()), _size(x.size()) {
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

  template <typename T>
  struct handle_borrowed {
    using T0 = std::remove_const_t<T>;

    private:
    handle_heap<T0, void> const *_parent = nullptr; // Parent, Required for regular->shared promotion in Python Converter
    T *_data                             = nullptr; // Pointer to the start of the memory block

    public:
    using value_type = T;

    handle_borrowed() = default;

    handle_borrowed(T *ptr) noexcept : _data(ptr) {}
    handle_borrowed(handle_borrowed<T> const &x) = default;

    handle_borrowed(handle_borrowed<T> const &x, long offset) noexcept : _data(x.data() + offset) {}

    handle_borrowed(handle_heap<T0, void> const &x, long offset = 0) noexcept : _parent(&x), _data(x.data() + offset) {}

    template <typename Alloc>
    handle_borrowed(handle_heap<T0, Alloc> const &x, long offset = 0) noexcept : _parent(nullptr), _data(x.data() + offset) {}

    handle_borrowed(handle_shared<T0> const &x, long offset = 0) noexcept : _data(x.data() + offset) {}
    handle_borrowed(handle_borrowed<T0> const &x, long offset = 0) noexcept REQUIRES(std::is_const_v<T>) : _data(x.data() + offset) {}

    template <size_t Size>
    handle_borrowed(handle_stack<T0, Size> const &x, long offset = 0) noexcept : _data(x.data() + offset) {}

    template <size_t SSO_Size>
    handle_borrowed(handle_sso<T0, SSO_Size> const &x, long offset = 0) noexcept : _data(x.data() + offset) {}

    T &operator[](long i) noexcept { return _data[i]; }
    T const &operator[](long i) const noexcept { return _data[i]; }

    // warnings supp
    handle_borrowed &operator=(handle_borrowed const &) = default;
    handle_borrowed &operator=(handle_borrowed &&) = default;

    [[nodiscard]] bool is_null() const noexcept { return _data == nullptr; }

    [[nodiscard]] handle_heap<T0, void> const *parent() const { return _parent; }

    // A const-handle does not entail T const data
    [[nodiscard]] T *data() const noexcept { return _data; }
  };

} // namespace nda::mem
