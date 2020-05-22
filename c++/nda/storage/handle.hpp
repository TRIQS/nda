/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017, O. Parcollet
 * Copyright (C) 2018-2019, The Simons Foundation
 *   authors : O. Parcollet, N. Wentzell
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <limits>
#include <complex>
#include <type_traits>
#include <cstring>
#include "./blk.hpp"
#include "./rtable.hpp"

namespace nda::mem {

  // -------------- is_complex ----------------

  template <typename T>
  static constexpr bool is_complex_v = false;
  template <typename T>
  static constexpr bool is_complex_v<std::complex<T>> = true;

  // -------------- Allocation Functions ---------------------------

  allocators::blk_t allocate(size_t size);
  allocators::blk_t allocate_zero(size_t size);
  void deallocate(allocators::blk_t b);

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
  template <typename T> struct handle_heap; 
  template <typename T> struct handle_shared; 
  template <typename T> struct handle_borrowed; 

  template <typename T, size_t Size> struct handle_stack;
  // clang-format on

  // ------------------  HEAP -------------------------------------

  template <typename T>
  struct handle_heap {
    private:
    T *_data     = nullptr; // Pointer to the start of the memory block
    size_t _size = 0;       // Size of the memory block. Invariant: size > 0 iif data != 0

    // The regular handle can share its memory with handle_shared<T>
    mutable long _id = 0; // The id in the refcounts table.
                          // id == 0 corresponds to the case of no memory sharing
                          // This field must be mutable for the cross construction of 'S'. Cf 'S'.
                          // Invariant: id == 0 if data == nullptr
    friend handle_shared<T>;

    void decref() noexcept {
      static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");
      if (is_null()) return;

      // Check if the memory is shared and still pointed to
      if (has_shared_memory() and not globals::rtable.decref(_id)) return;

      // If needed, call the T destructors
      if constexpr (!std::is_trivial_v<T>) {
        for (size_t i = 0; i < _size; ++i) _data[i].~T();
      }

      // Deallocate the memory block
      deallocate({(char *)_data, _size * sizeof(T)});
    }

    public:
    using value_type = T;

    ~handle_heap() noexcept { decref(); }

    handle_heap() = default;

    handle_heap(handle_heap &&x) noexcept {
      _data   = x._data;
      _size   = x._size;
      _id     = x._id;
      x._data = nullptr;
      x._size = 0;
      x._id   = 0;
    }

    handle_heap &operator=(handle_heap const &x) {
      *this = handle_heap{x};
      return *this;
    }

    handle_heap &operator=(handle_heap &&x) noexcept {
      decref();
      _data   = x._data;
      _size   = x._size;
      _id     = x._id;
      x._data = nullptr;
      x._size = 0;
      x._id   = 0;
      return *this;
    }

    // Set up a memory block of the correct size without initializing it
    handle_heap(long size, do_not_initialize_t) {
      if (size == 0) return;               // no size -> null handle
      auto b = allocate(size * sizeof(T)); //, alignof(T));
      ASSERT(b.ptr != nullptr);
      _data = (T *)b.ptr;
      _size = size;
    }

    // Set up a memory block of the correct size without initializing it
    handle_heap(long size, init_zero_t) {
      static_assert(std::is_scalar_v<T> or is_complex_v<T>, "Internal Error");
      if (size == 0) return;                    // no size -> null handle
      auto b = allocate_zero(size * sizeof(T)); //, alignof(T));
      ASSERT(b.ptr != nullptr);
      _data = (T *)b.ptr;
      _size = size;
    }

    // Construct a new block of memory of given size and init if needed.
    handle_heap(long size) {
      if (size == 0) return; // no size -> null handle

      allocators::blk_t b;
      if constexpr (is_complex_v<T> && globals::init_dcmplx)
        b = allocate_zero(size * sizeof(T));
      else
        b = allocate(size * sizeof(T));

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
      EXPECTS(_data != nullptr or _id == 0);
      EXPECTS((_data == nullptr) == (_size == 0));
#endif
      return _data == nullptr;
    }
    bool has_shared_memory() const noexcept { return _id != 0; }

    // A const-handle does not entail T const data
    T *data() const noexcept { return _data; }

    long size() const noexcept { return _size; }
  };

  // ------------------  Stack -------------------------------------

  template <typename T, size_t Size>
  struct alignas(alignof(T)) handle_stack {
    private:
    std::array<char, sizeof(T) * Size> __buffer; //
    //char __buffer[sizeof(T) * Size]; //

    public:
    using value_type = T;

    //
    T *data() const noexcept { return (T *)__buffer.data(); }

    T &operator[](long i) noexcept { return data()[i]; }
    T const &operator[](long i) const noexcept { return data()[i]; }

    handle_stack() {
      // Call placement new except for complex types
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < Size; ++i) new (data() + i) T();
      }
    }

    handle_stack(long /*size*/) : handle_stack{} {}

    handle_stack(long /*size*/, do_not_initialize_t) {}

    ~handle_stack() noexcept {
      static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");
      //if (is_null()) retun;
      // If needed, call the T destructors
      if constexpr (!std::is_trivial_v<T>) {
        for (size_t i = 0; i < Size; ++i) data()[i].~T();
      }
    }

    handle_stack &operator=(handle_stack const &x) {
      if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(data(), x.data(), Size * sizeof(T));
      } else {
        for (size_t i = 0; i < Size; ++i) new (data() + i) T(x[i]); // placement new
      }
      return *this;
    }
    // Construct by making a clone of the data
    handle_stack(handle_stack const &x) noexcept { // if an exception occurs in T construction, so be it, we terminate
      operator=(x);
    }

    handle_stack(handle_stack &&x) noexcept { operator=(x); } // no move makes a copy, we are on stack

    handle_stack &operator=(handle_stack &&x) noexcept {
      operator=(x);
      return *this;
    }

    // Set up a memory block of the correct size without initializing it
    handle_stack(long /*size*/, init_zero_t) { static_assert(std::is_scalar_v<T> or is_complex_v<T>, "Internal Error"); }

    static constexpr bool is_null() noexcept { return false; }
    static constexpr long size() noexcept { return Size; }
    //static constexpr bool has_shared_memory() const noexcept { return false; }
  };

  // ------------------  Shared -------------------------------------

  template <typename T>
  struct handle_shared {
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle requires the value_type to have a non-throwing constructor");

    private:
    T *_data     = nullptr; // Pointer to the start of the memory block
    size_t _size = 0;       // Size of the memory block. Invariant: size > 0 iif data != 0

    long _id = 0; // The id in the refcounts table. id == 0 corresponds to a null-handle
                  // Invariant: id == 0 iif data == nullptr

    // Allow to take ownership of a shared pointer from another lib, e.g. numpy.
    void *_foreign_handle = nullptr; // Memory handle of the foreign library
    void *_foreign_decref = nullptr; // Function pointer to decref of the foreign library (void (*)(void *))

    public:
    using value_type = T;

    void decref() noexcept {
      if (is_null()) return;

      // Check if the memory is shared and still pointed to
      if (!globals::rtable.decref(_id)) return;

      // If the memory was allocated in a foreign foreign lib, release it there
      if (_foreign_handle) {
        ((void (*)(void *))_foreign_decref)(_foreign_handle);
        return;
      }

      // If needed, call the T destructors
      if constexpr (!std::is_trivial_v<T>) {
        for (size_t i = 0; i < _size; ++i) _data[i].~T();
      }

      // Deallocate the memory block
      deallocate({(char *)_data, _size * sizeof(T)});
    }

    void incref() noexcept {
#ifdef NDA_DEBUG
      EXPECTS(!is_null());
#endif
      globals::rtable.incref(_id);
    }

    private:
    // basic part of copy, no ref handling here
    void _copy(handle_shared const &x) noexcept {
      _data           = x._data;
      _size           = x._size;
      _id             = x._id;
      _foreign_handle = x._foreign_handle;
      _foreign_decref = x._foreign_decref;
    }

    public:
    ~handle_shared() noexcept { decref(); }

    handle_shared() = default;

    handle_shared(handle_shared const &x) noexcept {
      _copy(x);
      if (!is_null()) incref();
    }

    handle_shared(handle_shared &&x) noexcept {
      _copy(x);

      // Invalidate x so it destructs trivally
      x._data = nullptr;
      x._size = 0;
      x._id   = 0;
    }

    handle_shared &operator=(handle_shared const &x) noexcept {
      decref(); // Release my ref if I have one
      _copy(x);
      incref();
      return *this;
    }

    handle_shared &operator=(handle_shared &&x) noexcept {
      decref(); // Release my ref if I have one
      _copy(x);

      // Invalidate x so it destructs trivially
      x._data = nullptr;
      x._size = 0;
      x._id   = 0;

      return *this;
    }

    // Construct from foreign library shared object
    handle_shared(T *data, size_t size, void *foreign_handle, void *foreign_decref) noexcept
       : _data(data), _size(size), _foreign_handle(foreign_handle), _foreign_decref(foreign_decref) {
      // Only one thread should fetch the id
      std::lock_guard<std::mutex> lock(globals::rtable.mtx);
      _id = globals::rtable.get();
    }

    // Cross construction from a regular handle
    handle_shared(handle_heap<T> const &x) noexcept : _data(x.data()), _size(x.size()) {
      if (x.is_null()) return;

      // Get an id if necessary
      if (x._id == 0) {
        // Only one thread should fetch the id
        std::lock_guard<std::mutex> lock(globals::rtable.mtx);
        if (x._id == 0) x._id = globals::rtable.get();
      }
      _id = x._id;
      // Increase refcount
      incref();
    }

    T &operator[](long i) noexcept { return _data[i]; }
    T const &operator[](long i) const noexcept { return _data[i]; }

    [[nodiscard]] bool is_null() const noexcept {
#ifdef NDA_DEBUG
      // Check the Invariants in Debug Mode
      EXPECTS((_data == nullptr) == (_size == 0));
      EXPECTS((_data == nullptr) == (_id == 0));
#endif
      return _data == nullptr;
    }

    [[nodiscard]] long refcount() const noexcept { return globals::rtable.refcounts()[_id]; }

    // A constant handle does not entail T const data
    [[nodiscard]] T *data() const noexcept { return _data; }

    [[nodiscard]] long size() const noexcept { return _size; }
  };

  // ------------------  Borrowed -------------------------------------

  template <typename T>
  struct handle_borrowed {
    using T0 = std::remove_const_t<T>;

    private:
    handle_heap<T0> const *_parent = nullptr; // Parent, Required for regular->shared promotion in Python Converter
    T *_data                       = nullptr; // Pointer to the start of the memory block

    public:
    using value_type = T;

    handle_borrowed() = default;

    handle_borrowed(T *ptr) noexcept : _data(ptr) {}
    handle_borrowed(handle_borrowed<T> const &x) = default;

    handle_borrowed(handle_borrowed<T> const &x, long offset) noexcept : _data(x.data() + offset) {}

    handle_borrowed(handle_heap<T0> const &x, long offset = 0) noexcept : _parent(&x), _data(x.data() + offset) {}
    handle_borrowed(handle_shared<T0> const &x, long offset = 0) noexcept : _data(x.data() + offset) {}
    handle_borrowed(handle_borrowed<T0> const &x, long offset = 0) noexcept REQUIRES(std::is_const_v<T>) : _data(x.data() + offset) {}

    template <size_t Size>
    handle_borrowed(handle_stack<T0, Size> const &x, long offset = 0) noexcept : _data(x.data() + offset) {}

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
