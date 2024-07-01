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
// Authors: Miguel Morales, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides various handles to take care of memory management for nda::basic_array
 * and nda::basic_array_view types.
 */

#pragma once

#include "./allocators.hpp"
#include "./memcpy.hpp"
#include "../concepts.hpp"
#include "../macros.hpp"

#include <array>
#include <memory>
#include <type_traits>
#include <utility>

namespace nda::mem {

  /**
   * @addtogroup mem_utils
   * @{
   */

  /// Should we initialize memory for complex double types to zero.
  static constexpr bool init_dcmplx = true;

  /**
   * @brief Wraps an arbitrary type to have a specified alignment.
   *
   * @tparam T Given type.
   * @tparam Al Alignment.
   */
  template <typename T, int Al>
  struct alignas(Al) aligner {
    /// Wrapped object of type T.
    T x;

    /// Get reference to the wrapped object.
    [[nodiscard]] T &get() noexcept { return x; }

    /// Get const reference to the wrapped object.
    [[nodiscard]] T const &get() const noexcept { return x; }
  };

  /// Tag used in constructors to indicate that the memory should not be initialized.
  struct do_not_initialize_t {};

  /// Instance of nda::mem::do_not_initialize_t.
  inline static constexpr do_not_initialize_t do_not_initialize{};

  /// Tag used in constructors to indicate that the memory should be initialized to zero.
  struct init_zero_t {};

  /// Instance of nda::mem::init_zero_t.
  inline static constexpr init_zero_t init_zero{};

  /** @} */

  /**
   * @addtogroup mem_handles
   * @{
   */

  /// @cond
  // Forward declaration.
  template <typename T, AddressSpace AdrSp = Host>
  struct handle_shared;
  /// @endcond

  /**
   * @brief A handle for a memory block on the heap.
   *
   * @details By default it uses nda::mem::mallocator to allocate/deallocate memory.
   *
   * @tparam T Value type of the data.
   * @tparam A nda::mem::Allocator type.
   */
  template <typename T, Allocator A = mallocator<>>
  struct handle_heap {
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle_heap requires the value_type to have a non-throwing destructor");

    private:
    // Pointer to the start of the actual data.
    T *_data = nullptr;

    // Size of the data (number of T elements). Invariant: size > 0 iif data != nullptr.
    size_t _size = 0;

    // Allocator to use.
#ifndef NDA_DEBUG_LEAK_CHECK
    static inline A allocator; // NOLINT (allocator is not specific to a single instance)
#else
    static inline leak_check<A> allocator; // NOLINT (allocator is not specific to a single instance)
#endif

    // For shared ownership (points to a blk_T_t).
    mutable std::shared_ptr<void> sptr;

    // Type of the memory block, i.e. a pointer to the data and its size.
    using blk_T_t = std::pair<T *, size_t>;

    // Release the handled memory (data pointer and size are not set to null here).
    static void destruct(blk_T_t b) noexcept {
      auto [data, size] = b;

      // do nothing if the data is null
      if (data == nullptr) return;

      // if needed, call the destructors of the objects stored
      if constexpr (A::address_space == Host and !(std::is_trivial_v<T> or nda::is_complex_v<T>)) {
        for (size_t i = 0; i < size; ++i) data[i].~T();
      }

      // deallocate the memory block
      allocator.deallocate({(char *)data, size * sizeof(T)});
    }

    // Deleter for the shared pointer.
    static void deleter(void *p) noexcept { destruct(*((blk_T_t *)p)); }

    public:
    /// Value type of the data.
    using value_type = T;

    /// nda::mem::Allocator type.
    using allocator_type = A;

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = allocator_type::address_space;

    /**
     * @brief Get a shared pointer to the memory block.
     * @return A copy of the shared pointer stored in the current handle.
     */
    std::shared_ptr<void> get_sptr() const {
      if (not sptr) sptr.reset(new blk_T_t{_data, _size}, deleter);
      return sptr;
    }

    /**
     * @brief Destructor for the handle.
     * @details If the shared pointer is set, it does nothing. Otherwise, it explicitly calls the destructor of
     * non-trivial objects and deallocates the memory.
     */
    ~handle_heap() noexcept {
      if (not sptr and not(is_null())) destruct({_data, _size});
    }

    /// Default constructor leaves the handle in a null state (`nullptr` and size 0).
    handle_heap() = default;

    /**
     * @brief Move constructor simply copies the pointers and size and resets the source handle to a null state.
     * @param h Source handle.
     */
    handle_heap(handle_heap &&h) noexcept : _data(h._data), _size(h._size), sptr(std::move(h.sptr)) {
      h._data = nullptr;
      h._size = 0;
    }

    /**
     * @brief Move assignment operator first releases the resources held by the current handle and then moves the
     * resources from the source to the current handle.
     *
     * @param h Source handle.
     */
    handle_heap &operator=(handle_heap &&h) noexcept {
      // release current resources if they are not shared and not null
      if (not sptr and not(is_null())) destruct({_data, _size});

      // move the resources from the source handle
      _data = h._data;
      _size = h._size;
      sptr  = std::move(h.sptr);

      // reset the source handle to a null state
      h._data = nullptr;
      h._size = 0;
      return *this;
    }

    /**
     * @brief Copy constructor makes a deep copy of the data from another handle.
     * @param h Source handle.
     */
    explicit handle_heap(handle_heap const &h) : handle_heap(h.size(), do_not_initialize) {
      if (is_null()) return;
      if constexpr (std::is_trivially_copyable_v<T>) {
        memcpy<address_space, address_space>(_data, h.data(), h.size() * sizeof(T));
      } else {
        for (size_t i = 0; i < _size; ++i) new (_data + i) T(h[i]);
      }
    }

    /**
     * @brief Copy assignment operator utilizes the copy constructor and move assignment operator to make a deep copy of
     * the data and size from the source handle.
     *
     * @param h Source handle.
     */
    handle_heap &operator=(handle_heap const &h) {
      *this = handle_heap{h};
      return *this;
    }

    /**
     * @brief Construct a handle by making a deep copy of the data from another handle.
     *
     * @tparam H nda::mem::OwningHandle type.
     * @param h Source handle.
     */
    template <OwningHandle<value_type> H>
    explicit handle_heap(H const &h) : handle_heap(h.size(), do_not_initialize) {
      if (is_null()) return;
      if constexpr (std::is_trivially_copyable_v<T>) {
        memcpy<address_space, H::address_space>((void *)_data, (void *)h.data(), _size * sizeof(T));
      } else {
        static_assert(address_space == H::address_space,
                      "Constructing an nda::mem::handle_heap from a handle of a different address space requires a trivially copyable value_type");
        for (size_t i = 0; i < _size; ++i) new (_data + i) T(h[i]);
      }
    }

    /**
     * @brief Assignment operator utilizes another constructor and move assignment to make a deep copy of the data and
     * size from the source handle.
     *
     * @tparam AS Allocator type of the source handle.
     * @param h Source handle with a different allocator.
     */
    template <Allocator AS>
    handle_heap &operator=(handle_heap<T, AS> const &h) {
      *this = handle_heap{h};
      return *this;
    }

    /**
     * @brief Construct a handle by allocating memory for the data of a given size but without initializing it.
     * @param size Size of the data (number of elements).
     */
    handle_heap(long size, do_not_initialize_t) {
      if (size == 0) return;
      auto b = allocator.allocate(size * sizeof(T));
      if (not b.ptr) throw std::bad_alloc{};
      _data = (T *)b.ptr;
      _size = size;
    }

    /**
     * @brief Construct a handle by allocating memory for the data of a given size and initializing it to zero.
     * @param size Size of the data (number of elements).
     */
    handle_heap(long size, init_zero_t) {
      if (size == 0) return;
      auto b = allocator.allocate_zero(size * sizeof(T));
      if (not b.ptr) throw std::bad_alloc{};
      _data = (T *)b.ptr;
      _size = size;
    }

    /**
     * @brief Construct a handle by allocating memory for the data of a given size and initializing it depending on the
     * value type.
     *
     * @details The data is initialized as follows:
     * - If `T` is std::complex and nda::mem::init_dcmplx is true, the data is initialized to zero.
     * - If `T` is not trivial and not complex, the data is default constructed by placement new operator calls.
     * - Otherwise, the data is not initialized.
     *
     * @param size Size of the data (number of elements).
     */
    handle_heap(long size) {
      if (size == 0) return;
      blk_t b;
      if constexpr (is_complex_v<T> && init_dcmplx)
        b = allocator.allocate_zero(size * sizeof(T));
      else
        b = allocator.allocate(size * sizeof(T));
      if (not b.ptr) throw std::bad_alloc{};
      _data = (T *)b.ptr;
      _size = size;

      // call placement new for non trivial and non complex types
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < size; ++i) new (_data + i) T();
      }
    }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Reference to the element at the given index.
     */
    [[nodiscard]] T &operator[](long i) noexcept { return _data[i]; }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Const reference to the element at the given index.
     */
    [[nodiscard]] T const &operator[](long i) const noexcept { return _data[i]; }

    /**
     * @brief Check if the handle is in a null state.
     * @return True if the data is a `nullptr` (and the size is 0).
     */
    [[nodiscard]] bool is_null() const noexcept {
#ifdef NDA_DEBUG
      // check the invariants in debug mode
      EXPECTS((_data == nullptr) == (_size == 0));
#endif
      return _data == nullptr;
    }

    /**
     * @brief Get a pointer to the stored data.
     * @return Pointer to the start of the handled memory.
     */
    [[nodiscard]] T *data() const noexcept { return _data; }

    /**
     * @brief Get the size of the handle.
     * @return Number of elements of type `T` in the handled memory.
     */
    [[nodiscard]] long size() const noexcept { return _size; }
  };

  /**
   * @brief A handle for a memory block on the stack.
   *
   * @note This handle only works with the `Host` nda::mem::AddressSpace.
   *
   * @tparam T Value type of the data.
   * @tparam Size Size of the data (number of elements).
   */
  template <typename T, size_t Size>
  struct handle_stack {
    static_assert(std::is_copy_constructible_v<T>, "nda::mem::handle_stack requires the value_type to be copy constructible");
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle_stack requires the value_type to have a non-throwing destructor");

    private:
    // Memory buffer on the stack to store the data.
    std::array<char, sizeof(T) * Size> buffer;

    public:
    /// Value type of the data.
    using value_type = T;

    /// nda::mem::AddressSpace in which the memory is allocated (always on `Host`).
    static constexpr auto address_space = Host;

    /**
     * @brief Destructor for the handle.
     * @details For non-trivial objects, it explicitly calls their destructors. Otherwise, it does nothing.
     */
    ~handle_stack() noexcept {
      if constexpr (!std::is_trivial_v<T>) {
        // explicitly call the destructor for each non-trivial object
        for (size_t i = 0; i < Size; ++i) data()[i].~T();
      }
    }

    /// Default constructor leaves the data uninitialized.
    handle_stack() = default;

    /**
     * @brief Move constructor simply calls the copy assignment operator.
     * @details If an exception occurs in the constructor of `T`, the program terminates.
     * @param h Source handle.
     */
    handle_stack(handle_stack &&h) noexcept { operator=(h); }

    /**
     * @brief Move assignment operator simply calls the copy assignment operator.
     * @details If an exception occurs in the constructor of `T`, the program terminates.
     * @param h Source handle.
     */
    handle_stack &operator=(handle_stack &&h) noexcept {
      operator=(h);
      return *this;
    }

    /**
     * @brief Copy constructor simply calls the copy assignment operator.
     * @details If an exception occurs in the constructor of `T`, the program terminates.
     * @param h Source handle.
     */
    handle_stack(handle_stack const &h) noexcept { operator=(h); }

    /**
     * @brief Copy assignment operator makes a deep copy of the data from the source handle using placement new.
     * @param h Source handle.
     */
    handle_stack &operator=(handle_stack const &h) {
      for (size_t i = 0; i < Size; ++i) new (data() + i) T(h[i]);
      return *this;
    }

    /// Construct a handle and do not initialize the data.
    handle_stack(long /*size*/, do_not_initialize_t) {}

    /// Construct a handle and initialize the data to zero (only for scalar and complex types).
    handle_stack(long /*size*/, init_zero_t) {
      static_assert(std::is_scalar_v<T> or is_complex_v<T>, "nda::mem::handle_stack can only be initialized to zero for scalar and complex types");
      for (size_t i = 0; i < Size; ++i) data()[i] = 0;
    }

    /**
     * @brief Construct a handle and initialize the data depending on the value type.
     *
     * @details The data is initialized as follows:
     * - If `T` is not trivial and not complex, the data is default constructed by placement new operator calls.
     * - Otherwise, the data is not initialized.
     */
    handle_stack(long /*size*/) : handle_stack{} {
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < Size; ++i) new (data() + i) T();
      }
    }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Reference to the element at the given index.
     */
    [[nodiscard]] T &operator[](long i) noexcept { return data()[i]; }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Const reference to the element at the given index.
     */
    [[nodiscard]] T const &operator[](long i) const noexcept { return data()[i]; }

    /**
     * @brief Check if the handle is in a null state.
     * @return False (there is no null state on the stack).
     */
    static constexpr bool is_null() noexcept { return false; }

    /**
     * @brief Get a pointer to the stored data.
     * @return Pointer to the start of the handled memory.
     */
    [[nodiscard]] T *data() const noexcept { return (T *)buffer.data(); }

    /**
     * @brief Get the size of the handle.
     * @return Number of elements of type `T` in the handled memory.
     */
    static constexpr long size() noexcept { return Size; }
  };

  /**
   * @brief A handle for a memory block on the heap or stack depending on the size of the data.
   *
   * @details If the size of the data is less than or equal to the template parameter `Size`, the data is stored on the
   * stack, otherwise it is stored on the heap (nda::mem::mallocator is used to allocate/deallocate memory on the heap).
   *
   * It simulates the small string optimization (SSO) for strings.
   *
   * @note This handle only works with the `Host` nda::mem::AddressSpace.
   *
   * @tparam T Value type of the data.
   * @tparam Size Max. size of the data to store on the stack (number of elements).
   */
  template <typename T, size_t Size>
  struct handle_sso { // struct alignas(alignof(T)) handle_sso {
    static_assert(Size > 0, "Size == 0 makes no sense in nda::mem::handle_sso");
    static_assert(std::is_copy_constructible_v<T>, "nda::mem::handle_sso requires the value_type to be copy constructible");
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle_sso requires the value_type to have a non-throwing destructor");

    private:
    // Memory buffer on the stack to store the data.
    std::array<char, sizeof(T) * Size> buffer;

    // Pointer to the start of the actual data.
    T *_data = nullptr;

    // Size of the data (number of T elements). Invariant: size > 0 iif data != nullptr.
    size_t _size = 0;

    // Release the handled memory.
    void clean() noexcept {
      if (is_null()) return;
      if constexpr (!std::is_trivial_v<T>) {
        // explicitly call the destructor for each non-trivial object
        for (size_t i = 0; i < _size; ++i) data()[i].~T();
      }
      if (on_heap()) mallocator<>::deallocate({(char *)_data, _size * sizeof(T)});
      _data = nullptr;
      _size = 0;
    }

    public:
    /// Value type of the data.
    using value_type = T;

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = Host;

    /// Default constructor.
    handle_sso(){}; // NOLINT (user-defined constructor to avoid value initialization of the buffer)

    /**
     * @brief Destructor for the handle.
     *
     * @details For non-trivial objects, it explicitly calls their destructors. Additionally, it deallocates the memory
     * if it is stored on the heap and resets the handle to its null state.
     */
    ~handle_sso() noexcept { clean(); }

    /**
     * @brief Move constructor either copies the heap pointers or makes a deep copy of the stack data.
     * @details In both cases, it resets the source handle to a null state.
     * @param h Source handle.
     */
    handle_sso(handle_sso &&h) noexcept : _size(h._size) {
      if (on_heap()) {
        _data = h._data;
      } else {
        if (_size != 0) {
          _data = (T *)buffer.data();
          for (size_t i = 0; i < _size; ++i) new (data() + i) T(h[i]);
        }
      }
      h._data = nullptr;
      h._size = 0;
    }

    /**
     * @brief Move assignment operator first releases the resources held by the current handle and then either copies
     * the heap pointers or makes a deep copy of the stack data.
     *
     * @details In both cases, it resets the source handle to a null state.
     *
     * @param h Source handle.
     */
    handle_sso &operator=(handle_sso &&h) noexcept {
      clean();
      _size = h._size;
      if (on_heap()) {
        _data = h._data;
      } else {
        if (_size != 0) {
          _data = (T *)buffer.data();
          for (size_t i = 0; i < _size; ++i) new (_data + i) T(h[i]);
        }
      }
      h._data = nullptr;
      h._size = 0;
      return *this;
    }

    /**
     * @brief Copy construct a handle by making a deep copy of the data from the source handle.
     * @param h Source handle.
     */
    handle_sso(handle_sso const &h) : handle_sso(h.size(), do_not_initialize) {
      if (is_null()) return;
      if constexpr (std::is_trivially_copyable_v<T>) {
        memcpy<address_space, address_space>((void *)_data, (void *)h.data(), _size * sizeof(T));
      } else {
        for (size_t i = 0; i < _size; ++i) new (_data + i) T(h[i]);
      }
    }

    /**
     * @brief Copy assignment operator first cleans up the current handle and then makes a deep copy of the data from
     * the source handle.
     *
     * @param h Source handle.
     */
    handle_sso &operator=(handle_sso const &h) {
      if (this == &h) return *this; // self assignment
      clean();
      _size = h._size;
      if (_size == 0) return *this;
      if (on_heap()) {
        auto b = mallocator<>::allocate(_size * sizeof(T));
        if (not b.ptr) throw std::bad_alloc{};
        _data = (T *)b.ptr;
      } else {
        _data = (T *)buffer.data();
      }
      for (size_t i = 0; i < _size; ++i) new (_data + i) T(h[i]);
      return *this;
    }

    /**
     * @brief Construct a handle by making a deep copy of the data from another owning handle.
     * @details Depending on the size, the memory of the data is either allocated on the heap or on the stack.
     * @param h Source handle.
     */
    template <OwningHandle<value_type> H>
    explicit handle_sso(H const &h) : handle_sso(h.size(), do_not_initialize) {
      if (is_null()) return;
      if constexpr (std::is_trivially_copyable_v<T>) {
        memcpy<address_space, H::address_space>((void *)_data, (void *)h.data(), _size * sizeof(T));
      } else {
        static_assert(address_space == H::address_space,
                      "Constructing an nda::mem::handle_sso from a handle of a different address space requires a trivially copyable value_type");
        for (size_t i = 0; i < _size; ++i) new (_data + i) T(h[i]);
      }
    }

    /**
     * @brief Construct a handle for the data of a given size and do not initialize it.
     * @details Depending on the size, the memory of the data is either allocated on the heap or on the stack.
     * @param size Size of the data (number of elements).
     */
    handle_sso(long size, do_not_initialize_t) {
      if (size == 0) return;
      _size = size;
      if (not on_heap()) {
        _data = (T *)buffer.data();
      } else {
        auto b = mallocator<>::allocate(size * sizeof(T));
        if (not b.ptr) throw std::bad_alloc{};
        _data = (T *)b.ptr;
      }
    }

    /**
     * @brief Construct a handle for the data of a given size and initialize it to zero (only for scalar and complex
     * types).
     *
     * @details Depending on the size, the memory of the data is either allocated on the heap or on the stack.
     *
     * @param size Size of the data (number of elements).
     */
    handle_sso(long size, init_zero_t) {
      static_assert(std::is_scalar_v<T> or is_complex_v<T>, "nda::mem::handle_sso can only be initialized to zero for scalar and complex types");
      if (size == 0) return;
      _size = size;
      if (not on_heap()) {
        _data = (T *)buffer.data();
        for (size_t i = 0; i < _size; ++i) data()[i] = 0;
      } else {
        auto b = mallocator<>::allocate_zero(size * sizeof(T)); //, alignof(T));
        if (not b.ptr) throw std::bad_alloc{};
        _data = (T *)b.ptr;
      }
    }

    /**
     * @brief Construct a handle for the data of a given size and initialize it depending on the value type.
     *
     * @details The data is initialized as follows:
     * - If `T` is std::complex and nda::mem::init_dcmplx is true, the data is initialized to zero.
     * - If `T` is not trivial and not complex, the data is default constructed by placement new operator calls.
     * - Otherwise, the data is not initialized.
     *
     * @param size Size of the data (number of elements).
     */
    handle_sso(long size) {
      if (size == 0) return;
      _size = size;
      if (not on_heap()) {
        _data = (T *)buffer.data();
      } else {
        blk_t b;
        if constexpr (is_complex_v<T> && init_dcmplx)
          b = mallocator<>::allocate_zero(size * sizeof(T));
        else
          b = mallocator<>::allocate(size * sizeof(T));
        if (not b.ptr) throw std::bad_alloc{};
        _data = (T *)b.ptr;
      }

      // call placement new for non trivial and non complex types
      if constexpr (!std::is_trivial_v<T> and !is_complex_v<T>) {
        for (size_t i = 0; i < size; ++i) new (_data + i) T();
      }
    }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Reference to the element at the given index.
     */
    [[nodiscard]] T &operator[](long i) noexcept { return _data[i]; }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Const reference to the element at the given index.
     */
    [[nodiscard]] T const &operator[](long i) const noexcept { return _data[i]; }

    /**
     * @brief Check if the data is/should be stored on the heap.
     * @return True if the size is greater than the `Size`.
     */
    [[nodiscard]] bool on_heap() const { return _size > Size; }

    /**
     * @brief Check if the handle is in a null state.
     * @return True if the data is a `nullptr` (and the size is 0).
     */
    [[nodiscard]] bool is_null() const noexcept {
#ifdef NDA_DEBUG
      EXPECTS((_data == nullptr) == (_size == 0));
#endif
      return _data == nullptr;
    }

    /**
     * @brief Get a pointer to the stored data.
     * @return Pointer to the start of the handled memory.
     */
    [[nodiscard]] T *data() const noexcept { return _data; }

    /**
     * @brief Get the size of the handle.
     * @return Number of elements of type `T` in the handled memory.
     */
    [[nodiscard]] long size() const noexcept { return _size; }
  };

  /**
   * @brief A handle for a memory block on the heap with shared ownership.
   *
   * @tparam T Value type of the data.
   * @tparam AdrSp nda::mem::AddressSpace in which the memory is allocated.
   */
  template <typename T, AddressSpace AdrSp>
  struct handle_shared {
    static_assert(std::is_nothrow_destructible_v<T>, "nda::mem::handle_shared requires the value_type to have a non-throwing destructor");

    private:
    // Pointer to the start of the actual data.
    T *_data = nullptr;

    // Size of the data (number of T elements). Invariant: size > 0 iif data != 0.
    size_t _size = 0;

    // Type of the memory block, i.e. a pointer to the data and its size.
    using blk_t = std::pair<T *, size_t>;

    // For shared ownership (points to a blk_T_t).
    std::shared_ptr<void> sptr;

    public:
    /// Value type of the data.
    using value_type = T;

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = AdrSp;

    /// Default constructor leaves the handle in a null state (`nullptr` and size 0).
    handle_shared() = default;

    /**
     * @brief Construct a handle from a shared object from a foreign library.
     *
     * @param data Pointer to the start of the shared data.
     * @param size Size of the data (number of elements).
     * @param foreign_handle Pointer to the shared object.
     * @param foreign_decref Function to decrease the reference count of the shared object.
     */
    handle_shared(T *data, size_t size, void *foreign_handle, void (*foreign_decref)(void *)) noexcept
       : _data(data), _size(size), sptr{foreign_handle, foreign_decref} {}

    /**
     * @brief Construct a shared handle from an nda::mem::handle_heap.
     *
     * @tparam A nda::mem::Allocator type of the source handle.
     * @param h Source handle.
     */
    template <Allocator A>
    handle_shared(handle_heap<T, A> const &h) noexcept
      requires(A::address_space == address_space)
       : _data(h.data()), _size(h.size()) {
      if (not h.is_null()) sptr = h.get_sptr();
    }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Reference to the element at the given index.
     */
    [[nodiscard]] T &operator[](long i) noexcept { return _data[i]; }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Const reference to the element at the given index.
     */
    [[nodiscard]] T const &operator[](long i) const noexcept { return _data[i]; }

    /**
     * @brief Check if the handle is in a null state.
     * @return True if the data is a `nullptr` (and the size is 0).
     */
    [[nodiscard]] bool is_null() const noexcept {
#ifdef NDA_DEBUG
      // Check the Invariants in Debug Mode
      EXPECTS((_data == nullptr) == (_size == 0));
#endif
      return _data == nullptr;
    }

    /**
     * @brief Get the reference count of the shared object.
     * @return Reference count of the shared pointer.
     */
    [[nodiscard]] long refcount() const noexcept { return sptr.use_count(); }

    /**
     * @brief Get a pointer to the stored data.
     * @return Pointer to the start of the handled memory.
     */
    [[nodiscard]] T *data() const noexcept { return _data; }

    /**
     * @brief Get the size of the handle.
     * @return Number of elements of type `T` in the handled memory.
     */
    [[nodiscard]] long size() const noexcept { return _size; }
  };

  /**
   * @brief A non-owning handle for a memory block on the heap.
   *
   * @tparam T Value type of the data.
   * @tparam AdrSp nda::mem::AddressSpace in which the memory is allocated.
   */
  template <typename T, AddressSpace AdrSp = Host>
  struct handle_borrowed {
    private:
    // Value type of the data with const removed.
    using T0 = std::remove_const_t<T>;

    // Parent handle (required for regular -> shared promotion in Python Converter).
    handle_heap<T0> const *_parent = nullptr;

    // Pointer to the start of the actual data.
    T *_data = nullptr;

    public:
    /// Value type of the data.
    using value_type = T;

    /// nda::mem::AddressSpace in which the memory is allocated.
    static constexpr auto address_space = AdrSp;

    /// Default constructor leaves the handle in a null state (nullptr).
    handle_borrowed() = default;

    /// Default move assignment operator.
    handle_borrowed &operator=(handle_borrowed &&) = default;

    /// Default copy constructor.
    handle_borrowed(handle_borrowed const &) = default;

    /// Default copy assignment operator.
    handle_borrowed &operator=(handle_borrowed const &) = default;

    /**
     * @brief Construct a borrowed handle from a pointer to the data.
     * @param ptr Pointer to the start of the data.
     */
    handle_borrowed(T *ptr) noexcept : _data(ptr) {}

    /**
     * @brief Construct a borrowed handle from a another handle.
     *
     * @tparam H nda::mem::Handle type.
     * @param h Other handle.
     * @param offset Pointer offset from the start of the data (in number of elements).
     */
    template <Handle H>
      requires(address_space == H::address_space and (std::is_const_v<value_type> or !std::is_const_v<typename H::value_type>)
               and std::is_same_v<const value_type, const typename H::value_type>)
    handle_borrowed(H const &h, long offset = 0) noexcept : _data(h.data() + offset) {
      if constexpr (std::is_same_v<H, handle_heap<T0>>) _parent = &h;
    }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Reference to the element at the given index.
     */
    [[nodiscard]] T &operator[](long i) noexcept { return _data[i]; }

    /**
     * @brief Subscript operator to access the data.
     *
     * @param i Index of the element to access.
     * @return Const reference to the element at the given index.
     */
    [[nodiscard]] T const &operator[](long i) const noexcept { return _data[i]; }

    /**
     * @brief Check if the handle is in a null state.
     * @return True if the data is a `nullptr`.
     */
    [[nodiscard]] bool is_null() const noexcept { return _data == nullptr; }

    /**
     * @brief Get a pointer to the parent handle.
     * @return Pointer to the parent handle.
     */
    [[nodiscard]] handle_heap<T0> const *parent() const { return _parent; }

    /**
     * @brief Get a pointer to the stored data.
     * @return Pointer to the start of the handled memory.
     */
    [[nodiscard]] T *data() const noexcept { return _data; }
  };

  /** @} */

} // namespace nda::mem
