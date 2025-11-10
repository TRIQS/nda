// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic malloc and free function for different address spaces.
 */

#pragma once

#include "./address_space.hpp"
#include "../device.hpp"

#ifdef NDA_HAVE_MPI
#include <mpi/communicator.hpp>
#endif

#include <cstdlib>

namespace nda::mem {

  /**
   * @addtogroup mem_utils
   * @{
   */

#ifdef NDA_HAVE_MPI
  /**
   * @class mpi_shm
   * @brief Manages the global MPI shared communicator for shared memory allocation.
   *
   * This class provides a mechanism for retrieving and setting the global
   * `mpi::shared_communicator` instance used for MPI shared memory allocation.
   * It ensures that all components accessing shared memory use the same communicator.
   *
   * @note This class is not thread-safe. Concurrent modifications may lead to undefined behavior.
   */
  class mpi_shm { /// mpi_shm + default constructor in private
    /**
     * @brief Return reference to the singleton for the global MPI shared communicator instance of the MPI shared memory allocator.
     *
     * @warning This function is not thread-safe.
     */
    static mpi::shared_communicator &_impl_communicator() {
      static mpi::shared_communicator shm = mpi::communicator{}.split_shared();
      return shm;
    }

    public:
    /**
     * @brief Return the global MPI shared communicator instance of the MPI shared memory allocator.
     *
     * @warning This function is not thread-safe.
     */
    inline static mpi::shared_communicator get_communicator() { return _impl_communicator(); }

    /**
      * @brief Set the global MPI shared communicator instance of the MPI shared memory allocator.
      *
      * @warning This function is not thread-safe.
      */
    inline static void set_communicator(mpi::shared_communicator const &shm) { _impl_communicator() = shm; }
  };
#endif

  /**
   * @brief Call the correct `malloc` function based on the given address space.
   *
   * @details It makes the following function calls depending on the address space:
   * - `std::malloc` for `Host`.
   * - `cudaMalloc` for `Device`.
   * - `cudaMallocManaged` for `Unified`.
   *
   * @tparam AdrSp nda::mem::AddressSpace.
   * @param size Size in bytes to be allocated.
   * @return Pointer to the allocated memory.
   */
  template <AddressSpace AdrSp>
  void *malloc(size_t size) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    void *ptr = nullptr;
    if constexpr (AdrSp == Host) {
      ptr = std::malloc(size); // NOLINT (we want to return a void*)
    } else if constexpr (AdrSp == Device) {
      device_error_check(cudaMalloc((void **)&ptr, size), "cudaMalloc");
    } else if constexpr (AdrSp == Unified) {
      device_error_check(cudaMallocManaged((void **)&ptr, size), "cudaMallocManaged");
    } else {
      static_assert(false, "Not implemented!");
    }
    return ptr;
  }

  /**
   * @brief Call the correct `free` function based on the given address space.
   *
   * @details It makes the following function calls depending on the address space:
   * - `std::free` for `Host`.
   * - `cudaFree` for `Device` and `Unified`.
   *
   * @tparam AdrSp nda::mem::AddressSpace.
   * @param p Pointer to the memory to be freed.
   */
  template <AddressSpace AdrSp>
  void free(void *p) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (AdrSp == Host) {
      std::free(p); // NOLINT (we want to call free with a void*)
    } else if (AdrSp == Device || AdrSp == Unified) {
      device_error_check(cudaFree(p), "cudaFree");
    } else {
      static_assert(false, "Not implemented!");
    }
  }

  /** @} */

} // namespace nda::mem
