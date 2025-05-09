// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides definitions and type traits involving the different memory address
 * spaces supported by nda.
 */

#pragma once

#include "../concepts.hpp"
#include "../device.hpp"

#include <algorithm>

namespace nda {

  /// @cond
  // Forward declarations
  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  struct expr;
  template <typename F, Array... As>
  struct expr_call;
  template <char OP, Array A>
  struct expr_unary;
  /// @endcond

} // namespace nda

namespace nda::mem {

  /**
   * @addtogroup mem_addrspcs
   * @{
   */

  /**
   * @brief Enum providing identifiers for the different memory address spaces.
   *
   * @details The following address spaces are available:
   * - `None`: No address space.
   * - `Host`: Address on CPU RAM.
   * - `Device`: Address on GPU memory.
   * - `Unified`: CUDA Unified memory address.
   */
  enum class AddressSpace { None, Host, Device, Unified }; // Do not change order!

  /// Using declaration for the `Device` address space (see nda::mem::AddressSpace).
  using AddressSpace::Device;

  /// Using declaration for the `Host` address space (see nda::mem::AddressSpace).
  using AddressSpace::Host;

  /// Using declaration for the `None` address space (see nda::mem::AddressSpace).
  using AddressSpace::None;

  /// Using declaration for the `Unified` address space (see nda::mem::AddressSpace).
  using AddressSpace::Unified;

  /**
   * @brief Variable template providing the address space for different types.
   * @details Has to be specialized for each type/concept separately.
   * @tparam T Type for which to get the address space.
   */
  template <typename T>
  static constexpr AddressSpace get_addr_space = mem::None;

  // Specialization of nda::mem::get_addr_space for const types.
  template <typename T>
  static constexpr AddressSpace get_addr_space<T const> = get_addr_space<T>;

  // Specialization of nda::mem::get_addr_space for reference types.
  template <typename T>
  static constexpr AddressSpace get_addr_space<T &> = get_addr_space<T>;

  /**
   * @brief Promotion rules for nda::mem::AddressSpace values.
   *
   * @details `Host` and `Device` address spaces are not compatible and will result in a compilation error.
   *
   * The promotion rules are as follows:
   * - `None` -> `Host` -> `Unified`.
   * - `None` -> `Device` -> `Unified`.
   *
   * @tparam A1 First address space.
   * @tparam A2 Second address space.
   * @tparam As Remaining address spaces.
   */
  template <AddressSpace A1, AddressSpace A2 = None, AddressSpace... As>
  constexpr AddressSpace combine = []() {
    static_assert(!(A1 == Host && A2 == Device) && !(A1 == Device && A2 == Host),
                  "Error in nda::mem::combine: Cannot combine Host and Device address spaces");
    if constexpr (sizeof...(As) > 0) { return combine<std::max(A1, A2), As...>; }
    return std::max(A1, A2);
  }();

  /**
   * @brief Get common address space for a number of given nda::Array types.
   *
   * @details See nda::mem::combine for how the address spaces are combined.
   *
   * @tparam A1 nda::Array type.
   * @tparam As nda::Array types.
   */
  template <Array A1, Array... As>
  constexpr AddressSpace common_addr_space = combine<get_addr_space<A1>, get_addr_space<As>...>;

  /// Specialization of nda::mem::get_addr_space for nda::Memory Array types.
  template <MemoryArray A>
  static constexpr AddressSpace get_addr_space<A> = A::storage_t::address_space;

  /// Specialization of nda::mem::get_addr_space for nda::Handle types.
  template <Handle H>
  static constexpr AddressSpace get_addr_space<H> = H::address_space;

  /// Specialization of nda::mem::get_addr_space for binary expressions involving two nda::ArrayOrScalar types.
  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  static constexpr AddressSpace get_addr_space<expr<OP, L, R>> = combine<get_addr_space<L>, get_addr_space<R>>;

  /// Specialization of nda::mem::get_addr_space for function call expressions involving nda::Array types.
  template <typename F, Array... As>
  static constexpr AddressSpace get_addr_space<expr_call<F, As...>> = combine<get_addr_space<As>...>;

  /// Specialization of nda::mem::get_addr_space for unary expressions involving an nda::Array type.
  template <char OP, Array A>
  static constexpr AddressSpace get_addr_space<expr_unary<OP, A>> = get_addr_space<A>;

  /**
   * @brief Check validity of a set of nda::mem::AddressSpace values.
   *
   * @details Checks that the address spaces are not `None` and that the `Device` or `Unified` address spaces are only
   * used when compiling with GPU support.
   *
   * @tparam AdrSpcs Address spaces to check.
   */
  template <AddressSpace... AdrSpcs>
  static const auto check_adr_sp_valid = []() {
    static_assert(((AdrSpcs != None) & ...), "Error in nda::mem::check_adr_sp_valid: Cannot use None address space");
    static_assert(nda::have_device or ((AdrSpcs == Host) & ...),
                  "Error in nda::mem::check_adr_sp_valid: Device address space requires compiling with GPU support.");
  };

  /// Constexpr variable that is true if all given types have a `Host` address space.
  template <typename... Ts>
    requires(sizeof...(Ts) > 0)
  static constexpr bool on_host = ((get_addr_space<Ts> == mem::Host) and ...);

  /// Constexpr variable that is true if all given types have a `Device` address space.
  template <typename... Ts>
    requires(sizeof...(Ts) > 0)
  static constexpr bool on_device = ((get_addr_space<Ts> == mem::Device) and ...);

  /// Constexpr variable that is true if all given types have a `Unified` address space.
  template <typename... Ts>
    requires(sizeof...(Ts) > 0)
  static constexpr bool on_unified = ((get_addr_space<Ts> == mem::Unified) and ...);

  /// Constexpr variable that is true if all given types have the same address space.
  template <typename A0, typename... A>
  static constexpr bool have_same_addr_space = ((get_addr_space<A0> == get_addr_space<A>) and ... and true);

  /// Constexpr variable that is true if all given types have an address space compatible with `Host`.
  template <typename... Ts>
  static constexpr bool have_host_compatible_addr_space = ((on_host<Ts> or on_unified<Ts>) and ...);

  /// Constexpr variable that is true if all given types have an address space compatible with `Device`.
  template <typename... Ts>
  static constexpr bool have_device_compatible_addr_space = ((on_device<Ts> or on_unified<Ts>) and ...);

  /// Constexpr variable that is true if all given types have compatible address spaces.
  template <typename... Ts>
  static constexpr bool have_compatible_addr_space = (have_host_compatible_addr_space<Ts...> or have_device_compatible_addr_space<Ts...>);

  // Test various combinations of address spaces.
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

  /** @} */

} // namespace nda::mem
