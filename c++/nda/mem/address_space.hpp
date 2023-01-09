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
// Authors: Nils Wentzell

#pragma once
#include <algorithm>

#include "../concepts.hpp"

namespace nda {
  // Forward declarations
  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  struct expr;
  template <typename F, Array... As>
  struct expr_call;
  template <char OP, Array A>
  struct expr_unary;
} // namespace nda

namespace nda::mem {

  /**
   * Enum providing idientifiers for the different Memory Address-Spaces 
   *
   * None    -> No address in memory
   * Host    -> Address in RAM
   * Device  -> Address on GPU memory
   * Unified -> CUDA Unified memory address
   */
  enum class AddressSpace { None, Host, Device, Unified };
  using AddressSpace::Device;
  using AddressSpace::Host;
  using AddressSpace::None;
  using AddressSpace::Unified;

  // Variable template providing the address space for different types
  // To be specialized for each case / concept
  template <typename T>
  static constexpr AddressSpace get_addr_space = mem::None;
  template <typename T>
  static constexpr AddressSpace get_addr_space<T &> = get_addr_space<T>;

  // Promotion rules for the AddressSpace in unnary and binary operations (e.g. +)
  template <AddressSpace A1, AddressSpace A2 = None, AddressSpace... AN>
  constexpr AddressSpace combine = []() {
    static_assert(!(A1 == Host && A2 == Device) && !(A1 == Device && A2 == Host),
                  "Cannot promote AddressSpace for Op(Host, Device) or Op(Device, Host)");
    if constexpr (sizeof...(AN) > 0) { return combine<std::max(A1, A2), AN...>; }
    return std::max(A1, A2);
  }();

  // -------------  Specializations -------------

  template <MemoryArray A>
  static constexpr AddressSpace get_addr_space<A> = A::storage_t::address_space;

  template <Handle H>
  static constexpr AddressSpace get_addr_space<H> = H::address_space;

  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  static constexpr AddressSpace get_addr_space<expr<OP, L, R>> = combine<get_addr_space<L>, get_addr_space<R>>;

  template <typename F, Array... As>
  static constexpr AddressSpace get_addr_space<expr_call<F, As...>> = combine<get_addr_space<As>...>;

  template <char OP, Array A>
  static constexpr AddressSpace get_addr_space<expr_unary<OP, A>> = get_addr_space<A>;

  // ------------- Additional helper traits -------------

  // Check if all Ts have memory on the host
  template <typename... Ts>
  requires(sizeof...(Ts) > 0)
  static constexpr bool on_host = ((get_addr_space<Ts> == mem::Host) and ...);

  // Check if all Ts have memory on the device
  template <typename... Ts>
  requires(sizeof...(Ts) > 0)
  static constexpr bool on_device = ((get_addr_space<Ts> == mem::Device) and ...);

  // Check if all Ts have unified memory
  template <typename... Ts>
  requires(sizeof...(Ts) > 0)
  static constexpr bool on_unified = ((get_addr_space<Ts> == mem::Unified) and ...);

  // Check all A have the same address space 
  template <typename A0, typename... A>
  static constexpr bool have_same_addr_space_v = ((get_addr_space<A0> == get_addr_space<A>)and... and true);

  // Check all Ts are host compatible 
  template <typename... Ts>
  static constexpr bool have_host_compatible_addr_space_v = ((on_host<Ts> or on_unified<Ts>) and ...);

  // Check all Ts are device compatible 
  template <typename... Ts>
  static constexpr bool have_device_compatible_addr_space_v = ((on_device<Ts> or on_unified<Ts>) and ...);

  // Check all Ts have compatible address spaces
  template <typename... Ts>
  static constexpr bool have_compatible_addr_space_v = (have_host_compatible_addr_space_v<Ts...> or have_device_compatible_addr_space_v<Ts...>);

  // ------------- Test Promotion for various cases ------------

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
} // namespace nda::mem
