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
  template <typename F, Array... A>
  struct expr_call;
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

  template <char OP, typename L, typename R>
  static constexpr AddressSpace get_addr_space<expr<OP, L, R>> = combine<get_addr_space<L>, get_addr_space<R>>;

  template <typename F, typename... A>
  static constexpr AddressSpace get_addr_space<expr_call<F, A...>> = combine<get_addr_space<A>...>;

  // ------------- Additional helper traits -------------

  template <typename T>
  static constexpr bool on_host = (get_addr_space<T> == mem::Host);

  template <typename T>
  static constexpr bool on_device = (get_addr_space<T> == mem::Device || get_addr_space<T> == mem::Unified);

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
