// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#include <nda/nda.hpp>
#include <concepts>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>

namespace nda_bench {

  struct input_range;

  template <typename Band>
  concept InputBand = requires {
    requires std::same_as<std::remove_cvref_t<decltype(Band::value)>, input_range>;           // Provides an input_range.
    typename std::integral_constant<std::remove_cvref_t<decltype(Band::value)>, Band::value>; // The range is a compile-time constant.
  };

  template <typename Input, typename ValueType>
  concept BenchmarkInput = requires(typename Input::template storage_type<ValueType> &storage, std::int64_t N, std::mt19937_64 &rng) {
    typename Input::template storage_type<ValueType>; // Storage type exists for this numeric type.
    {
      Input::template generate<ValueType>(N, rng)
    } -> std::same_as<typename Input::template storage_type<ValueType>>;          // Returns the declared storage type.
    { Input::get(storage) } -> nda::ArrayOrScalar;                                // Exposes an array or scalar operand.
    { Input::template storage_size<ValueType>(N) } -> std::same_as<std::int64_t>; // Reports storage element count.
    { Input::template id<ValueType>() } -> std::same_as<std::string>;             // Provides the operand ID used in benchmark names.
    requires std::same_as<std::remove_cvref_t<decltype(Input::algebra)>, char>;   // Algebra has type char.
    typename std::integral_constant<char, Input::algebra>;                        // Algebra is a compile-time constant.
  };

  template <typename Operation, typename... OperandTypes>
  concept BenchmarkOperation =
     (nda::ArrayOrScalar<std::remove_cvref_t<OperandTypes>> && ...) && requires(std::remove_cvref_t<OperandTypes> const &...operands) {
       // Support flags must be compile-time booleans.
       requires std::same_as<std::remove_cvref_t<decltype(Operation::supports_complex)>, bool>;
       requires std::same_as<std::remove_cvref_t<decltype(Operation::supports_rank2)>, bool>;
       requires std::same_as<std::remove_cvref_t<decltype(Operation::supports_rank3)>, bool>;
       requires std::same_as<std::remove_cvref_t<decltype(Operation::supports_matrix)>, bool>;
       typename std::bool_constant<Operation::supports_complex>;
       typename std::bool_constant<Operation::supports_rank2>;
       typename std::bool_constant<Operation::supports_rank3>;
       typename std::bool_constant<Operation::supports_matrix>;
       { Operation::op(operands...) } -> nda::ArrayOrScalar; // Accepts the actual const operands and returns an array or scalar.
       Operation::result_shape(operands...);                 // Provides shape discovery for these operands.
     };

} // namespace nda_bench
