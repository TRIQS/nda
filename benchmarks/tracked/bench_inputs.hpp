// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#include "./benchmark_concepts.hpp"
#include <nda/nda.hpp>
#include <array>
#include <complex>
#include <concepts>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace nda_bench {
  // Tag used in the benchmark name.
  template <typename ValueType>
  consteval std::string_view type_tag() {
    if constexpr (std::same_as<ValueType, float>) {
      return "f32";
    } else if constexpr (std::same_as<ValueType, double>) {
      return "f64";
    } else if constexpr (std::same_as<ValueType, std::complex<float>>) {
      return "c64";
    } else if constexpr (std::same_as<ValueType, std::complex<double>>) {
      return "c128";
    } else {
      return "Unknown";
    }
  }

  template <typename Layout>
  consteval std::string_view layout_tag() {
    if constexpr (std::same_as<Layout, nda::C_layout>) {
      return "C_layout";
    } else if constexpr (std::same_as<Layout, nda::F_layout>) {
      return "F_layout";
    } else {
      return "Unknown";
    }
  }

  // Each input selects bounds for the range its expression needs.
  // The default keeps values away from zero so division, log and sqrt
  // behave, but that is wrong for asin/acos (domain [-1,1]) and product (overflows), which
  // override it.
  struct input_range {
    double lower = 1.0;
    double upper = 2.0; // default: [1, 2)
  };

  struct product_band {
    static constexpr input_range value{.lower = 0.5 - 5e-8, .upper = 0.5 + 5e-8};
  };
  struct signed_band {
    static constexpr input_range value{.lower = -1.0, .upper = 1.0};
  };
  struct positive_band {
    static constexpr input_range value{};
  };

  template <typename ValueType>
  ValueType random_value(std::mt19937_64 &rng, input_range const &range) {
    constexpr bool supported = nda::is_blas_lapack_v<ValueType>;
    static_assert(supported, "Benchmark input generation supports only float, double, std::complex<float>, and std::complex<double>");
    if constexpr (nda::is_complex_v<ValueType>) {
      using RealType = nda::remove_complex_t<ValueType>;
      return ValueType{random_value<RealType>(rng, range),
                       random_value<RealType>(rng, input_range{.lower = 0.0, .upper = range.upper - range.lower})};
    } else {
      return std::uniform_real_distribution<ValueType>{static_cast<ValueType>(range.lower), static_cast<ValueType>(range.upper)}(rng);
    }
  }

  template <nda::ArrayOrScalar OperandType>
  std::int64_t array_elements(OperandType const &x) {
    if constexpr (nda::Array<OperandType>) {
      return x.size();
    } else {
      return 0;
    }
  }

  template <nda::ArrayOrScalar OperandType>
  std::int64_t logical_bytes(OperandType const &x) {
    return array_elements(x) * static_cast<std::int64_t>(sizeof(nda::get_value_t<OperandType>));
  }

  template <std::int32_t Rank, char Algebra = 'A', typename Layout = nda::C_layout, InputBand Band = positive_band>
  struct array_input {
    static_assert(Rank > 0 && (Algebra == 'A' || (Algebra == 'M' && Rank == 2) || (Algebra == 'V' && Rank == 1)),
                  "Input rank must be positive; algebra must be A, M (rank 2), or V (rank 1)");

    static constexpr std::string_view name = Algebra == 'M' ? "M" : Algebra == 'V' ? "V" : "A";

    static constexpr char algebra = Algebra;

    template <typename ValueType>
    using storage_type = nda::basic_array<ValueType, Rank, Layout, Algebra, nda::heap<>>;

    template <typename ValueType>
    static std::string id() {
      return std::string(name) + std::to_string(Rank) + "_" + std::string(layout_tag<Layout>());
    }

    template <typename ValueType>
    static constexpr std::int64_t storage_size(std::int64_t N) {
      std::int64_t n = 1;
      for (std::int32_t i = 0; i < Rank; ++i) { n *= N; }
      return n;
    }

    template <typename ValueType>
    static auto generate(std::int64_t N, std::mt19937_64 &rng) {
      std::array<std::int64_t, Rank> shape;
      shape.fill(N);
      storage_type<ValueType> result(shape);
      for (auto &x : result) { x = random_value<ValueType>(rng, Band::value); }
      return result;
    }

    template <typename StorageType>
    static auto &get(StorageType &storage) {
      return storage;
    }
  };

  template <typename Layout = nda::C_layout, InputBand Band = positive_band>
  using matrix_input = array_input<2, 'M', Layout, Band>;

  template <typename Layout = nda::C_layout, InputBand Band = positive_band>
  using vector_input = array_input<1, 'V', Layout, Band>;

  template <InputBand Band = positive_band>
  struct scalar_input {
    static constexpr std::string_view name = "S";

    static constexpr char algebra = 'S';

    template <typename ValueType>
    using storage_type = ValueType;

    template <typename ValueType>
    static std::string id() {
      return std::string(name);
    }

    template <typename ValueType>
    static constexpr std::int64_t storage_size(std::int64_t) {
      return 0;
    }

    template <typename ValueType>
    static ValueType generate(std::int64_t, std::mt19937_64 &rng) {
      return random_value<ValueType>(rng, Band::value);
    }

    template <typename StorageType>
    static auto &get(StorageType &storage) {
      return storage;
    }
  };

  template <typename Input, std::int32_t Axis, std::int64_t Step, std::int64_t Start = 0>
  struct slice_input : Input {
    static_assert(Axis >= 0 && Step > 0 && Start >= 0, "Slice axis/start must be nonnegative and step positive");

    static constexpr std::string_view name = "slice";

    template <typename ValueType>
    static std::string id() {
      return Input::template id<ValueType>() + "." + std::string(name) + "(axis=" + std::to_string(Axis) + ",start=" + std::to_string(Start)
         + ",step=" + std::to_string(Step) + ")";
    }

    template <typename StorageType>
    static auto get(StorageType &storage) {
      auto &&a = Input::get(storage);
      // A temporary owning array would be destroyed on return, leaving the slice dangling.
      static_assert(!nda::is_regular_v<std::remove_cvref_t<decltype(a)>> || std::is_lvalue_reference_v<decltype(a)>,
                    "Input::get must return owning arrays by lvalue reference when used with slice_input");
      static_assert(Axis < nda::get_rank<decltype(a)>, "Slice axis exceeds input rank");
      if (Start >= a.extent(Axis)) { throw std::invalid_argument("Slice is empty at this benchmark size"); }
      return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return a(nda::range(I == Axis ? Start : 0, a.extent(I), I == Axis ? Step : 1)...);
      }(std::make_index_sequence<nda::get_rank<decltype(a)>>{});
    }
  };

  template <typename Input, std::int64_t Step = 2, std::int64_t Start = 0>
  using row_slice = slice_input<Input, 0, Step, Start>;

  template <typename Input, std::int64_t Step = 2, std::int64_t Start = 0>
  using col_slice = slice_input<Input, 1, Step, Start>;

} // namespace nda_bench
