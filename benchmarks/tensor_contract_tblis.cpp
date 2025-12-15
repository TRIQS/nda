// Copyright (c) 2025 The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include <benchmark/benchmark.h>
#include <nda/nda.hpp>
#include <nda/tensor.hpp>
#include <nda/traits.hpp>
#include <nda/tensor/interface/tblis_interface.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#if !defined(NDA_HAVE_TBLIS)

static void TblisNotAvailable(benchmark::State &state) { state.SkipWithError("nda::tensor::contract benchmark requires TBLIS support"); }
BENCHMARK(TblisNotAvailable);

#else

namespace {

  template <typename Value, bool = nda::is_complex_v<Value>>
  struct real_type_helper {
    using type = Value;
  };

  template <typename Value>
  struct real_type_helper<Value, true> {
    using type = typename Value::value_type;
  };

  template <typename Value>
  using real_type_t = typename real_type_helper<Value>::type;

  template <typename Value>
  Value make_value(long i, long j, long k) {
    using real_t     = real_type_t<Value>;
    real_t real_part = real_t((i + 1) * 13 + (j + 1) * 7 + (k + 1) * 5);
    real_part /= real_t{64};
    if constexpr (nda::is_complex_v<Value>) {
      real_t imag_part = real_t((i + 1) * 11 + (j + 1) * 3 + (k + 1) * 17);
      imag_part /= real_t{128};
      return Value{real_part, imag_part};
    } else {
      return static_cast<Value>(real_part);
    }
  }

  template <typename Value>
  auto max_abs_difference(nda::matrix<Value> const &lhs, nda::matrix<Value> const &rhs) {
    using real_t   = real_type_t<Value>;
    real_t max_err = real_t{0};
    for (long i = 0; i < lhs.extent(0); ++i) {
      for (long j = 0; j < lhs.extent(1); ++j) {
        real_t delta = static_cast<real_t>(std::abs(lhs(i, j) - rhs(i, j)));
        if (delta > max_err) max_err = delta;
      }
    }
    return max_err;
  }

  template <typename Value>
  auto max_abs_value(nda::matrix<Value> const &m) {
    using real_t   = real_type_t<Value>;
    real_t max_val = real_t{0};
    for (long i = 0; i < m.extent(0); ++i)
      for (long j = 0; j < m.extent(1); ++j) {
        auto magnitude = static_cast<real_t>(std::abs(m(i, j)));
        if (magnitude > max_val) max_val = magnitude;
      }
    return max_val;
  }

  template <typename Value>
  void fill_inputs(nda::array<Value, 3> &a, nda::array<Value, 3> &b) {
    for (long i = 0; i < a.extent(0); ++i)
      for (long k = 0; k < a.extent(1); ++k)
        for (long l = 0; l < a.extent(2); ++l) a(i, k, l) = make_value<Value>(i, k, l);

    for (long k = 0; k < b.extent(0); ++k)
      for (long j = 0; j < b.extent(1); ++j)
        for (long l = 0; l < b.extent(2); ++l) b(k, j, l) = make_value<Value>(k, j, l + 1);
  }

  template <typename Value>
  void compute_reference(nda::array<Value, 3> const &a, nda::array<Value, 3> const &b, nda::matrix<Value> &ref) {
    ref() = Value{0};
    for (long i = 0; i < a.extent(0); ++i) {
      for (long j = 0; j < b.extent(1); ++j) {
        Value acc = Value{0};
        for (long k = 0; k < a.extent(1); ++k)
          for (long l = 0; l < a.extent(2); ++l) acc += a(i, k, l) * b(k, j, l);
        ref(i, j) = acc;
      }
    }
  }

  template <typename Value>
  void tblis_contract(Value alpha, nda::array<Value, 3> const &a, std::string_view idx_a, nda::array<Value, 3> const &b, std::string_view idx_b,
                      Value beta, nda::matrix<Value> &c, std::string_view idx_c) {
    using nda::tensor::nda_tblis::tensor;
    tensor<Value, 3> a_t(a, alpha);
    tensor<Value, 3> b_t(b);
    tensor<Value, 2> c_t(c, beta);
    ::tblis::tblis_tensor_mult(nullptr, nullptr, &a_t, idx_a.data(), &b_t, idx_b.data(), &c_t, idx_c.data());
  }

  template <typename Value>
  static void TblisTensorContract(benchmark::State &state) {
    auto const I = state.range(0);
    auto const J = state.range(1);
    auto const K = state.range(2);
    auto const L = state.range(3);

    nda::array<Value, 3> a(I, K, L);
    nda::array<Value, 3> b(K, J, L);
    nda::matrix<Value> result(I, J);
    nda::matrix<Value> reference(I, J);

    fill_inputs(a, b);
    compute_reference(a, b, reference);

    result() = Value{0};
    tblis_contract(Value{1}, a, "ikl", b, "kjl", Value{0}, result, "ij");

    auto const max_err = max_abs_difference(result, reference);
    using real_t       = real_type_t<Value>;
    auto const ref_amp = max_abs_value(reference);
    real_t tolerance   = std::numeric_limits<real_t>::epsilon() * real_t{500} * std::max(real_t{1}, ref_amp);
    if constexpr (std::is_same_v<real_t, float>)
      tolerance = std::max(tolerance, real_t{5e-3});
    else
      tolerance = std::max(tolerance, real_t{1e-10});
    if (max_err > tolerance) {
      std::ostringstream oss;
      oss << "Contraction mismatch (max error = " << max_err << ", tolerance = " << tolerance << ")";
      state.SkipWithError(oss.str().c_str());
      return;
    }

    for (auto _ : state) {
      result() = Value{0};
      tblis_contract(Value{1}, a, "ikl", b, "kjl", Value{0}, result, "ij");
      benchmark::DoNotOptimize(result.data());
    }

    double const volume      = static_cast<double>(I) * static_cast<double>(J) * static_cast<double>(K) * static_cast<double>(L);
    state.counters["volume"] = volume;
    state.counters["flops"]  = benchmark::Counter(2.0 * volume, benchmark::Counter::kIsIterationInvariantRate);
  }

} // namespace

BENCHMARK_TEMPLATE(TblisTensorContract, float)->Args({64, 64, 64, 64})->Args({64, 64, 64, 64})->Args({64, 96, 96, 24})->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(TblisTensorContract, double)
   ->Args({64, 64, 64, 64})
   ->Args({64, 64, 64, 64})
   ->Args({64, 96, 96, 24})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(TblisTensorContract, std::complex<float>)
   ->Args({64, 64, 64, 64})
   ->Args({64, 24, 64, 64})
   ->Args({96, 64, 80, 40})
   ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(TblisTensorContract, std::complex<double>)
   ->Args({64, 64, 64, 64})
   ->Args({64, 24, 64, 64})
   ->Args({96, 64, 80, 40})
   ->Unit(benchmark::kMillisecond);

#endif
