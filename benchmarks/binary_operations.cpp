#include "./bench_common.hpp"
#include <nda/nda.hpp>
#include <benchmark/benchmark.h>
#include <string>

constexpr long Nmin = (1 << 5) - 1;
constexpr long Nmax = 1 << 8;
using namespace nda;

/**
 * @brief Applies the custom size sequence f(x) = 2x + 1 to the benchmark.
 *
 * This generates the sequence: 7, 15, 31, 63, ...
 */
static void custom_range(benchmark::internal::Benchmark *b) {
  for (long n = Nmin; n <= Nmax; n = 2 * n + 1) { b->Args({n}); }
}

// =================== Operation Functors ===================
// Each struct defines a single operation to be benchmarked.

// --- Binary Ops ---
struct op_add {
  static auto op(auto const &A, auto const &B, auto const &C) { return A + B; }
  static constexpr bool supports_complex = true;
};
struct op_sub {
  static auto op(auto const &A, auto const &B, auto const &C) { return A - B; }
  static constexpr bool supports_complex = true;
};
struct op_mul {
  static auto op(auto const &A, auto const &B, auto const &C) { return A * B; }
  static constexpr bool supports_complex = true;
};
struct op_div {
  static auto op(auto const &A, auto const &B, auto const &C) { return A / B; }
  static constexpr bool supports_complex = true;
};

// --- Fused Ops ---
struct op_fma {
  static auto op(auto const &A, auto const &B, auto const &C) { return A * B + C; }
  static constexpr bool supports_complex = true;
};
struct op_fms {
  static auto op(auto const &A, auto const &B, auto const &C) { return A * B - C; }
  static constexpr bool supports_complex = true;
};
struct op_add3 {
  static auto op(auto const &A, auto const &B, auto const &C) { return A + B + C; }
  static constexpr bool supports_complex = true;
};
struct op_addsub {
  static auto op(auto const &A, auto const &B, auto const &C) { return A + B - C; }
  static constexpr bool supports_complex = true;
};

// --- Arithmetic Functions ---
struct op_max_elem {
  static auto op(auto const &A, auto const &B, auto const &C) { return max_element(A + B); }
  static constexpr bool supports_complex = false; // max_element not for complex
};
struct op_min_elem {
  static auto op(auto const &A, auto const &B, auto const &C) { return min_element(A + B); }
  static constexpr bool supports_complex = false; // min_element not for complex
};
struct op_frob_norm {
  static auto op(auto const &A, auto const &B, auto const &C) { return frobenius_norm(A + B); }
  static constexpr bool supports_complex = true;
};
struct op_sum {
  static auto op(auto const &A, auto const &B, auto const &C) { return sum(A + B); }
  static constexpr bool supports_complex = true;
};
struct op_prod {
  static auto op(auto const &A, auto const &B, auto const &C) { return product(A + B); }
  static constexpr bool supports_complex = true;
};

// --- Mapped Functions ---
struct op_pow {
  static auto op(auto const &A, auto const &B, auto const &C) { return pow(A, 3.5); }
  static constexpr bool supports_complex = true;
};
struct op_conj {
  static auto op(auto const &A, auto const &B, auto const &C) { return conj(A); }
  static constexpr bool supports_complex = true;
};
struct op_abs {
  static auto op(auto const &A, auto const &B, auto const &C) { return abs(A); }
  static constexpr bool supports_complex = true;
};
struct op_imag {
  static auto op(auto const &A, auto const &B, auto const &C) { return imag(A); }
  static constexpr bool supports_complex = true;
};
struct op_floor {
  static auto op(auto const &A, auto const &B, auto const &C) { return floor(A); }
  static constexpr bool supports_complex = false; // floor not for complex
};
struct op_real {
  static auto op(auto const &A, auto const &B, auto const &C) { return real(A); }
  static constexpr bool supports_complex = true;
};
struct op_abs2 {
  static auto op(auto const &A, auto const &B, auto const &C) { return abs2(A); }
  static constexpr bool supports_complex = true;
};
struct op_isnan {
  static auto op(auto const &A, auto const &B, auto const &C) { return isnan(A); }
  static constexpr bool supports_complex = true;
};
struct op_exp {
  static auto op(auto const &A, auto const &B, auto const &C) { return exp(A); }
  static constexpr bool supports_complex = true;
};
struct op_cos {
  static auto op(auto const &A, auto const &B, auto const &C) { return cos(A); }
  static constexpr bool supports_complex = true;
};
struct op_sin {
  static auto op(auto const &A, auto const &B, auto const &C) { return sin(A); }
  static constexpr bool supports_complex = true;
};
struct op_tan {
  static auto op(auto const &A, auto const &B, auto const &C) { return tan(A); }
  static constexpr bool supports_complex = true;
};
struct op_cosh {
  static auto op(auto const &A, auto const &B, auto const &C) { return cosh(A); }
  static constexpr bool supports_complex = true;
};
struct op_sinh {
  static auto op(auto const &A, auto const &B, auto const &C) { return sinh(A); }
  static constexpr bool supports_complex = true;
};
struct op_tanh {
  static auto op(auto const &A, auto const &B, auto const &C) { return tanh(A); }
  static constexpr bool supports_complex = true;
};
struct op_acos {
  static auto op(auto const &A, auto const &B, auto const &C) { return acos(A); }
  static constexpr bool supports_complex = true;
};
struct op_asin {
  static auto op(auto const &A, auto const &B, auto const &C) { return asin(A); }
  static constexpr bool supports_complex = true;
};
struct op_atan {
  static auto op(auto const &A, auto const &B, auto const &C) { return atan(A); }
  static constexpr bool supports_complex = true;
};
struct op_log {
  static auto op(auto const &A, auto const &B, auto const &C) { return log(A); }
  static constexpr bool supports_complex = true;
};
struct op_sqrt {
  static auto op(auto const &A, auto const &B, auto const &C) { return sqrt(A); }
  static constexpr bool supports_complex = true;
};

// --- Deep Expressions ---
struct op_deep_expr1 {
  static auto op(auto const &A, auto const &B, auto const &C) {
    auto e1 = A * B + C;
    auto e2 = sin(e1) + cos(B);
    auto e3 = log(e1 * conj(e1) + 1.0);
    auto e4 = e2 + e3;
    auto e5 = sqrt(e4 * conj(e4) + 1.0);
    auto e6 = e5 / (A * A + e2 * e2 + 1.0);
    return e6;
  }
  static constexpr bool supports_complex = true;
};

struct op_deep_expr2 {
  static auto op(auto const &A, auto const &B, auto const &C) {
    auto e1 = conj(A) * B;
    auto e2 = e1 - C;
    auto e3 = exp(e2 / (B * B + 12351213.0));
    auto e4 = sinh(e1) + cosh(e3);
    auto e5 = e4 * conj(e4);
    auto e6 = e5 * e3 + A;
    auto e7 = e6 * e6;
    return e7;
  }
  static constexpr bool supports_complex = true;
};

struct op_deep_expr3 {
  static auto op(auto const &A, auto const &B, auto const &C) {
    auto e1 = tan(A - B);
    auto e2 = log(C * C + 1.0);
    auto e3 = e1 * e2;
    auto e4 = sqrt(e3 * conj(e3) + 1.0);
    auto e5 = e4 + A * B * C;
    auto e6 = e5 * conj(e4);
    auto e7 = e6 - e3;
    auto e8 = e7 / (e1 * e1 + 1.0);
    return e8;
  }
  static constexpr bool supports_complex = true;
};

struct op_deep_expr4 {
  static auto op(auto const &A, auto const &B, auto const &C) {
    auto e1 = A + B;
    auto e2 = A - B;
    auto e3 = e1 * e2;
    auto e4 = asin(C / (C * C + 1.0));
    auto e5 = e3 * e4;
    auto e6 = exp(e5) + C;
    auto e7 = conj(e1) * e6;
    auto e8 = log(e7 * conj(e7) + 1.0);
    auto e9 = e8 + tanh(e3);
    return e9;
  }
  static constexpr bool supports_complex = true;
};

struct op_deep_expr5 {
  static auto op(auto const &A, auto const &B, auto const &C) {
    auto e1  = A * B;
    auto e2  = B * C;
    auto e3  = C * A;
    auto e4  = e1 + e2 + e3;
    auto e5  = cos(e4) + sin(e1);
    auto e6  = sqrt(e5 * conj(e5) + 1.0);
    auto e7  = e6 / (e2 + 1.0);
    auto e8  = e7 - conj(e5);
    auto e9  = log(e8 * conj(e8) + 1.0);
    auto e10 = e9 * e4 + A;
    return e10;
  }
  static constexpr bool supports_complex = true;
};

struct op_deep_expr6 {
  static auto op(auto const &A, auto const &B, auto const &C) {
    auto e1 = acos(B / (B * B + 1312412451.0));
    auto e2 = exp(A) - C;
    auto e3 = e1 * e2;
    auto e4 = log(e3 * conj(e3) + 1.0);
    auto e5 = e4 * conj(e4) + A;
    auto e6 = e5 / (e1 + 1.0);
    auto e7 = e6 + e2;
    auto e8 = sqrt(e7 * conj(e7) + 1.0);
    return sum(e8);
  }
  static constexpr bool supports_complex = true;
};

// =================== Unified Benchmark Function ===================

/**
 * @brief Unified benchmark function.
 * @tparam T   Value type (float, double, complex)
 * @tparam Op  Operation functor (e.g., op_add, op_fma, op_log)
 */
template <typename T, typename Op>
static void run_benchmark(benchmark::State &state) {

  // --- 1. Check Type Support ---
  if constexpr (is_complex_v<T> and !Op::supports_complex) {
    state.SkipWithError("Complex Type is not supported for this Operation");
    return;
  }

  // --- 2. Setup ---
  long N        = state.range(0); // Size;
  using array_t = array<T, 2>;

  array_t A = array_t::rand({N, N});
  array_t B = array_t::rand({N, N});
  array_t C = array_t::rand({N, N});

  // Make everything >= 1. So some functions do not fail (division with zero and etc...)
  A = abs(A) + 1.0;
  B = abs(B) + 1.0;
  C = abs(C) + 1.0;

  benchmark::DoNotOptimize(A);
  benchmark::DoNotOptimize(B);
  benchmark::DoNotOptimize(C);

  for (auto s : state) {
    auto R = make_regular(Op::op(A, B, C));
    benchmark::DoNotOptimize(R);
  }

  long num_elements = N * N;
  state.SetComplexityN(N);
  state.SetItemsProcessed(state.iterations() * num_elements);
}
// Helper struct to register benchmarks.
// The constructor is called for each static instance,
// allowing if constexpr to be used *inside* a function body.
template <typename OpType>
struct benchmark_registrar {
  benchmark_registrar(const std::string &name) {
    // We are inside a function, so if constexpr is legal.
    BENCHMARK_TEMPLATE(run_benchmark, float, OpType)->Name(name + "/float")->Apply(custom_range)->Unit(benchmark::kMicrosecond)->Complexity();

    BENCHMARK_TEMPLATE(run_benchmark, double, OpType)->Name(name + "/double")->Apply(custom_range)->Unit(benchmark::kMicrosecond)->Complexity();

    if constexpr (OpType::supports_complex) {
      BENCHMARK_TEMPLATE(run_benchmark, std::complex<float>, OpType)
         ->Name(name + "/complex_float")
         ->Apply(custom_range)
         ->Unit(benchmark::kMicrosecond)
         ->Complexity();

      BENCHMARK_TEMPLATE(run_benchmark, std::complex<double>, OpType)
         ->Name(name + "/complex_double")
         ->Apply(custom_range)
         ->Unit(benchmark::kMicrosecond)
         ->Complexity();
    }
  }
};

// The new macro just creates a static instance of the registrar
#define NDA_BENCHMARK_ALL_TYPES(OP_TYPE, NAME) static benchmark_registrar<OP_TYPE> registrar_##OP_TYPE(NAME);

// --- Binary Ops ---
NDA_BENCHMARK_ALL_TYPES(op_add, "Addition")
NDA_BENCHMARK_ALL_TYPES(op_sub, "Subtraction")
NDA_BENCHMARK_ALL_TYPES(op_mul, "Multiplication")
NDA_BENCHMARK_ALL_TYPES(op_div, "Division")

// --- Fused Ops ---
NDA_BENCHMARK_ALL_TYPES(op_fma, "A*B+C")
NDA_BENCHMARK_ALL_TYPES(op_fms, "A*B-C")
NDA_BENCHMARK_ALL_TYPES(op_add3, "A+B+C")
NDA_BENCHMARK_ALL_TYPES(op_addsub, "A+B-C")

// --- Arithmetic Functions ---
NDA_BENCHMARK_ALL_TYPES(op_max_elem, "max_element(A)")
NDA_BENCHMARK_ALL_TYPES(op_min_elem, "min_element(A)")
NDA_BENCHMARK_ALL_TYPES(op_frob_norm, "frobenius_norm(A)")
NDA_BENCHMARK_ALL_TYPES(op_sum, "sum(A)")
NDA_BENCHMARK_ALL_TYPES(op_prod, "product(A)")

// --- Mapped Functions (on A) ---
NDA_BENCHMARK_ALL_TYPES(op_pow, "pow(A, 3.5)")
NDA_BENCHMARK_ALL_TYPES(op_conj, "conj(A)")
NDA_BENCHMARK_ALL_TYPES(op_abs, "abs(A)")
NDA_BENCHMARK_ALL_TYPES(op_imag, "imag(A)")
NDA_BENCHMARK_ALL_TYPES(op_floor, "floor(A)")
NDA_BENCHMARK_ALL_TYPES(op_real, "real(A)")
NDA_BENCHMARK_ALL_TYPES(op_abs2, "abs2(A)")
NDA_BENCHMARK_ALL_TYPES(op_isnan, "isnan(A)")
NDA_BENCHMARK_ALL_TYPES(op_exp, "exp(A)")
NDA_BENCHMARK_ALL_TYPES(op_cos, "cos(A)")
NDA_BENCHMARK_ALL_TYPES(op_sin, "sin(A)")
NDA_BENCHMARK_ALL_TYPES(op_tan, "tan(A)")
NDA_BENCHMARK_ALL_TYPES(op_cosh, "cosh(A)")
NDA_BENCHMARK_ALL_TYPES(op_sinh, "sinh(A)")
NDA_BENCHMARK_ALL_TYPES(op_tanh, "tanh(A)")
NDA_BENCHMARK_ALL_TYPES(op_acos, "acos(A)")
NDA_BENCHMARK_ALL_TYPES(op_asin, "asin(A)")
NDA_BENCHMARK_ALL_TYPES(op_atan, "atan(A)")
NDA_BENCHMARK_ALL_TYPES(op_log, "log(A)")
NDA_BENCHMARK_ALL_TYPES(op_sqrt, "sqrt(A)")

// --- Deep Expressions ---
NDA_BENCHMARK_ALL_TYPES(op_deep_expr1, "Deep_expr/Mix_1")
NDA_BENCHMARK_ALL_TYPES(op_deep_expr2, "Deep_expr/Mix_2")
NDA_BENCHMARK_ALL_TYPES(op_deep_expr3, "Deep_expr/Mix_3")
NDA_BENCHMARK_ALL_TYPES(op_deep_expr4, "Deep_expr/Mix_4")
NDA_BENCHMARK_ALL_TYPES(op_deep_expr5, "Deep_expr/Mix_5")
NDA_BENCHMARK_ALL_TYPES(op_deep_expr6, "Deep_expr/Mix_6")

BENCHMARK_MAIN();