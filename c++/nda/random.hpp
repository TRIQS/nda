// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a random number generator and a uniform distribution for nda arrays.
 */

#pragma once

#include "./concepts.hpp"
#include "./exceptions.hpp"
#include "./layout/for_each.hpp"
#include "./mem/address_space.hpp"
#include "./traits.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

namespace nda::rng {

  /**
   * @addtogroup av_factories
   * @{
   */

  /**
   * @brief Check if a given type is a random number engine that produces 64 random bits per call, e.g.
   * `std::mt19937_64`.
   *
   * @tparam E Type to check.
   */
  template <typename E>
  concept Engine = std::uniform_random_bit_generator<E> and (E::min() == 0) and (E::max() == std::numeric_limits<std::uint64_t>::max());

  /**
   * @brief Interval between two values given in any order.
   *
   * @tparam T nda::Arithmetic type of the values.
   */
  template <Arithmetic T = double>
  struct interval {
    /// First end of the interval.
    T a = 0;

    /// Second end of the interval.
    T b = 1;

    /// Get the smaller end of the interval.
    [[nodiscard]] constexpr T min() const { return std::min(a, b); }

    /// Get the larger end of the interval.
    [[nodiscard]] constexpr T max() const { return std::max(a, b); }
  };

  // Class template argument deduction guide.
  template <typename A, typename B>
  interval(A, B) -> interval<std::common_type_t<A, B>>;

  /**
   * @brief Rectangle in the complex plane.
   *
   * @tparam T nda::Arithmetic type of the values.
   */
  template <Arithmetic T = double>
  struct rectangle {
    /// Interval of the real part.
    interval<T> re;

    /// Interval of the imaginary part.
    interval<T> im;
  };

  namespace detail {

    template <typename T, Engine E>
    T canonical(E &e) {
      auto const bits = static_cast<std::uint64_t>(e());
      if constexpr (std::same_as<T, double>) {
        return static_cast<double>(bits >> 11) * 0x1.0p-53;
      } else {
        return static_cast<float>(bits >> 40) * 0x1.0p-24f;
      }
    }

  } // namespace detail

  /**
   * @brief Check if a given type is a distribution that draws a value when called with an engine of type `E`.
   *
   * @tparam D Type to check.
   * @tparam E nda::rng::Engine type.
   */
  template <typename D, typename E>
  concept Distribution = Engine<E> and std::invocable<D &, E &>;

  /**
   * @brief Uniform distribution over an nda::rng::interval or an nda::rng::rectangle.
   *
   * @details Values are drawn from
   * - [min, max) for floating point types,
   * - [min, max] for integer types,
   * - [min, max) for both the real and the imaginary part for complex types.
   *
   * @tparam T Integer type (except `bool`), `float`, `double`, `std::complex<float>` or `std::complex<double>`.
   */
  template <typename T>
  class uniform_distribution {
    static_assert((std::integral<T> and not std::same_as<T, bool>) or AnyOf<T, float, double, std::complex<float>, std::complex<double>>,
                  "Error in nda::rng::uniform_distribution: Unsupported value type");

    using real_t = remove_complex_t<T>;

    public:
    /// Type of the drawn values.
    using result_type = T;

    /// Default constructor draws from the interval [0, 1) (complex types: from the rectangle [0, 1) x [0, 1)).
    uniform_distribution() = default;

    /**
     * @brief Construct a uniform distribution over an interval (integer and floating point types).
     *
     * @details Throws an nda::runtime_error if the width of a floating point interval is not finite.
     *
     * @tparam U nda::Arithmetic type of the interval.
     * @param bounds Interval to draw from.
     */
    template <typename U>
    explicit uniform_distribution(interval<U> const &bounds) : re{static_cast<real_t>(bounds.min()), static_cast<real_t>(bounds.max())} {
      static_assert(not is_complex_v<T>, "Error in nda::rng::uniform_distribution: Use a rectangle for complex values");
      check_finite_width(re);
    }

    /**
     * @brief Construct a uniform distribution over a rectangle (complex types).
     *
     * @details Throws an nda::runtime_error if the width of one of its intervals is not finite.
     *
     * @tparam U nda::Arithmetic type of the rectangle.
     * @param bounds Rectangle to draw from.
     */
    template <typename U>
    explicit uniform_distribution(rectangle<U> const &bounds)
       : re{static_cast<real_t>(bounds.re.min()), static_cast<real_t>(bounds.re.max())},
         im{static_cast<real_t>(bounds.im.min()), static_cast<real_t>(bounds.im.max())} {
      static_assert(is_complex_v<T>, "Error in nda::rng::uniform_distribution: Use an interval for real values");
      check_finite_width(re);
      check_finite_width(im);
    }

    /**
     * @brief Draw a value.
     *
     * @tparam E nda::rng::Engine type.
     * @param e Engine.
     * @return Random value.
     */
    template <Engine E>
    result_type operator()(E &e) const {
      if constexpr (is_complex_v<T>) {
        auto const x = draw(re, e);
        auto const y = draw(im, e);
        return {x, y};
      } else {
        return draw(re, e);
      }
    }

    private:
    static void check_finite_width(std::pair<real_t, real_t> const &bounds) {
      if constexpr (not std::integral<T>) {
        auto const [min, max] = bounds;
        if (not std::isfinite(max - min))
          NDA_RUNTIME_ERROR << "Error in nda::rng::uniform_distribution: The width of the interval [" << min << ", " << max << ") is not finite";
      }
    }

    template <Engine E>
    static real_t draw(std::pair<real_t, real_t> const &bounds, E &e) {
      auto const [min, max] = bounds;
      if constexpr (std::integral<T>) {
        auto const span = static_cast<std::uint64_t>(max) - static_cast<std::uint64_t>(min);
        // smallest 2^k - 1 >= span
        auto const mask = span == 0 ? std::uint64_t{0} : ~std::uint64_t{0} >> std::countl_zero(span);
        while (true) {
          auto const r = static_cast<std::uint64_t>(e()) & mask;
          if (r <= span) return static_cast<T>(static_cast<std::uint64_t>(min) + r);
        }
      } else {
        auto const x = std::fma(max - min, detail::canonical<real_t>(e), min);
        return x >= max ? std::nextafter(max, min) : x;
      }
    }

    std::pair<real_t, real_t> re = {0, 1};
    std::pair<real_t, real_t> im = {0, 1};
  };

  /**
   * @brief Random number generator that owns an engine and draws values and arrays from it.
   *
   * @details Engines with fewer than 64 bits per call, e.g. `std::mt19937`, are wrapped in
   * `std::independent_bits_engine<E, 64, std::uint64_t>`. The generator itself satisfies nda::rng::Engine.
   *
   * @tparam E Engine type satisfying `std::uniform_random_bit_generator`.
   */
  template <std::uniform_random_bit_generator E = std::mt19937_64>
  class random_number_generator {
    using engine64_t = std::conditional_t<Engine<E>, E, std::independent_bits_engine<E, 64, std::uint64_t>>;

    public:
    /// Type of the engine.
    using engine_type = E;

    /// Type of the random bits returned by operator()().
    using result_type = std::uint64_t;

    /// Get the smallest value returned by operator()().
    static constexpr result_type min() { return 0; }

    /// Get the largest value returned by operator()().
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    /// Default constructor uses a default seeded engine.
    random_number_generator() = default; //NOLINT

    /**
     * @brief Construct a generator with an engine seeded with the given seed.
     * @param seed Seed of the engine.
     */
    explicit random_number_generator(std::uint64_t seed) : engine(seed) {}

    /**
     * @brief Construct a generator from a given engine.
     * @param e Engine.
     */
    explicit random_number_generator(E e) : engine(std::move(e)) {}

    /**
     * @brief Seed the engine.
     * @param seed Seed of the engine.
     */
    void seed(std::uint64_t seed) { engine.seed(seed); }

    /**
     * @brief Draw 64 random bits.
     * @return Random 64-bit unsigned integer.
     */
    result_type operator()() { return static_cast<result_type>(engine()); }

    /**
     * @brief Draw a value from a distribution.
     *
     * @tparam D nda::rng::Distribution type.
     * @param dist Distribution.
     * @return Random value.
     */
    template <Distribution<engine64_t> D>
    auto operator()(D &&dist) {
      return std::forward<D>(dist)(engine);
    }

    /**
     * @brief Draw a value from the default nda::rng::uniform_distribution, i.e. from [0, 1).
     *
     * @tparam T Value type.
     * @return Random value.
     */
    template <typename T>
    T random() {
      return uniform_distribution<T>{}(engine);
    }

    /**
     * @brief Draw a value from an interval (see nda::rng::uniform_distribution).
     *
     * @tparam T Integer or floating point value type.
     * @tparam U nda::Arithmetic type of the interval.
     * @param bounds Interval to draw from.
     * @return Random value.
     */
    template <typename T, typename U>
    T uniform(interval<U> const &bounds) {
      return uniform_distribution<T>{bounds}(engine);
    }

    /**
     * @brief Draw a complex value from a rectangle (see nda::rng::uniform_distribution).
     *
     * @tparam T Complex value type.
     * @tparam U nda::Arithmetic type of the rectangle.
     * @param bounds Rectangle to draw from.
     * @return Random value.
     */
    template <typename T, typename U>
    T uniform(rectangle<U> const &bounds) {
      return uniform_distribution<T>{bounds}(engine);
    }

    /**
     * @brief Fill an array or view with values drawn from a distribution.
     *
     * @details The elements are filled in memory order. A distribution passed as an lvalue keeps its state.
     *
     * @tparam A nda::MemoryArray type in host-accessible memory.
     * @tparam D nda::rng::Distribution type.
     * @param a Array or view to fill.
     * @param dist Distribution.
     */
    template <MemoryArray A, Distribution<engine64_t> D>
    void fill(A &&a, D &&dist) { // NOLINT (a is written and dist is called as an lvalue: neither is forwarded)
      static_assert(mem::have_host_compatible_addr_space<std::remove_cvref_t<A>>,
                    "Error in nda::rng::random_number_generator::fill: The array must be in host-accessible memory");
      nda::for_each_static<0, get_layout_info<A>.stride_order>(a.shape(), [&](auto... is) { a(is...) = dist(engine); });
    }

    /**
     * @brief Fill an array or view with values drawn from the distribution `Dist<get_value_t<A>>(args...)`.
     *
     * @details For example, `rng.fill<uniform_distribution>(a, interval{2, 10})`.
     *
     * @tparam Dist Distribution class template with the value type as its template parameter.
     * @tparam A nda::MemoryArray type in host-accessible memory.
     * @tparam Args Types of the distribution's constructor arguments.
     * @param a Array or view to fill.
     * @param args Constructor arguments of the distribution.
     */
    template <template <typename> class Dist, MemoryArray A, typename... Args>
    void fill(A &&a, Args &&...args) { // NOLINT (a is written, not forwarded)
      using dist_t = Dist<get_value_t<A>>;
      static_assert(std::constructible_from<dist_t, Args...>,
                    "Error in nda::rng::random_number_generator::fill: The distribution cannot be made from the arguments");
      fill(a, dist_t(std::forward<Args>(args)...));
    }

    /**
     * @brief Fill an array or view with values drawn from an nda::rng::uniform_distribution.
     *
     * @tparam A nda::MemoryArray type in host-accessible memory.
     * @tparam B nda::rng::interval or nda::rng::rectangle type.
     * @param a Array or view to fill.
     * @param bounds Interval or rectangle to draw from.
     */
    template <MemoryArray A, typename B>
      requires(not Distribution<B, engine64_t>)
    void fill(A &&a, B const &bounds) { // NOLINT (a is written, not forwarded)
      fill<uniform_distribution>(a, bounds);
    }

    private:
    engine64_t engine;
  };

  /**
   * @brief Get the global random number generator used by nda::rand.
   *
   * @note The global generator is not thread-safe.
   *
   * @return Reference to the global generator.
   */
  inline random_number_generator<> &global_rng() {
    static random_number_generator<> rng{};
    return rng;
  }

  /** @} */

} // namespace nda::rng
