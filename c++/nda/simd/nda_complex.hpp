#pragma once
#include <xsimd/xsimd.hpp>

namespace nda {
  template <typename T, typename A = xsimd::default_arch>
    requires(std::is_same_v<T, std::complex<float>> or std::is_same_v<T, std::complex<double>>)
  struct complex_batch : public xsimd::types::simd_register<T, A>, public xsimd::types::integral_only_operators<T, A> {
    static constexpr std::size_t size = sizeof(xsimd::types::simd_register<typename T::value_type, A>) / sizeof(T);

    using value_type    = T;
    using scalar_t      = typename T::value_type;
    using arch_type     = A;
    using batch_t       = xsimd::batch<scalar_t, A>;
    using register_type = typename xsimd::types::simd_register<scalar_t, A>::register_type;
    batch_t value;

    // constructors
    complex_batch() = default;

    complex_batch(T val) noexcept {
      auto get_val = [&](std::size_t I) { return (I % 2 == 0) ? val.real() : val.imag(); };

      [&]<std::size_t... I>(std::index_sequence<I...>) {
        value = xsimd::kernel::set(value, A{}, get_val(I)...);
      }(std::make_index_sequence<batch_t::size>{});
    }

    template <class... Ts>
    complex_batch(T val0, T val1, Ts... vals) noexcept {

      const std::array<T, batch_t::size> complex_values = {val0, val1, static_cast<T>(vals)...};

      auto get_val = [&](std::size_t I) {
        const T &complex_val = complex_values[I / 2];
        return (I % 2 == 0) ? complex_val.real() : complex_val.imag();
      };

      [&]<std::size_t... I>(std::index_sequence<I...>) {
        value = xsimd::kernel::set(value, A{}, get_val(I)...);
      }(std::make_index_sequence<batch_t::size>{});
    }

    complex_batch(register_type reg) noexcept : value(reg) {}

    template <class U>
    static complex_batch broadcast(U val) noexcept {
      if constexpr (std::is_same_v<std::complex<float>, U> or std::is_same_v<std::complex<double>, U>) {
        return complex_batch(static_cast<T>(val));
      } else {
        return complex_batch(std::complex<scalar_t>{static_cast<U>(val), 0});
      }
    }

    // memory operators
    template <class U>
    void store_aligned(U *mem) const noexcept {
      value.store_aligned(mem);
    }

    template <class U>
    void store_unaligned(U *mem) const noexcept {
      value.store_unaligned(mem);
    }

    template <class U>
    void store(U *mem, xsimd::aligned_mode) const noexcept {
      value.store_aligned(mem);
    }

    template <class U>
    void store(U *mem, xsimd::unaligned_mode) const noexcept {
      value.store_unaligned(mem);
    }

    template <class U>
    static complex_batch load_aligned(U const *mem) noexcept {
      return complex_batch(batch_t::load_aligned(mem));
    }

    template <class U>
    static complex_batch load_unaligned(U const *mem) noexcept {
      return complex_batch(batch_t::load_unaligned(mem));
    }

    template <class U>
    static complex_batch load(U const *mem, xsimd::aligned_mode) noexcept {
      return complex_batch(batch_t::load_aligned(mem));
    }

    template <class U>
    static complex_batch load(U const *mem, xsimd::unaligned_mode) noexcept {
      return complex_batch(batch_t::load_unaligned(mem));
    }

    // Update operators
    complex_batch &operator+=(complex_batch const &other) noexcept {
      this->value += other.value;
      return *this;
    };

    complex_batch &operator-=(complex_batch const &other) noexcept {
      this->value -= other.value;
      return *this;
    };

    [[gnu::optimize(
       "O3")]] // TODO: This is needed currently as without optimizations mask in xsimd::swizzle is not a compile time expression and it fails to compile.
    complex_batch &operator*=(complex_batch const &other) noexcept {
      struct swap_pair {
        static constexpr unsigned get(unsigned i, unsigned) noexcept { return i ^ 1u; }
      };
      struct dup_real {
        static constexpr unsigned get(unsigned i, unsigned) noexcept { return i & ~1u; }
      };
      struct dup_imag {
        static constexpr unsigned get(unsigned i, unsigned) noexcept { return i | 1u; }
      };

      static constexpr auto swap_idx = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<scalar_t>, swap_pair, arch_type>();
      static constexpr auto real_idx = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<scalar_t>, dup_real, arch_type>();
      static constexpr auto imag_idx = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<scalar_t>, dup_imag, arch_type>();

      static_assert(xsimd::kernel::detail::is_dup_lo<decltype(real_idx)>());
      static_assert(xsimd::kernel::detail::is_dup_hi<decltype(imag_idx)>());

      const auto other_im = xsimd::swizzle(other.value, imag_idx); // [bi0,bi0,bi1,bi1]
      const auto value_sw = xsimd::swizzle(value, swap_idx);
      const auto cross    = value_sw * other_im;
      const auto other_re = xsimd::swizzle(other.value, real_idx); // [br0,br0,br1,br1]
      value               = xsimd::fmas(other_re, value, cross);
      return *this;
    };

    [[gnu::optimize(
       "O3")]] // TODO: This is needed currently as without optimizations mask in xsimd::swizzle is not a compile time expression and it fails to compile.
    complex_batch &operator/=(complex_batch const &other) noexcept {

      struct dup_real {
        static constexpr unsigned get(unsigned i, unsigned) noexcept { return i & ~1u; }
      };
      struct dup_imag {
        static constexpr unsigned get(unsigned i, unsigned) noexcept { return i | 1u; }
      };

      constexpr auto real_idx = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<scalar_t>, dup_real, arch_type>();
      constexpr auto imag_idx = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<scalar_t>, dup_imag, arch_type>();

      // (a+bi) / (c+di) = (a+bi) * (c-di) / (c^2+d^2);
      const batch_t other_real = xsimd::swizzle(other.value, real_idx);
      const batch_t other_img  = xsimd::swizzle(other.value, imag_idx);

      const batch_t denom = (other_real * other_real) + (other_img * other_img);

      static auto get_val = [&](const std::size_t I) -> xsimd::as_unsigned_integer_t<scalar_t> {
        if constexpr (sizeof(xsimd::as_unsigned_integer_t<scalar_t>) == 8)
          return (I % 2 == 0) ? 0x0000000000000000 : 0x8000000000000000;
        else
          return (I % 2 == 0) ? 0x00000000 : 0x80000000;
      };

      static const batch_t mask = [&]<std::size_t... I>(std::index_sequence<I...>) {
        return xsimd::bitwise_cast<batch_t, xsimd::batch<xsimd::as_unsigned_integer_t<scalar_t>>>(
           xsimd::kernel::set(xsimd::bitwise_cast<xsimd::batch<xsimd::as_unsigned_integer_t<scalar_t>>, batch_t>(value), A{}, get_val(I)...));
      }(std::make_index_sequence<batch_t::size>{});

      const batch_t other_conj = other.value ^ mask;

      const complex_batch numerator = (*this) * complex_batch(other_conj);

      this->value = numerator.value / denom;
      return *this;
    };

    complex_batch &operator&=(complex_batch const &other) noexcept {
      this->value &= other.value;
      return *this;
    };

    complex_batch &operator|=(complex_batch const &other) noexcept {
      this->value |= other.value;
      return *this;
    };

    complex_batch &operator^=(complex_batch const &other) noexcept {
      this->value ^= other.value;
      return *this;
    };

    // arithmetic operators. They are defined as friend to enable automatic
    // conversion of parameters from scalar to batch. Inline implementation
    // is required to avoid warnings.

    friend complex_batch operator+(complex_batch const &self, complex_batch const &other) noexcept { return complex_batch(self) += other; }

    friend complex_batch operator-(complex_batch const &self, complex_batch const &other) noexcept { return complex_batch(self) -= other; }

    friend complex_batch operator*(complex_batch const &self, complex_batch const &other) noexcept { return complex_batch(self) *= other; }

    friend complex_batch operator/(complex_batch const &self, complex_batch const &other) noexcept { return complex_batch(self) /= other; }

    friend complex_batch operator&(complex_batch const &self, complex_batch const &other) noexcept { return complex_batch(self) &= other; }

    friend complex_batch operator|(complex_batch const &self, complex_batch const &other) noexcept { return complex_batch(self) |= other; }

    friend complex_batch operator^(complex_batch const &self, complex_batch const &other) noexcept { return complex_batch(self) ^= other; }
  };
} // namespace nda