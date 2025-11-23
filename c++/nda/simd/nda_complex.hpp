#pragma once

#include <xsimd/xsimd.hpp>

namespace nda {
  template <typename T, typename A = xsimd::default_arch>
    requires(std::is_same_v<T, std::complex<float>> or std::is_same_v<T, std::complex<double>>)
  struct complex_batch : public xsimd::types::simd_register<T, A>, public xsimd::types::integral_only_operators<T, A> {

    private:
#ifdef NDA_ENFORCE_BOUNDCHECK
    static constexpr bool has_no_boundcheck = false;
#else
    static constexpr bool has_no_boundcheck = true;
#endif

    public:
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

      const std::array<T, size> complex_values = {val0, val1, static_cast<T>(vals)...};

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
        return complex_batch(T{static_cast<U>(val), 0});
      }
    }

    // memory operators
    template <class U>
      requires(std::is_same_v<U, T> or std::is_same_v<U, scalar_t>)
    void store_aligned(U *mem) const noexcept {
      value.store_aligned(reinterpret_cast<scalar_t *>(mem));
    }

    template <class U>
      requires(std::is_same_v<U, T> or std::is_same_v<U, scalar_t>)
    void store_unaligned(U *mem) const noexcept {
      value.store_unaligned(reinterpret_cast<scalar_t *>(mem));
    }

    template <class U>
      requires(std::is_same_v<U, T> or std::is_same_v<U, scalar_t>)
    void store(U *mem, xsimd::aligned_mode) const noexcept {
      value.store_aligned(reinterpret_cast<scalar_t *>(mem));
    }

    template <class U>
      requires(std::is_same_v<U, T> or std::is_same_v<U, scalar_t>)
    void store(U *mem, xsimd::unaligned_mode) const noexcept {
      value.store_unaligned(reinterpret_cast<scalar_t *>(mem));
    }

    template <class U>
      requires(std::is_same_v<U, T> or std::is_same_v<U, scalar_t>)
    static complex_batch load_aligned(U const *mem) noexcept {
      return complex_batch(batch_t::load_aligned(reinterpret_cast<scalar_t const *>(mem)));
    }

    template <class U>
      requires(std::is_same_v<U, T> or std::is_same_v<U, scalar_t>)
    static complex_batch load_unaligned(U const *mem) noexcept {
      return complex_batch(batch_t::load_unaligned(reinterpret_cast<scalar_t const *>(mem)));
    }

    template <class U>
      requires(std::is_same_v<U, T> or std::is_same_v<U, scalar_t>)
    static complex_batch load(U const *mem, xsimd::aligned_mode) noexcept {
      return complex_batch(batch_t::load_aligned(reinterpret_cast<scalar_t const *>(mem)));
    }

    template <class U>
      requires(std::is_same_v<U, T> or std::is_same_v<U, scalar_t>)
    static complex_batch load(U const *mem, xsimd::unaligned_mode) noexcept {
      return complex_batch(batch_t::load_unaligned(reinterpret_cast<scalar_t const *>(mem)));
    }

    XSIMD_INLINE T first() const noexcept {
      alignas(arch_type::alignment()) std::array<T, size> in_buf;
      value.store_aligned(reinterpret_cast<scalar_t *>(in_buf.data()));
      return in_buf[0];
    }

    XSIMD_INLINE T get(size_t i) const noexcept(has_no_boundcheck) {
      if constexpr (!has_no_boundcheck) {
        if (i >= size) { throw std::runtime_error("Index out of bounds for lane access"); }
      }
      alignas(arch_type::alignment()) std::array<T, size> in_buf;
      value.store_aligned(reinterpret_cast<scalar_t *>(in_buf.data()));
      return in_buf[i];
    }

    complex_batch operator-() const noexcept { return complex_batch(-this->value); }

    // Update operators
    complex_batch &operator+=(complex_batch const &other) noexcept {
      this->value += other.value;
      return *this;
    };

    complex_batch &operator-=(complex_batch const &other) noexcept {
      this->value -= other.value;
      return *this;
    };
    // TODO: This is needed currently as without
    // optimizations mask in xsimd::swizzle is not a
    // compile time expression and it fails to compile.
    [[gnu::optimize("O3")]]
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
    // TODO: This is needed currently as without
    // optimizations mask in xsimd::swizzle is not a
    // compile time expression and it fails to compile.
    [[gnu::optimize("O3")]]
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

namespace xsimd {
  template <class A = default_arch, class From>
    requires(std::is_same_v<std::complex<float>, From> or std::is_same_v<std::complex<double>, From>)
  XSIMD_INLINE nda::complex_batch<From, A> load_aligned(From const *ptr) noexcept {
    return nda::complex_batch<From, A>::load_aligned(ptr);
  }

  template <class A = default_arch, class From>
    requires(std::is_same_v<std::complex<float>, From> or std::is_same_v<std::complex<double>, From>)
  XSIMD_INLINE nda::complex_batch<From, A> load_unaligned(From const *ptr) noexcept {
    return nda::complex_batch<From, A>::load_unaligned(ptr);
  }

  namespace detail {
    template <class T, class A, typename Func>
    XSIMD_INLINE nda::complex_batch<T, A> scalar_op(nda::complex_batch<T, A> const &x, Func f) noexcept {
      using batch_t              = nda::complex_batch<T, A>;
      constexpr size_t size      = batch_t::size;
      constexpr size_t alignment = batch_t::arch_type::alignment();

      alignas(alignment) std::array<T, size> in_buf;
      x.store_aligned(in_buf.data());
      alignas(alignment) std::array<T, size> out_buf;

      for (size_t i = 0; i < size; ++i) { out_buf[i] = f(in_buf[i]); }

      return batch_t::load_aligned(out_buf.data());
    }
  } // namespace detail

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> pow(nda::complex_batch<T, A> const &x, nda::complex_batch<T, A> const &y) noexcept {
    using batch_t              = nda::complex_batch<T, A>;
    constexpr size_t size      = batch_t::size;
    constexpr size_t alignment = batch_t::arch_type::alignment();

    alignas(alignment) std::array<T, size> in_buf;
    x.store_aligned(in_buf.data());

    alignas(alignment) std::array<T, size> exponent_buf;
    y.store_aligned(exponent_buf.data());

    alignas(alignment) std::array<T, size> out_buf;

    for (size_t i = 0; i < size; ++i) { out_buf[i] = std::pow(in_buf[i], exponent_buf[i]); }

    return batch_t::load_aligned(out_buf.data());
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> conj(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::conj(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> exp(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::exp(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> cos(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::cos(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> sin(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::sin(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> tan(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::tan(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> cosh(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::cosh(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> sinh(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::sinh(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> tanh(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::tanh(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> acos(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::acos(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> asin(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::asin(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> atan(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::atan(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> log(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::log(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> sqrt(nda::complex_batch<T, A> const &x) noexcept {
    return detail::scalar_op(x, [](T val) { return std::sqrt(val); });
  }

  template <class T, class A>
  XSIMD_INLINE nda::complex_batch<T, A> fma(nda::complex_batch<T, A> const &x, nda::complex_batch<T, A> const &y,
                                            nda::complex_batch<T, A> const &z) noexcept {
    return x * y + z;
  }

  template <class T, class A>
  XSIMD_INLINE T reduce_add(nda::complex_batch<T, A> const &x) noexcept {
    using batch_t              = nda::complex_batch<T, A>;
    constexpr size_t size      = batch_t::size;
    constexpr size_t alignment = batch_t::arch_type::alignment();

    alignas(alignment) std::array<T, size> in_buf;
    x.store_aligned(in_buf.data());
    T acc = 0;

    for (size_t i = 0; i < size; ++i) { acc += in_buf[i]; }

    return acc;
  }

  template <class T, class A>
  XSIMD_INLINE T reduce_mul(nda::complex_batch<T, A> const &x) noexcept {
    using batch_t              = nda::complex_batch<T, A>;
    constexpr size_t size      = batch_t::size;
    constexpr size_t alignment = batch_t::arch_type::alignment();

    alignas(alignment) std::array<T, size> in_buf;
    x.store_aligned(in_buf.data());
    T acc = 1;

    for (size_t i = 0; i < size; ++i) { acc *= in_buf[i]; }

    return acc;
  }

  //Traits:
  template <class T, class A>
  struct is_batch_complex<nda::complex_batch<T, A>> : std::true_type {};

} // namespace xsimd