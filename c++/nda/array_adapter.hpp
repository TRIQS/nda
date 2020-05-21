#pragma once

namespace nda {

  /// A pair shape + lambda --> an immutable array
  template <int R, typename F>
  class array_adapter {
    std::array<long, R> myshape;
    F f;

    public:
    template <typename Int>
    array_adapter(std::array<Int, R> const &shape, F f) : myshape(make_std_array<long>(shape)), f(f) {}

    std::array<long, R> const &shape() const { return myshape; }

    template <typename... Long>
    auto operator()(long i, Long... is) const {
      static_assert((std::is_convertible_v<Long, long> and ...), "Arguments must be convertible to long");
      return f(i, is...);
    }
  };

  // CTAD
  template <auto R, typename Int, typename F>
  array_adapter(std::array<Int, R>, F) -> array_adapter<R, F>;

  // C++17 concept emulation 
#if not __cplusplus > 201703L
  template <int R, typename F>
  inline constexpr bool is_ndarray_v<array_adapter<R, F>> = true;
#endif

} // namespace nda
