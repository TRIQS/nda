namespace nda {

  template <typename T> struct _is_complex : std::false_type {};
  template <typename T> struct _is_complex<std::complex<T>> : std::true_type {};

  template <typename T> inline constexpr bool is_complex_v = _is_complex<T>::value;

  template <typename S> inline constexpr bool is_scalar_v = std::is_arithmetic_v<S> or nda::is_complex_v<S>;

  template <typename S> inline constexpr bool is_scalar_or_convertible_v = is_scalar_v<S> or std::is_constructible_v<std::complex<double>, S>;

  template <typename S, typename A>
  inline constexpr bool is_scalar_for_v = (is_scalar_v<typename A::value_t> ? is_scalar_or_convertible_v<S> : std::is_same_v<S, typename A::value_t>);

} // namespace nda

