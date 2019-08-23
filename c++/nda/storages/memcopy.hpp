#pragma once

namespace nda::mem {

  namespace details {

    // beware : complex is not a pod
    template <class S>
    struct is_scalar_or_pod : std::integral_constant<bool, std::is_arithmetic<S>::value || triqs::is_complex<S>::value || std::is_pod<S>::value> {};

    // copy such that it is a fast memcpy for scalar / pod objects
    // when not a scalar object, loop on elements

    template <typename T1, typename T2> void memcopy_impl(T1 *restrict p1, const T2 *restrict p2, size_t size, std::false_type) {
      for (size_t i = 0; i < size; ++i) p1[i] = p2[i];
    }

    template <typename T> void memcopy_impl(T *restrict p1, const T *restrict p2, size_t size, std::true_type) {
      if (std::abs(p2 - p1) > size) {
        memcpy(p1, p2, size * sizeof(T));
      } // guard against aliasing of data
      else {
        memcopy_impl(p1, p2, size, std::false_type());
      }
    }
  } // namespace details

  template <typename T1, typename T2> void memcopy(T1 *restrict p1, const T2 *restrict p2, size_t size) {
    constexpr bool use_memcpy = std::is_pod<T1>::value && std::is_same<typename std::remove_cv<T1>::type, typename std::remove_cv<T2>::type>::value;
    details::memcopy_impl(p1, p2, size, std::integral_constant<bool, use_memcpy>());
  }

} // namespace nda::mem

