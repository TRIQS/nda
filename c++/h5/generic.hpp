#pragma once
#include <type_traits>

namespace h5 {

  // A generic read
  template <typename T> T h5_read(group gr, std::string const &name) {
    if constexpr (std::is_default_constructible_v<T>) {
      T x;
      h5_read(gr, name, x);
      return x;
    } else {
      return T::h5_read_construct(gr, name);
    }
  }

  /// Returns the attribute name of obj, and "" if the attribute does not exist.
  template <typename T> T h5_read_attribute(hid_t id, std::string const &name) {
    using h5::h5_read_attribute;
    T x;
    h5_read_attribute(id, name, x);
    return x;
  }

} // namespace h5
