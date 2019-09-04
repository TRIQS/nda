#pragma once
#include "./group.hpp"
#include "./array_interface.hpp"
namespace h5 {

  namespace details {

    template <typename S> h5_array_view h5_array_view_from_scalar(S &&s) { return {hdf5_type<S>, (void *)(&s), 0}; }
  } // namespace details

  template <typename T> std::enable_if_t<std::is_arithmetic_v<T>> h5_write(group g, std::string const &name, T const &x) {
    details::write(g, name, details::h5_array_view_from_scalar(x));
  }

  template <typename T> std::enable_if_t<std::is_arithmetic_v<T>> h5_read(group g, std::string const &name, T &x) {
    details::read(g, name, details::h5_array_view_from_scalar(x), details::get_h5_lengths_type(g, name));
  }

  template <typename T> std::enable_if_t<std::is_arithmetic_v<T>> h5_write_attribute(hid_t id, std::string const &name, T const &x) {
    details::write_attribute(id, name, details::h5_array_view_from_scalar(x));
  }

  template <typename T> std::enable_if_t<std::is_arithmetic_v<T>> h5_read_attribute(hid_t id, std::string const &name, T &x) {
    details::read_attribute(id, name, details::h5_array_view_from_scalar(x));
  }

} // namespace h5
