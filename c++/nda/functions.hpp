#pragma once
namespace nda {

  template <typename A, typename B>
  bool operator==(A const &a, B const &b) {
    static_assert(std::is_same_v<get_value_t<A>, get_value_t<B>> and std::is_integral_v<get_value_t<A>>, "Only make sense for integers ");
    EXPECTS(a.shape()== b.shape());
    bool r = true;
    nda::for_each(a.shape(), [&](auto &&... x) { r &= (a(x...) == b(x...)); });
    return r;
  }

} // namespace nda


