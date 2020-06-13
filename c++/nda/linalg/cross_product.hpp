#pragma once

namespace nda::linalg {

  /** Cross product. Dim 3 only */
  template <typename V>
  auto cross_product(V const &a, V const &b) {
    EXPECTS_WITH_MESSAGE(a.shape()[0] == 3,
                         "nda::linalg::cross_product : works only in d=3 while you gave a vector of size " + std::to_string(a.shape()[0]));
    EXPECTS_WITH_MESSAGE(b.shape()[0] == 3,
                         "nda::linalg::cross_product : works only in d=3 while you gave a vector of size " + std::to_string(b.shape()[0]));
    array<get_value_t<V>, 1> r(3);
    r(0) = a(1) * b(2) - b(1) * a(2);
    r(1) = -a(0) * b(2) + b(0) * a(2);
    r(2) = a(0) * b(1) - b(0) * a(1);
    return r;
  }

} // namespace nda::linalg
