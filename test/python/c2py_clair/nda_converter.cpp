#include <c2py/c2py.hpp>
#include <nda/nda.hpp>
#include <complex>

namespace nc {

  // ============================================================
  // Functions taking arrays/views as const reference
  // ============================================================

  inline double sum_array(nda::array<double, 1> const &a) {
    double s = 0;
    for (long i = 0; i < a.size(); ++i) s += a(i);
    return s;
  }

  inline double sum_const_view(nda::array_const_view<double, 1> v) {
    double s = 0;
    for (long i = 0; i < v.size(); ++i) s += v(i);
    return s;
  }

  // ============================================================
  // By value (copy in C++)
  // ============================================================

  inline nda::array<double, 1> double_array(nda::array<double, 1> a) {
    a *= 2;
    return a;
  }

  // ============================================================
  // Modifying views and array refs in-place
  // ============================================================

  inline void fill_matrix(nda::array_view<double, 2> m, double val) { m = val; }

  // ============================================================
  // Returning arrays by value
  // ============================================================

  inline nda::matrix<double> make_identity(long n) { return nda::eye(n); }

  // ============================================================
  // Returning expressions (py_converter<expr> handles these)
  // ============================================================

  inline auto scale_array(nda::array<double, 1> const &a, double s) { return a * s; }

  inline auto negate_array(nda::array<double, 1> const &a) { return -a; }

  inline auto conj_array(nda::array<std::complex<double>, 1> const &a) { return conj(a); }

  inline auto add_arrays(nda::array<long, 2> const &a, nda::array<long, 2> const &b) { return a + b; }

  // ============================================================
  // Class holding an array with const& getter
  // ============================================================

  class array_container {
    nda::array<double, 2> data_;

    public:
    array_container(long rows, long cols) : data_(rows, cols) { data_ = 0; }

    nda::array<double, 2> const &data_const() const { return data_; }

    nda::array<double, 2> &data() { return data_; }

    nda::array<double, 2> data_copy() const { return data_; }

    void set_data(nda::array_view<double, 2> v) { data_ = v; }
  };

  // ============================================================
  // Different scalar types
  // ============================================================

  inline long sum_int_array(nda::array<long, 1> const &a) {
    long s = 0;
    for (long i = 0; i < a.size(); ++i) s += a(i);
    return s;
  }

  inline std::complex<double> sum_complex_array(nda::array<std::complex<double>, 1> const &a) {
    std::complex<double> s = 0;
    for (long i = 0; i < a.size(); ++i) s += a(i);
    return s;
  }

  // ============================================================
  // Higher-rank arrays (3D)
  // ============================================================

  inline double sum_3d(nda::array<double, 3> const &a) {
    double s = 0;
    for (long i = 0; i < a.shape()[0]; ++i)
      for (long j = 0; j < a.shape()[1]; ++j)
        for (long k = 0; k < a.shape()[2]; ++k) s += a(i, j, k);
    return s;
  }

  inline void fill_3d(nda::array_view<double, 3> v, double val) { v = val; }

} // namespace nc

#include "nda_converter.wrap.cxx"
