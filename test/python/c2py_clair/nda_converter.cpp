#include <c2py/c2py.hpp>
#include <nda/nda.hpp>

#include <complex>
#include <string>
#include <vector>

// -- arg: array<T, R> const & ----------------------------------------

double sum_array(nda::array<double, 1> const &a) { return nda::sum(a); }

double sum_matrix(nda::array<double, 2> const &a) { return nda::sum(a); }

// -- arg: array<T, R> by value ----------------------------------------

nda::array<double, 1> double_array(nda::array<double, 1> a) {
  a *= 2;
  return a;
}

// -- arg: array<T, 2> by value (for layout tests) ----------------------

double get_01(nda::array<double, 2> a) { return a(0, 1); }

// -- arg: array_view<T, R> (mutable) ----------------------------------

void fill_view_1d(nda::array_view<double, 1> v, double val) { v = val; }

void fill_view_2d(nda::array_view<double, 2> m, double val) { m = val; }

// -- arg: array_const_view<T, R> --------------------------------------

double sum_const_view(nda::array_const_view<double, 1> v) { return nda::sum(v); }

// -- return: array<T, R> by value -------------------------------------

nda::array<double, 1> make_sequence(long n) {
  nda::array<double, 1> a(n);
  nda::for_each(a.shape(), [&](auto i) { a(i) = static_cast<double>(i); });
  return a;
}

// -- return: array const & and array & (via class) --------------------

class array_container {
  nda::array<double, 2> data_;

  public:
  array_container(long rows, long cols) : data_(rows, cols) { data_ = 0; }

  nda::array<double, 2> const &data_const() const { return data_; }
  nda::array<double, 2> &data() { return data_; }
  nda::array<double, 2> data_copy() const { return data_; }
  void set_data(nda::array_view<double, 2> v) { data_ = v; }
};

// -- return: array of arrays (via class) ------------------------------

class nested_container {
  nda::array<nda::array<double, 1>, 1> data_;

  public:
  nested_container(long n, long inner_size) : data_(n) {
    nda::for_each(data_.shape(), [&](auto i) {
      data_(i) = nda::array<double, 1>(inner_size);
      data_(i) = static_cast<double>(i);
    });
  }

  nda::array<nda::array<double, 1>, 1> const &data_const() const { return data_; }
  nda::array<nda::array<double, 1>, 1> &data() { return data_; }
  nda::array<nda::array<double, 1>, 1> data_copy() const { return data_; }
};

// -- return: expressions ----------------------------------------------

auto scale_array(nda::array<double, 1> const &a, double s) { return a * s; }
auto negate_array(nda::array<double, 1> const &a) { return -a; }
auto conj_array(nda::array<std::complex<double>, 1> const &a) { return conj(a); }
auto add_arrays(nda::array<long, 2> const &a, nda::array<long, 2> const &b) { return a + b; }

// -- matrix algebra (matrix_view operator= has different semantics) ---

void fill_matrix_view(nda::matrix_view<double> m, double val) { m = val; }

// -- scalar types: long, complex<double> ------------------------------

long sum_int_array(nda::array<long, 1> const &a) { return nda::sum(a); }

std::complex<double> sum_complex_array(nda::array<std::complex<double>, 1> const &a) { return nda::sum(a); }

// -- higher rank (3D) -------------------------------------------------

double sum_3d(nda::array<double, 3> const &a) { return nda::sum(a); }

// -- non-npy element types (converter element-by-element path) --------

nda::array<std::string, 1> reverse_strings(nda::array<std::string, 1> const &a) {
  nda::array<std::string, 1> res(a.shape());
  nda::for_each(a.shape(), [&](auto i) { res(i) = std::string(a(i).rbegin(), a(i).rend()); });
  return res;
}

nda::array<std::vector<double>, 1> make_ranges(long n) {
  nda::array<std::vector<double>, 1> res(n);
  nda::for_each(res.shape(), [&](auto i) {
    auto &v = res(i);
    v.resize(i + 1);
    for (long j = 0; j <= i; ++j) v[j] = static_cast<double>(j);
  });
  return res;
}

std::vector<double> flatten_array_of_vectors(nda::array<std::vector<double>, 1> const &a) {
  std::vector<double> res;
  nda::for_each(a.shape(), [&](auto i) { res.insert(res.end(), a(i).begin(), a(i).end()); });
  return res;
}

nda::array<std::vector<int>, 2> make_grid(long rows, long cols) {
  nda::array<std::vector<int>, 2> res(rows, cols);
  nda::for_each(res.shape(), [&](auto i, auto j) { res(i, j) = {static_cast<int>(i), static_cast<int>(j)}; });
  return res;
}

long count_elements_2d(nda::array<std::vector<int>, 2> const &a) {
  long count = 0;
  nda::for_each(a.shape(), [&](auto i, auto j) { count += static_cast<long>(a(i, j).size()); });
  return count;
}

// -- matrix<T> and vector<T> aliases -----------------------------------

double sum_nda_vector(nda::vector<double> const &v) { return nda::sum(v); }

nda::matrix<double> make_matrix(long rows, long cols) {
  nda::matrix<double> m(rows, cols);
  nda::for_each(m.shape(), [&](auto i, auto j) { m(i, j) = static_cast<double>(i * cols + j); });
  return m;
}

// -- by-value round-trip for complex and long ----------------------------

nda::array<std::complex<double>, 1> scale_complex_array(nda::array<std::complex<double>, 1> a, std::complex<double> s) {
  a *= s;
  return a;
}

nda::array<long, 1> double_int_array(nda::array<long, 1> a) {
  a *= 2;
  return a;
}

// -- views for long and complex types ------------------------------------

void fill_view_long(nda::array_view<long, 1> v, long val) { v = val; }

void fill_view_complex(nda::array_view<std::complex<double>, 1> v, std::complex<double> val) { v = val; }

// -- bool arrays ---------------------------------------------------------

long count_true(nda::array<bool, 1> const &a) {
  long count = 0;
  nda::for_each(a.shape(), [&](auto i) {
    if (a(i)) ++count;
  });
  return count;
}

nda::array<bool, 1> negate_bools(nda::array<bool, 1> a) {
  nda::for_each(a.shape(), [&](auto i) { a(i) = !a(i); });
  return a;
}

#include "nda_converter.wrap.cxx"
