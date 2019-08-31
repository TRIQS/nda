/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <array>
#include <utility>

// missing part of std ...
namespace nda {

  template <typename T, size_t... Is>
  constexpr std::array<T, sizeof...(Is)> make_initialized_array_impl(T v, std::index_sequence<Is...>) {
    return {(Is ? v : v)...};
  } // always v, just a trick to have the pack

  /**
   * @tparam R 
   * @tparam T
   * make a std::array<T, R> initialized to v
   */
  template <int R, typename T>
  constexpr std::array<T, R> make_initialized_array(T v) {
    return make_initialized_array_impl(v, std::make_index_sequence<R>{});
  }

} // namespace nda
/*


namespace triqs::utility {

  // missing serialization : external
  //friend class boost::serialization::access;
  //template<class Archive> void serialize(Archive & ar, const unsigned int version) { ar & TRIQS_MAKE_NVP("_data",_data); }

  // implement serialization for std::array
  template <class Archive, class T, std::size_t N> void serialize(Archive &ar, std::array<T, N> &a, const unsigned int ) {
    ar &*static_cast<T(*)[N]>(static_cast<void *>(a.data()));
  }

  ///construct on a std::vector
  template <typename T2> std::array(const std::vector<T2> &v) {
    if (v.size() != R) TRIQS_RUNTIME_ERROR << "std::array construction : vector size incorrect  : expected " << R << " got : " << v.size();
    for (int i = 0; i < R; ++i) _data[i] = v[i];
  }

  template <typename T2> std::array &operator=(const std::array<T2, R> &x) {
    for (int i = 0; i < R; ++i) _data[i] = x[i];
    return *this;
  }

  std::array &operator=(T x) {
    for (int i = 0; i < R; ++i) _data[i] = x;
    return *this;
  }

  ///conversion to std::vector
  std::vector<T> to_vector() const {
    std::vector<T> V(R);
    for (int i = 0; i < R; ++i) V[i] = _data[i];
    return V;
  }

  ///append element to std::array (increases rank by 1)
  std::array<T, R + 1> append(T const &x) const {
    std::array<T, R + 1> res;
    for (int i = 0; i < R; ++i) res[i] = _data[i];
    res[R] = x;
    return res;
  }

  ///prepend element to std::array (increases rank by 1)
  template <typename T, int R> std::array<T, R + 1> front_append(std::array<T, R> const &a, T const &x) {
    std::array<T, R + 1> res;
    res[0] = x;
    for (int i = 0; i < R; ++i) res[i + 1]= a[i];
    return res;
  }

  std::array<T, R - 1> pop() const {
    std::array<T, R - 1> res;
    for (int i = 0; i < R - 1; ++i) res[i] = _data[i];
    return res;
  }

  std::array<T, R - 1> front_pop() const {
    std::array<T, R - 1> res;
    for (int i = 1; i < R; ++i) res[i - 1] = _data[i];
    return res;
  }

  template <int N> std::array<T, R - N> front_mpop() const {
    std::array<T, R - N> res;
    for (int i = N; i < R; ++i) res[i - N] = _data[i];
    return res;
  }

  // return this
  friend std::ostream &operator<<(std::ostream &out, std::array const &v) { return out << v.to_string(); }
  friend std::stringstream &operator<<(std::stringstream &out, std::array const &v) {
    out << v.to_string();
    return out;
  }

  // used only twice in h5
  std::string to_string() const {
    std::stringstream fs;
    fs << "(";
    for (int i = 0; i < R; ++i) fs << (i == 0 ? "" : " ") << (*this)[i];
    fs << ")";
    return fs.str();
  }

  // ------------- + ou - --------------------------------------

  template <typename T, int R> std::array<T, R> operator+(std::array<T, R> const &a1, std::array<T, R> const &a2) {
    std::array<T, R> r;
    for (int i = 0; i < R; ++i) r[i] = a1[i] + a2[i];
    return r;
  }

  template <typename T, int R> std::array<T, R> operator-(std::array<T, R> const &a1, std::array<T, R> const &a2) {
    std::array<T, R> r;
    for (int i = 0; i < R; ++i) r[i] = a1[i] - a2[i];
    return r;
  }

  // ------------- join  --------------------------------------
  template <typename T, typename U, size_t R1, size_t R2> std::array<T, R1 + R2> join(std::array<T, R1> const &a1, std::array<U, R2> const &a2) {
    std::array<T, R1 + R2> res;
    for (int i = 0; i < R1; ++i) res[i] = a1[i];
    for (int i = 0; i < R2; ++i) res[R1 + i] = a2[i];
    return res;
  }

  // ------------- dot --------------------------------------
  template <typename T, typename U, size_t R> auto dot_product(std::array<T, R> const &a1, std::array<U, R> const &a2) {
    if constexpr (R > 0) {
      auto res = a1[0] * a2[0];
      for (int i = 1; i < R; ++i) res += a1[i] * a2[i];
      return res;
    } else
      return 0;
  }


  /// REMOVE THIS : only use ONCE ! Look for sum as well
  // ------------- product_of_elements --------------------------------------
  template <typename T, size_t R> T product_of_elements(std::array<T, R> const &a) {
    auto res = a[0];
    for (int i = 1; i < R; ++i) res *= a[i];
    return res;
  }

  // ------------- sum --------------------------------------
  template <typename T, size_t R> T sum(std::array<T, R> const &a) {
    auto res = a[0];
    for (int i = 1; i < R; ++i) res += a[i];
    return res;
  }

  // ------- index maker ----------------

  template <typename... U> std::array<long, sizeof...(U)> mindex(U... x) {
    static_assert((std::is_integral<U>::value && ...), "Every argument of the mindex function must be an integer type");
    return std::array<long, sizeof...(U)>(x...); // better error message than with { }
  }

} // namespace triqs::utility
*/
