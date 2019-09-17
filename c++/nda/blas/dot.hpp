/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet, P. Dumitrescu, N. Wentzell
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

#include "f77/cxx_interface.hpp"
#include "tools.hpp"
#include <triqs/arrays/impl/traits.hpp>
#include <triqs/utility/exceptions.hpp>

namespace nda::blas {

  template <bool Star, typename VTX, typename VTY>
  auto dot(VTX const &X, VTY const &Y) {
    if ((X.size() != Y.size())) NDA_RUNTIME_ERROR << "Dimension mismatch in dot size are : X : " << X.size() << " and Y : " << Y.size();

    // if both value or view and both doube
    return f77::dot(X.size(), X.data_start(), X.stride(), Y.data_start(), Y.stride());

    // if both value or view and both conplex

    std::ptrdiff_t N = X.size(), incx = X.extent(0), incy = Y.extent(0);
    auto *px = X.data_start();
    auto *py = Y.data_start();

    auto res = px[0] * py[0];

    static constexpr bool use_conj = (Star and is_complex_v<typename X::value_type>);

    if ((incx == 1) and (incy == 1)) {
      for (size_t i = 1; i < N; ++i) {
        if constexpr (use_conj) {
          res += std::conj(X_[i]) * Y_[i];
        } else {
          res += X_[i] * Y_[i];
        }
      }
      else { // code for unequal increments or equal increments  not equal to 1
        for (size_t i = 1, ix = incx, iy = incy; i < N; ++i, ix += incx, iy += incy) {
          if constexpr (use_conj) {
            res += std::conj(X_[ix]) * Y_[iy];
          } else
            res += X_[ix] * Y_[iy];
        }
      }
    } 
    // general code for the concept. Is it really slower ?  cf below
    return res;

    // ELSE
    size_t N                  = X.size();
    decltype(X(0) * Y(0)) res = 0;
    for (size_t i = 0; i < N; ++i) res += _conj<Star>(X(i)) * Y(i);
    return res;
  }

  namespace triqs::arrays {
    namespace blas {

      template <typename VTX, typename VTY>
      struct dispatch {
        static constexpr bool are_both_value_view = is_amv_value_or_view_class<VTX>::value && is_amv_value_or_view_class<VTY>::value;
        static constexpr bool are_both_double =
           std::is_same<typename VTX::value_type, double>::value && std::is_same<typename VTY::value_type, double>::value;
        static constexpr int value = (are_both_value_view ? (are_both_double ? 0 : 1) : 2);
        typedef decltype(std::declval<VTX>()(0) * std::declval<VTY>()(0)) result_type;
      };

      /**
  * Calls dot product of 2 vectors.
  * Takes care of making temporary copies if necessary
  */
      template <bool Star, typename VTX, typename VTY>
      typename std::enable_if<dispatch<VTX, VTY>::value == 0, typename dispatch<VTX, VTY>::result_type>::type dot(VTX const &X, VTY const &Y) {
        static_assert(is_amv_value_or_view_class<VTX>::value, "blas1 bindings only take vector and vector_view");
        static_assert(is_amv_value_or_view_class<VTY>::value, "blas1 bindings only take vector and vector_view");
        if ((X.size() != Y.size())) TRIQS_RUNTIME_ERROR << "Dimension mismatch in dot size are : X : " << X.size() << " and Y : " << Y.size();
        return f77::dot(X.size(), X.data_start(), X.stride(), Y.data_start(), Y.stride());
      }

      template <bool Star>
      inline std::complex<double> _conj(std::complex<double> const &x);
      template <>
      inline std::complex<double> _conj<true>(std::complex<double> const &x) {
        return conj(x);
      }
      template <>
      inline std::complex<double> _conj<false>(std::complex<double> const &x) {
        return x;
      }
      template <bool Star>
      inline double _conj(double x) {
        return x;
      }

      /**
  * Calls dot product of 2 vectors.
  * Takes care of making temporary copies if necessary
  * general case. Also for complex since there is a bug on some machines (os X, weiss...) for zdotu, zdotc...
  * a transcription from netlib zdotu
  */
      template <bool Star, typename VTX, typename VTY>
      typename std::enable_if<dispatch<VTX, VTY>::value == 1, typename dispatch<VTX, VTY>::result_type>::type dot(VTX const &X, VTY const &Y) {
        if ((X.size() != Y.size())) TRIQS_RUNTIME_ERROR << "Dimension mismatch in dot size are : X : " << X.size() << " and Y : " << Y.size();
        size_t N = X.size(), incx = X.stride(), incy = Y.stride();
        decltype(X(0) * Y(0)) res = 0;
        // This only works for object with data (ISP), not only from the concept...
        auto *X_ = X.data_start();
        auto *Y_ = Y.data_start();
        if ((incx == 1) && (incy == 1)) {
          for (size_t i = 0; i < N; ++i) res += _conj<Star>(X_[i]) * Y_[i];
        } else { // code for unequal increments or equal increments  not equal to 1
          for (size_t i = 0, ix = 0, iy = 0; i < N; ++i, ix += incx, iy += incy) { res += _conj<Star>(X_[ix]) * Y_[iy]; }
        }
        // general code for the concept. Is it really slower ?  cf below
        return res;
      }

      /**
  * Generic case
  */
      template <bool Star, typename VTX, typename VTY>
      typename std::enable_if<dispatch<VTX, VTY>::value == 2, typename dispatch<VTX, VTY>::result_type>::type dot(VTX const &X, VTY const &Y) {
        if ((X.size() != Y.size())) TRIQS_RUNTIME_ERROR << "Dimension mismatch in dot size are : X : " << X.size() << " and Y : " << Y.size();
        size_t N                  = X.size();
        decltype(X(0) * Y(0)) res = 0;
        for (size_t i = 0; i < N; ++i) res += _conj<Star>(X(i)) * Y(i);
        return res;
      }

    } // namespace blas

    template <typename VTX, typename VTY>
    auto dot(VTX const &X, VTY const &Y) DECL_AND_RETURN(blas::dot<false>(X, Y));
    template <typename VTX, typename VTY>
    auto dotc(VTX const &X, VTY const &Y) DECL_AND_RETURN(blas::dot<true>(X, Y));

  } // namespace triqs::arrays
