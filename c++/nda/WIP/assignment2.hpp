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
#include "iterator_adapter.hpp"
#include "../indexmaps/cuboid/foreach.hpp"
#include "../storages/memcopy.hpp"

//https://godbolt.org/g/tkNLoE
// https://godbolt.org/g/SqrgBH
namespace triqs::arrays {

  // FIXME : rename
  namespace Tag {
    struct indexmap_storage_pair {};
  } // namespace Tag

  template <char OP, typename LHS, typename RHS> void _assignment_delegation(LHS &lhs, RHS const &rhs) noexcept(!_global_constexpr_check_bounds) {

    static_assert(!LHS::is_const, "Cannot assign to a const view !");
    static_assert((!std::is_const<typename LHS::value_type>::value), "Assignment : The value type of the LHS is const and cannot be assigned to !");
    static_assert(std::is_assignable<typename LHS::value_type, typename RHS::value_type>::value,
                  "Assignment impossible for the type of RHS into the type of LHS");
    static_assert(
       (!((OP == 'M' || OP == 'D') && MutableMatrix<LHS>::value && (!is_scalar_for<RHS, LHS>::value))),
       "*= and /= operator for non scalar RHS are deleted for a type modeling MutableMatrix (e.g. matrix, matrix_view) matrix, because this is ambiguous");

    // if RHS is mpi_lazy
    if constexpr (std::is_base_of_v<mpi_lazy_array_tag, RHS>) {
      impl_assign_from_lazy_array(lhs, rhs);
    }

    // general case if RHS is not a scalar (can be isp, expression...)
    else if constexpr (!is_scalar_for<RHS, LHS>::value) {
      static_assert(ImmutableCuboidArray<RHS>::value, "Error");

      static constexpr bool rhs_is_isp = std::is_base_of_v<Tag::indexmap_storage_pair, RHS>;

      if constexpr (_global_constexpr_check_bounds and rhs_is_isp) {
        if (!indexmaps::compatible_for_assignment(lhs.indexmap(), rhs.indexmap()))
          TRIQS_RUNTIME_ERROR << "Size mismatch in operation " << OP << " : LHS " << lhs << " \n RHS = " << rhs;
      }

      // FIXME : use overload !!  _nda<U, Rank, ....> : NO !!! EXpressions !!!
      // FIXME : DYNAMICAL ?
      auto l = [&lhs, &rhs ](auto const &... args) [[gnu::always_inline]] {
        if constexpr (OP == 'E') { lhs(args...) = rhs(args...); }
        if constexpr (OP == 'A') { lhs(args...) += rhs(args...); }
        if constexpr (OP == 'S') { lhs(args...) -= rhs(args...); }
        if constexpr (OP == 'M') { lhs(args...) *= rhs(args...); }
        if constexpr (OP == 'D') { lhs(args...) /= rhs(args...); }
      };

      if constexpr (rhs_is_isp and (OP == 'E')) {
        if (indexmaps::raw_copy_possible(lhs.indexmap(), rhs.indexmap())) {
          storages::memcopy(lhs.data_start(), rhs.data_start(), rhs.indexmap().domain().number_of_elements());
        } else
          _foreach(lhs, l);
      } else
        _foreach(lhs, l);
    }

    // RHS is a scalar for LHS
    else {
      // if LHS is a matrix, the unit has a specific interpretation.
      if constexpr (MutableMatrix<LHS>::value and (OP == 'A' || OP == 'S' || OP == 'E')) {

        // Check the matrix is square
        if constexpr (_global_constexpr_check_bounds) {
          if (first_dim(lhs) != second_dim(lhs)) NDA_ERROR << "Not square";
        }

        if constexpr (OP == 'E') { // first put 0 everywhere
          for (long i = 0; i < first_dim(lhs); ++i)
            for (long j = 0; j < first_dim(lhs); ++j) {
              if constexpr (is_scalar_or_convertible<RHS>::value)
                lhs(i, j) = 0;
              else
                lhs(i, j) = RHS{0 * rhs}; //FIXME : improve this
            }
        }
        // Diagonal
        for (long i = 0; i < first_dim(lhs); ++i) {
          if constexpr (OP == 'E') lhs(i, i) = rhs;
          if constexpr (OP == 'A') lhs(i, i) += rhs;
          if constexpr (OP == 'S') lhs(i, i) -= rhs;
        }
      }
    }
    else { // LHS is not a matrix
      auto l = [](auto &&... args) [[gnu::always_inline]] {
        if constexpr (OP == 'E') { lhs(args...) = rhs; }
        if constexpr (OP == 'A') { lhs(args...) += rhs; }
        if constexpr (OP == 'S') { lhs(args...) -= rhs; }
        if constexpr (OP == 'M') { lhs(args...) *= rhs; }
        if constexpr (OP == 'D') { lhs(args...) /= rhs; }
      };
      _foreach(lhs, l);
    }
  }
} // namespace triqs::arrays
