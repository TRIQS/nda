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

  template <char OP, typename LHS, typename RHS> void _assignment_delegation(LHS &lhs, RHS const &rhs) {

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

#ifdef TRIQS_ARRAYS_DEBUG
      if constexpr (rhs_is_isp) {
        if (!indexmaps::compatible_for_assignment(lhs.indexmap(), rhs.indexmap()))
          TRIQS_RUNTIME_ERROR << "Size mismatch in operation " << OP << " : LHS " << lhs << " \n RHS = " << rhs;
      }
#endif
      // FIXME : use overload !!  _nda<U, Rank, ....> : NO !!! EXpressions !!!
      // FIXME : DYNAMICAL ?
      if constexpr (rhs_is_isp and (OP == 'E')) {
        if (indexmaps::raw_copy_possible(lhs.indexmap(), rhs.indexmap())) {
          storages::memcopy(lhs.data_start(), rhs.data_start(), rhs.indexmap().domain().number_of_elements());
        } else {
          _foreach(lhs, [&lhs, &rhs](auto const &... args) { lhs(args...) = rhs(args...); });
        }
      } else { // constexpr
        auto l = [&lhs, &rhs](auto const &... args) {
          if constexpr (OP == 'E') { lhs(args...) = rhs(args...); }
          if constexpr (OP == 'A') { lhs(args...) += rhs(args...); }
          if constexpr (OP == 'S') { lhs(args...) -= rhs(args...); }
          if constexpr (OP == 'M') { lhs(args...) *= rhs(args...); }
          if constexpr (OP == 'D') { lhs(args...) /= rhs(args...); }
        };
        _foreach(lhs, l);
      }
    }

    // RHS is a scalar for LHS
    else {
      // if LHS is a matrix, the unit has a specific interpretation.
      if constexpr (MutableMatrix<LHS>::value and (OP == 'A' || OP == 'S' || OP == 'E')) {
        auto l = [&lhs, &rhs](auto &&x1, auto &&x2) {
          if constexpr (OP == 'A') {
            if (x1 == x2) lhs(x1, x2) += rhs;
          }
          if constexpr (OP == 'S') {
            if (x1 == x2) lhs(x1, x2) -= rhs;
          }
          if constexpr (OP == 'E') {
            if (x1 == x2)
              lhs(x1, x2) = rhs;
            else {
              if constexpr (is_scalar_or_convertible<RHS>::value)
                lhs(x1, x2) = 0;
              else
                lhs(x1, x2) = RHS{0 * rhs}; //FIXME : improve this
            }
          }
        }; // end lambda l
        _foreach(lhs, l);
      } else { // LHS is not a matrix
        auto l = [](auto &&... args) {
          if constexpr (OP == 'E') { lhs(args...) = rhs; }
          if constexpr (OP == 'A') { lhs(args...) += rhs; }
          if constexpr (OP == 'S') { lhs(args...) -= rhs; }
          if constexpr (OP == 'M') { lhs(args...) *= rhs; }
          if constexpr (OP == 'D') { lhs(args...) /= rhs; }
        };
        _foreach(lhs, l);
      }
    }
  }
} // namespace triqs::arrays
