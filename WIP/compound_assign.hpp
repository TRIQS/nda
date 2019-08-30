#pragma once
namespace nda {

  template <char OP, typename LHS, typename RHS> void compound_assign_impl(LHS &lhs, RHS const &rhs) {

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

      // MACRO : if constexpr???
      if constexpr (_global_constexpr_check_bounds and rhs_is_isp) {
        if (!indexmaps::compatible_for_assignment(lhs.indexmap(), rhs.indexmap()))
          TRIQS_RUNTIME_ERROR << "Size mismatch in operation " << OP << " : LHS " << lhs << " \n RHS = " << rhs;
      }

      // FIXME : use overload !!  _nda<U, Rank, ....> : NO !!! EXpressions !!!
      // FIXME : DYNAMICAL ?
      auto l = [&lhs, &rhs ](auto const &... args) [[gnu::always_inline]] {
        if constexpr (OP == 'A') { lhs(args...) += rhs(args...); }
        if constexpr (OP == 'S') { lhs(args...) -= rhs(args...); }
        if constexpr (OP == 'M') { lhs(args...) *= rhs(args...); }
        if constexpr (OP == 'D') { lhs(args...) /= rhs(args...); }
      };
      _foreach(lhs, l);
    }

    // RHS is a scalar for LHS
    else {
      // if LHS is a matrix, the unit has a specific interpretation.
      if constexpr (MutableMatrix<LHS>::value and (OP == 'A' || OP == 'S' || OP == 'E')) {
        auto l = [&lhs, &rhs ](auto &&x1, auto &&x2) [[gnu::always_inline]] {
          if constexpr (OP == 'A') {
            if (x1 == x2) lhs(x1, x2) += rhs;
          }
          if constexpr (OP == 'S') {
            if (x1 == x2) lhs(x1, x2) -= rhs;
          }
        }; // end lambda l
        _foreach(lhs, l);
      } else { // LHS is not a matrix
        auto l = [&lhs, &rhs ](auto &&x1, auto &&x2) [[gnu::always_inline]] {
          if constexpr (OP == 'A') { lhs(args...) += rhs; }
          if constexpr (OP == 'S') { lhs(args...) -= rhs; }
          if constexpr (OP == 'M') { lhs(args...) *= rhs; }
          if constexpr (OP == 'D') { lhs(args...) /= rhs; }
        };
        _foreach(lhs, l);
      }
    }
  }
} // namespace nda
