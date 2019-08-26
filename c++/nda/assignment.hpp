#pragma once
#include <cstring>
#include "./traits.hpp"
#include "./concepts.hpp"

//https://godbolt.org/g/tkNLoE
// https://godbolt.org/g/SqrgBH

namespace nda::details {

  // implementation of the assignment operator of the containers
  template <typename LHS, typename RHS> void assignment(LHS &lhs, RHS const &rhs) {

    static_assert(!LHS::is_const, "Cannot assign to a const view !");
    static_assert(std::is_assignable_v<typename LHS::value_t, typename RHS::value_t>,
                  "Assignment impossible for the type of RHS into the type of LHS");

    // if RHS is mpi_lazy
    //if constexpr (std::is_base_of_v<mpi_lazy_array_tag, RHS>) {
    //impl_assign_from_lazy_array(lhs, rhs);
    //return;
    //}

    // special case: we may have a direct memcopy
    if constexpr (is_regular_or_view_v<LHS>) {

#ifdef NDA_DEBUG
      if (lhs.indexmap().lengths() != rhs.indexmap().lengths())
        NDA_RUNTIME_ERROR << "Size mismatch in operation " << OP << " : LHS " << lhs << " \n RHS = " << rhs;
#endif
      constexpr bool can_consider_memcpy =
         std::is_trivially_copyable_v<typename LHS::value_t> and std::is_same_v<typename LHS::value_t, typename RHS::value_t>;

      if constexpr (can_consider_memcpy) {
        if ((lhs.indexmap().lengths() == rhs.indexmap().lengths()) and (lhs.indexmap().strides() == rhs.indexmap().strides())
            and (lhs.indexmap().is_contiguous())) {
          auto *p1 = lhs.data_start(), *p2 = rhs.data_start();
          long s = rhs.indexmap().size();
          if (std::abs(p2 - p1) > s) { // guard against overlapping of data
            std::memcpy(p1, p2, s * sizeof(typename LHS::value_t));
          } else {
            for (long i = 0; i < s; ++i) p1[i] = p2[i];
          }
          return;
        }
      } // non memcpy, we continue
    }

    // general case if RHS is not a scalar (can be isp, expression...)
    if constexpr (not is_scalar_for_v<RHS, LHS>) {

      //static constexpr bool rhs_is_a_container = std::is_base_of_v<tag::regular_or_view, RHS>;

      auto l = [&lhs, &rhs](auto const &... args) { lhs(args...) = rhs(args...); };
      nda::for_each(lhs.indexmap().lengths(), l);
    }

    else {
      // RHS is a scalar for LHS
      // if LHS is a matrix, the unit has a specific interpretation.

      if constexpr (is_matrix_regular_v<LHS> or is_matrix_view_v<LHS>) {
        // FIXME : Foreach on diagonal only !
        // WRITE THE LLOP !SAME in compound op !!
        auto l = [&lhs, &rhs](auto &&x1, auto &&x2) {
          if (x1 == x2)
            lhs(x1, x2) = rhs;
          else {
            if constexpr (is_scalar_or_convertible_v<RHS>)
              lhs(x1, x2) = 0;
            else
              lhs(x1, x2) = RHS{0 * rhs}; //FIXME : improve this
          }
        }; // end lambda l
        nda::for_each(lhs.indexmap().lengths(), l);
      } else { // LHS is not a matrix
        auto l = [&lhs, &rhs](auto const &... args) { lhs(args...) = rhs; };
        nda::for_each(lhs.indexmap().lengths(), l);
      }
    }
  }
} // namespace nda::details
