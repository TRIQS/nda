#pragma once
#include <cstring>
#include "./traits.hpp"

//https://godbolt.org/g/tkNLoE
// https://godbolt.org/g/SqrgBH

namespace nda::details {

  // implementation of the assignment operator of the containers
  template <typename LHS, typename RHS>
  void assignment(LHS &lhs, RHS const &rhs) {

    static_assert(!LHS::is_const, "Cannot assign to a const view !");

    // if RHS is mpi_lazy
    //if constexpr (std::is_base_of_v<mpi_lazy_array_tag, RHS>) {
    //impl_assign_from_lazy_array(lhs, rhs);
    //return;
    //}

    // special case: we may have a direct memcopy
    if constexpr (is_regular_or_view_v<LHS> and is_regular_or_view_v<RHS>) {
      static_assert(std::is_assignable_v<typename LHS::value_t &, typename RHS::value_t>,
                    "Assignment impossible for the type of RHS into the type of LHS");

#ifdef NDA_DEBUG
      if (lhs.shape() != rhs.shape())
        NDA_RUNTIME_ERROR << "Size mismatch in operation " << OP << " : LHS " << lhs << " \n RHS = " << rhs;
#endif
      constexpr bool can_consider_memcpy =
         std::is_trivially_copyable_v<typename LHS::value_t> and std::is_same_v<typename LHS::value_t, typename RHS::value_t>;

      if constexpr (can_consider_memcpy) {
        // if idx_map have the same len and strides and are contiguous.
        if ((lhs.shape() == rhs.shape()) and (lhs.indexmap().strides() == rhs.indexmap().strides())
            and (lhs.indexmap().is_contiguous())) {
          auto *p1       = lhs.data_start();
          const auto *p2 = rhs.data_start();
          long s         = rhs.indexmap().size();
          if (std::abs(p2 - p1) > s) { // guard against overlapping of data
            std::memcpy(p1, p2, s * sizeof(typename LHS::value_t));
          } else {
            for (long i = 0; i < s; ++i) p1[i] = p2[i]; // not really correct. We have NO protection if data overlap.
          }
          return; // we could memcopy, we are done.
        }
      } // non memcpy, we continue
    }

    // general case if RHS is not a scalar (can be isp, expression...)
    if constexpr (not is_scalar_for_v<RHS, LHS>) {
      static_assert(std::is_assignable_v<typename LHS::value_t &, get_value_t<RHS>>,
                    "Assignment impossible for the type of RHS into the type of LHS");

      if constexpr (guarantee::has_contiguous(get_guarantee<LHS>) and guarantee::has_contiguous(get_guarantee<RHS>)) {
        // They must have the same size ! or EXPECT is wrong
        // FIXME FOR OFFSET
        // a linear computation
        long L = lhs.size();
        for (long i = 0; i < L; ++i) lhs(_linear_index_t{i}) = rhs(_linear_index_t{i});

      } else {
        auto l = [&lhs, &rhs](auto const &... args) { lhs(args...) = rhs(args...); };
        nda::for_each(lhs.shape(), l);
      }
    } else {
      // RHS is a scalar for LHS
      // if LHS is a matrix, the unit has a specific interpretation.

      if constexpr (get_algebra<LHS> == 'M') {
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
        nda::for_each(lhs.shape(), l);
      } else { // LHS is not a matrix
        auto l = [&lhs, &rhs](auto const &... args) { lhs(args...) = rhs; };
        nda::for_each(lhs.shape(), l);
      }
    }
  }

  // ===========================  compound assignment ===========================================================

  template <char OP, typename LHS, typename RHS>
  void compound_assign_impl(LHS &lhs, RHS const &rhs) {

    static_assert(!LHS::is_const, "Cannot assign to a const view !");
    static_assert((!std::is_const<typename LHS::value_t>::value), "Assignment : The value type of the LHS is const and cannot be assigned to !");
    //static_assert(std::is_assignable_v<typename LHS::value_t &, typename RHS::value_t>,
                  //"Assignment impossible for the type of RHS into the type of LHS");

    static_assert(
       (!((OP == 'M' || OP == 'D') and (get_algebra<LHS> == 'M') and (not is_scalar_for_v<RHS, LHS>))),
       "*= and /= operator for non scalar RHS are deleted for a type modeling MutableMatrix (e.g. matrix, matrix_view) matrix, because this is ambiguous");

    //// if RHS is mpi_lazy
    //if constexpr (std::is_base_of_v<mpi_lazy_array_tag, RHS>) {
    //impl_assign_from_lazy_array(lhs, rhs);
    // return
    //}

    // general case if RHS is not a scalar (can be isp, expression...)
    if constexpr (not is_scalar_for_v<RHS, LHS>) {

      static_assert(is_ndarray_v<RHS>, "Error");

#ifdef NDA_DEBUG
      if constexpr (is_regular_or_view_v<RHS>) {
        if (!indexmaps::compatible_for_assignment(lhs.indexmap(), rhs.indexmap()))
          TRIQS_RUNTIME_ERROR << "Size mismatch in operation " << OP << " : LHS " << lhs << " \n RHS = " << rhs;
      }
#endif
      auto l = [&lhs, &rhs](auto const &... args) {
        if constexpr (OP == 'A') { lhs(args...) += rhs(args...); }
        if constexpr (OP == 'S') { lhs(args...) -= rhs(args...); }
        if constexpr (OP == 'M') { lhs(args...) *= rhs(args...); }
        if constexpr (OP == 'D') { lhs(args...) /= rhs(args...); }
      };
      nda::for_each(lhs.shape(), l);
    }

    // RHS is a scalar for LHS
    else {
      // if LHS is a matrix, the unit has a specific interpretation.
      if constexpr ((get_algebra<LHS> == 'M') and (OP == 'A' || OP == 'S')) {
        if (lhs.shape()[0] != lhs.shape()[1]) NDA_RUNTIME_ERROR << "Adding a number to a matrix only works if the matrix is square !";
        long s = lhs.shape()[0];
        for (long i = 0; i < s; ++i) { // diagonal only
          if constexpr (OP == 'A') { lhs(i, i) += rhs; }
          if constexpr (OP == 'S') { lhs(i, i) -= rhs; }
        }
      } else { // LHS is not a matrix
        long s = lhs.size();
        for (long i = 0; i < s; ++i) {
          if constexpr (OP == 'A') { lhs.data_start()[i] += rhs; }
          if constexpr (OP == 'S') { lhs.data_start()[i] -= rhs; }
          if constexpr (OP == 'M') { lhs.data_start()[i] *= rhs; }
          if constexpr (OP == 'D') { lhs.data_start()[i] /= rhs; }
        }
      }
    }
  }

} // namespace nda::details
