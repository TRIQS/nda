#include <nda/array.hpp>
#include "./simple_read_write.hpp"

namespace nda::h5_details {

  using v_t = h5::v_t; // the proper vector type for the h5 interface

  // given the lengths and strides, return a L_tot. One function for all ranks (save code).
  // Assume layout is C.
  // use stride[rank -1]  =   strides_h5 [rank -1]
  //     stride[rank -2]  =   L[rank-1] * strides_h5 [rank -2]
  //     stride[rank -3]  =   L[rank-1] * L[rank-2] * strides_h5 [rank -3]
  //     stride[0]        =   L[rank-1] * L[rank-2] * L[1] * strides_h5 [0]

  std::pair<v_t, v_t> get_L_tot_and_strides_h5(long const *stri, int rank, long total_size) {
    v_t Ltot(rank), strides_h5(rank);
    for (int u = 0; u < rank; ++u) strides_h5[u] = stri[u];
    Ltot[0] = total_size;

    for (int u = rank - 2; u >= 0; --u) {
      // L[u+1] as  gcd of size and stride[u] ... stride[0]
      long L = strides_h5[u];
      // L becomes the  gcd
      for (int v = u - 1; v >= 0; --v) { L = std::gcd(L, strides_h5[v]); }
      // divides
      for (int v = u; v >= 0; --v) { strides_h5[v] /= L; }
      Ltot[u + 1] = L;
    }

    //std::cout << " ------- RESULT ------- " << std::endl;
    //for (int u = 0; u < rank; ++u) {
      //NDA_PRINT(u);
      //NDA_PRINT(stri[u]);
      //NDA_PRINT(Ltot[u]);
      //NDA_PRINT(strides_h5[u]);
    //}
    //std::cout << "------- END RESULT --------- " << std::endl;

    return {Ltot, strides_h5};
  }

  // implementation of the write
  void write(h5::group g, std::string const &name, h5::datatype ty, void *start, int rank, bool is_complex, long const * lens, long const *strides, long total_size) {

    h5::array_interface::h5_array_view v{ty, start, rank, is_complex};

    auto [L_tot, strides_h5] = h5_details::get_L_tot_and_strides_h5(strides, rank, total_size);

    for (int u = 0; u < rank; ++u) { // size of lhs may be size of hte rhs vector + 1 if complex. Can not simply use =
      v.slab.count[u]  = lens[u];
      v.slab.stride[u] = strides_h5[u];
      v.L_tot[u]       = L_tot[u];
    }

    h5::array_interface::write(g, name, v, true);
  }
} // namespace nda::h5_details
