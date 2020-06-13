#include "./test_common.hpp"
// -------------------------------------

TEST(BUG, n1) { //NOLINT

  using expr_t = nda::expr<'+', nda::basic_array<std::complex<double>, 2, nda::C_layout, 'M', nda::heap> &,
                           const nda::basic_array_view<const std::complex<double>, 2, nda::basic_layout<0, 16, nda::layout_prop_e::none>, 'A',
                                                       nda::default_accessor, nda::borrowed> &>;

  EXPECT_EQ( (nda::layout_prop_e::none & nda::layout_prop_e::contiguous), nda::layout_prop_e::none);

  EXPECT_EQ(expr_t::l_is_scalar, false);
  EXPECT_EQ(expr_t::r_is_scalar, false);
 
  EXPECT_EQ(expr_t::layout_info.prop, nda::layout_prop_e::none);
 
  EXPECT_EQ(nda::get_layout_info<expr_t::R_t>.prop, nda::layout_prop_e::none);
  EXPECT_EQ(nda::get_layout_info<expr_t::L_t>.prop, nda::layout_prop_e::contiguous);

  EXPECT_EQ(nda::get_layout_info<expr_t::R_t>.prop & nda::get_layout_info<expr_t::L_t>.prop, nda::layout_prop_e::none);
  
  //EXPECT_EQ(nda::get_layout_info<expr_t>.prop, nda::layout_prop_e::none);
  //EXPECT_EQ(nda::get_layout_info<expr_t>.prop, nda::layout_prop_e::contiguous);
}


