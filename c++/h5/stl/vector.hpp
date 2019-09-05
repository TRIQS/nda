#pragma once
#include <vector>
#include <complex>
#include "../group.hpp"
#include "./string.hpp"
#include "../scalar.hpp"

namespace h5 {

  namespace details {
    template <typename T> h5_array_view h5_array_view_from_vector(std::vector<T> const &v) {
      h5_array_view res{hdf5_type<T>, (void *)v[0].data(), 1};
      res.slab.count[0] = v.size();
      return res;
    }

  } // namespace details

  // ----------------------------------------------------------------------------

  template <typename T> void h5_write(group g, std::string const &name, std::vector<T> const &v) {
    auto gr = g.create_group(name);
    if constexpr (std::is_arithmetic_v<T> or is_complex_v<T>) {
      details::write(g, name, details::h5_array_view_from_vector(v));
    } else { // generic type
      gr.write_hdf5_scheme(v);
      for (int i = 0; i < v.size(); ++i) h5_write(gr, std::to_string(i), v[i]);
    }
  }

  // ----------------------------------------------------------------------------

  template <typename T> void h5_read(group f, std::string name, std::vector<T> &v) {
    auto g = f.open_group(name);
    if constexpr (std::is_arithmetic_v<T> or is_complex_v<T>) {
      auto lt = details::get_h5_lengths_type(g, name);
      if (lt.rank() != 1) throw make_runtime_error("h5 : reading a vector and I got an array of rank", lt.rank());
      v.resize(lt.lengths[0]);
      details::read(g, name, details::h5_array_view_from_vector(v), lt);
    } else { // generic type
      v.resize(g.get_all_dataset_names().size() + g.get_all_subgroup_names().size());
      for (int i = 0; i < v.size(); ++i) { h5_read(g, std::to_string(i), v[i]); }
    }
  }
  // ----------------------------------------------------------------------------
  // FIXME : CLEAN THIS
  // --------------   Special case of vector < string >

  TRIQS_SPECIALIZE_HDF5_SCHEME2(std::vector<std::string>, vector<string>);

  template <typename T> struct hdf5_scheme_impl<std::vector<T>> {
    static std::string invoke() { return "PythonListWrap"; } //std::vector<" + hdf5_scheme_impl<T>::invoke() + ">"; }
    //static std::string invoke() { return "std::vector<" + hdf5_scheme_impl<T>::invoke() + ">"; }
  };

  void h5_write(group f, std::string const &name, std::vector<std::string> const &V);
  void h5_read(group f, std::string const &name, std::vector<std::string> &V);

  void h5_write_attribute(hid_t ob, std::string const &name, std::vector<std::vector<std::string>> const &V);
  void h5_read_attribute(hid_t ob, std::string const &name, std::vector<std::vector<std::string>> &V);

  //void h5_write_attribute (hid_t ob, std::string const & name, std::vector<std::string> const & V);
  //void h5_read_attribute (hid_t ob, std::string const & name, std::vector<std::string> & V);

} // namespace h5
