#pragma once

#include <vector>
#include <string>

#include "./group.hpp"

namespace h5::array_interface {


  // Stores the hdf5 type and the dims
  struct h5_lengths_type {
    v_t lengths;
    datatype ty;
    bool has_complex_attribute;

    int rank() const { return lengths.size(); }
  };

  // Store HDF5 hyperslab info, as in HDF5 manual
  // http://davis.lbl.gov/Manuals/HDF5-1.8.7/UG/12_Dataspaces.html
  struct hyperslab {
    v_t offset; // offset of the pointer from the start of data
    v_t stride; // stride in each dimension (in the HDF5 sense). 1 if contiguous. Always >0.
    v_t count;  // length in each dimension
    v_t block;  // block in each dimension

    // Constructor : unique to enforce the proper sizes of array
    // rank : the array will have rank + is_complex
    hyperslab(int rank, bool is_complex)
       : offset(rank + is_complex, 0), stride(rank + is_complex, 1), count(rank + is_complex, 0), block() { // block is often unused
      if (is_complex) {
        stride[rank] = 1;
        count[rank]  = 2;
      }
    }

    int rank() const { return count.size(); }
  };

  // Stores a view of an array.
  // scalar are array of rank 0, lengths, strides are empty, rank is 0, start is the scalar
  struct h5_array_view {
    datatype ty;    // HDF5 type
    void *start;    // start of data. It MUST be a pointer of T* with ty = hdf5_type<T>
    v_t L_tot;      // lengths of the parent contiguous array
    hyperslab slab; // hyperslab
    bool is_complex;

    // Constructor : unique to enforce the proper sizes of array
    h5_array_view(datatype ty, void *start, int rank, bool is_complex)
       : ty(ty), start(start), L_tot(rank + is_complex), slab(rank, is_complex), is_complex(is_complex) {
      if (is_complex) L_tot[rank] = 2;
    }

    int rank() const { return slab.rank(); }
  };

  // Retrieve lengths and hdf5 type from a file
  h5_lengths_type get_h5_lengths_type(group g, std::string const &name);

  // Write the view of the array to the group
  void write(group g, std::string const &name, h5_array_view const &a, bool compress = false);

  // EXPLAIN
  void read(group g, std::string const &name, h5_array_view v, h5_lengths_type lt);

  // Read the data into the view.
  //inline void read(group g, std::string const &name, h5_array_view v) { read(g, name, v, get_h5_lengths_type(g, name)); }

  // Write as attribute
  void write_attribute(hid_t id, std::string const &name, h5_array_view v);

  // Read as attribute
  void read_attribute(hid_t id, std::string const &name, h5_array_view v);

} // namespace h5::array_interface
