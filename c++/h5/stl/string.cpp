#include "./string.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <vector>

namespace h5 {

  static datatype str_datatype(long size) {
    datatype dt = H5Tcopy(H5T_C_S1);
    auto err    = H5Tset_size(dt, size);
    if (err < 0) throw std::runtime_error("Internal error in H5Tset_size");
    return dt;
  }

  // ------------------------------------------------------------------

  void h5_write(group g, std::string const &name, std::string const &value) {

    datatype dt     = str_datatype(value.size() + 1);
    dataspace space = H5Screate(H5S_SCALAR);
    // FIXME : remove create_dataset
    dataset ds      = g.create_dataset(name, dt, space);

    auto err = H5Dwrite(ds, dt, H5S_ALL, H5S_ALL, H5P_DEFAULT, (void *)(value.c_str()));
    if (err < 0) throw std::runtime_error("Error writing the string named" + name + " in the group" + g.name());
  }

  // ------------------------------------------------------------------

  void h5_write_attribute(hid_t id, std::string const &name, std::string const &value) {

    datatype dt     = str_datatype(value.size() + 1);
    dataspace space = H5Screate(H5S_SCALAR);

    attribute attr = H5Acreate2(id, name.c_str(), dt, space, H5P_DEFAULT, H5P_DEFAULT);
    if (!attr.is_valid()) throw std::runtime_error("Cannot create the attribute " + name);

    herr_t err = H5Awrite(attr, dt, (void *)(value.c_str()));
    if (err < 0) throw std::runtime_error("Cannot write the attribute " + name);
  }

  // -------------------- Read ----------------------------------------------

  void h5_read(group g, std::string const &name, std::string &value) {
    dataset ds            = g.open_dataset(name);
    h5::dataspace d_space = H5Dget_space(ds);
    int rank              = H5Sget_simple_extent_ndims(d_space);
    if (rank != 0) throw std::runtime_error("Reading a string and got rank !=0");
    size_t size = H5Dget_storage_size(ds);

    datatype dt = str_datatype(size);

    std::vector<char> buf(size + 1, 0x00);
    auto err = H5Dread(ds, dt, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buf[0]);
    if (err < 0) throw std::runtime_error("Error reading the string named" + name + " in the group" + g.name());

    value = "";
    value.append(&(buf.front()));
  }

  // -------------------- Read ----------------------------------------------

  /// Return the attribute name of obj, and "" if the attribute does not exist.
  void h5_read_attribute(hid_t id, std::string const &name, std::string &s) {
    s = "";

    // if the attribute is not present, return 0
    if (H5LTfind_attribute(id, name.c_str()) == 0) return; // not present

    attribute attr = H5Aopen(id, name.c_str(), H5P_DEFAULT);
    if (!attr.is_valid()) throw std::runtime_error("Cannot open the attribute " + name);

    dataspace space = H5Aget_space(attr);

    int rank = H5Sget_simple_extent_ndims(space);
    if (rank != 0) throw std::runtime_error("Reading a string attribute and got rank !=0");

    datatype strdatatype = H5Aget_type(attr);

    std::vector<char> buf(H5Aget_storage_size(attr) + 1, 0x00);
    auto err = H5Aread(attr, strdatatype, (void *)(&buf[0]));
    if (err < 0) throw std::runtime_error("Cannot read the attribute " + name);

    s.append(&(buf.front()));
  }

} // namespace h5
