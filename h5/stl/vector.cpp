#include "./vector.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <array>

namespace h5 {

  namespace {

    //------------------------------------------------

    // dataspace from lengths and strides. Correct for the complex. strides must be >0
    dataspace dataspace_from_LS(int rank, hsize_t const *L, hsize_t const *S) {
      hsize_t totdimsf[rank], dimsf[rank], stridesf[rank], offsetf[rank]; // dataset dimensions
      for (size_t u = 0; u < rank; ++u) {
        offsetf[u]  = 0;
        dimsf[u]    = L[u];
        totdimsf[u] = L[u];
        stridesf[u] = S[u];
      }

      dataspace ds = H5Screate_simple(rank, totdimsf, NULL);
      if (!ds.is_valid()) throw make_runtime_error("Cannot create the dataset");

      herr_t err = H5Sselect_hyperslab(ds, H5S_SELECT_SET, offsetf, stridesf, dimsf, NULL);
      if (err < 0) throw make_runtime_error("Cannot set hyperslab");

      return ds;
    }

    //------------------------------------------------

    struct aux_t {
      std::vector<char> buf;
      datatype strdatatype;
      dataspace d_space;

      aux_t(std::vector<std::string> const &V) {
        size_t s = 0;
        for (auto &x : V) s = std::max(s, x.size());

        strdatatype = H5Tcopy(H5T_C_S1);
        H5Tset_size(strdatatype, s + 1);
        // auto status = H5Tset_size (strdatatype, s);
        // auto status = H5Tset_size (strdatatype, H5T_VARIABLE);

        buf.resize(V.size() * (s + 1), 0x00);
        size_t i = 0;
        for (auto &x : V) {
          strcpy(&buf[i * (s + 1)], x.c_str());
          ++i;
        }

        hsize_t L[1] = {V.size()};
        hsize_t S[1] = {1};
        d_space      = dataspace_from_LS(1, L, S);
      }

      //-----
      aux_t(std::vector<std::vector<std::string>> const &V) {
        size_t s = 0, lv = 0;
        for (auto &v : V) {
          lv = std::max(lv, v.size());
          for (auto &x : v) s = std::max(s, x.size());
        }

        strdatatype = H5Tcopy(H5T_C_S1);
        H5Tset_size(strdatatype, s + 1);
        // auto status = H5Tset_size (strdatatype, s);
        // auto status = H5Tset_size (strdatatype, H5T_VARIABLE);

        buf.resize(V.size() * lv * (s + 1), 0x00);
        for (int i = 0, k = 0; i < V.size(); i++)
          for (int j = 0; j < lv; j++, k++) {
            if (j < V[i].size()) strcpy(&buf[k * (s + 1)], V[i][j].c_str());
          }

        // for (auto x : buf) std::cout << "buf "<< int(x) << std::endl;

        hsize_t L[2] = {V.size(), lv};
        hsize_t S[2] = {lv, 1};
        d_space      = dataspace_from_LS(2, L, S);
      }
    };
  } // namespace

  // -----------------------

  void h5_write(group g, std::string const &name, std::vector<std::string> const &V) {
    aux_t aux(V);

    h5::dataset ds = g.create_dataset(name, aux.strdatatype, aux.d_space);

    auto err = H5Dwrite(ds, aux.strdatatype, aux.d_space, H5S_ALL, H5P_DEFAULT, &aux.buf[0]);
    if (err < 0) throw make_runtime_error("Error writing the vector<string> ", name, " in the group", g.name());
  }

  // -----------------------

  void h5_write_attribute(hid_t id, std::string const &name, std::vector<std::string> const &V) {
    aux_t aux(V);

    attribute attr = H5Acreate2(id, name.c_str(), aux.strdatatype, aux.d_space, H5P_DEFAULT, H5P_DEFAULT);
    if (!attr.is_valid()) throw make_runtime_error("Cannot create the attribute ", name);

    herr_t status = H5Awrite(attr, aux.strdatatype, (void *)aux.buf.data());
    if (status < 0) throw make_runtime_error("Cannot write the attribute ", name);
  }

  // -----------------------

  void h5_write_attribute(hid_t id, std::string const &name, std::vector<std::vector<std::string>> const &V) {
    aux_t aux(V);

    attribute attr = H5Acreate2(id, name.c_str(), aux.strdatatype, aux.d_space, H5P_DEFAULT, H5P_DEFAULT);
    if (!attr.is_valid()) throw make_runtime_error("Cannot create the attribute ", name);

    herr_t status = H5Awrite(attr, aux.strdatatype, (void *)aux.buf.data());
    if (status < 0) throw make_runtime_error("Cannot write the attribute ", name);
  }

  // ----- read -----
  void h5_read(group g, std::string const &name, std::vector<std::string> &V) {
    dataset ds            = g.open_dataset(name);
    h5::dataspace d_space = H5Dget_space(ds);

    std::array<hsize_t, 1> dims_out;
    int ndims = H5Sget_simple_extent_dims(d_space, dims_out.data(), NULL);
    if (ndims != 1)
      throw make_runtime_error("triqs::h5 : Trying to read 1d array/vector . Rank mismatch : the array stored in the hdf5 file has rank = ",
                                   ndims);

    size_t Len = dims_out[0];
    V.resize(Len);
    size_t size = H5Dget_storage_size(ds);

    datatype strdatatype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strdatatype, size);
    // auto status = H5Tset_size (strdatatype, size);
    // auto status = H5Tset_size (strdatatype, H5T_VARIABLE);

    std::vector<char> buf(Len * (size + 1), 0x00);

    hsize_t L[1], S[1];
    L[0]          = V.size();
    S[0]          = 1;
    auto d_space2 = dataspace_from_LS(1, L, S);

    auto err = H5Dread(ds, strdatatype, d_space2, H5S_ALL, H5P_DEFAULT, &buf[0]);
    if (err < 0) throw make_runtime_error("Error reading the vector<string> ", name, " in the group", g.name());

    size_t i = 0;
    for (auto &x : V) {
      x = "";
      x.append(&buf[i * (size)]);
      ++i;
    }
  }

  // ----- read -----
  void h5_read_attribute(hid_t id, std::string const &name, std::vector<std::vector<std::string>> &V) {

    attribute attr = H5Aopen(id, name.c_str(), H5P_DEFAULT);
    if (!attr.is_valid()) throw make_runtime_error("Cannot open the attribute ", name);

    dataspace d_space = H5Aget_space(attr);

    std::array<hsize_t, 2> dims_out;
    int ndims = H5Sget_simple_extent_dims(d_space, dims_out.data(), NULL);
    if (ndims != 2)
      throw make_runtime_error("triqs::h5 : Trying to read vector<vector<string>>. Rank mismatch : the array stored in the hdf5 file has rank = ",
                                   ndims);

    V.resize(dims_out[0]);

    // datatype
    datatype datatype = H5Aget_type(attr);

    // buffer
    size_t size = H5Aget_storage_size(attr);
    std::vector<char> buf(size, 0x00);

    auto err = H5Aread(attr, datatype, (void *)(&buf[0]));
    if (err < 0) throw make_runtime_error("Cannot read the attribute ", name);

    // for (auto x : buf) std::cout << "bufr "<<  int(x) << std::endl;
    int s_size = size / dims_out[0] / dims_out[1]; // size of the string
    if (s_size * dims_out[0] * dims_out[1] != size) throw make_runtime_error("Error in rereading the attribute. Size mismatch of the array");
    V.clear();
    for (int i = 0, k = 0; i < dims_out[0]; ++i) {
      std::vector<std::string> v;
      for (int j = 0; j < dims_out[1]; ++j, ++k) {
        std::string x = "";
        x.append(&buf[k * (s_size)]);
        if (!x.empty()) v.push_back(x);
      }
      V.push_back(v);
    }
  }

} // namespace h5
