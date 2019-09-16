#include "./file.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <vector>

using namespace std::string_literals;

#define CHECK_OR_THROW(Cond, Mess)                                                                                                                   \
  if (!(Cond)) throw std::runtime_error("Error in h5 (de)serialization "s + Mess);

namespace h5 {

  file::file(const char *name, char mode) {

    switch (mode) {
      case 'r': id = H5Fopen(name, H5F_ACC_RDONLY, H5P_DEFAULT); break;

      case 'w': id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); break;

      case 'a': id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT); break;
      case 'e': id = H5Fopen(name, H5F_ACC_EXCL, H5P_DEFAULT); break;
      default: throw std::runtime_error("HDF5 file opening : mode is not r, w, a, e. Cf documentation");
    }

    if (id < 0) throw std::runtime_error("HDF5 : cannot "s + (((mode == 'r') or (mode == 'a')) ? "open" : "create") + "file : "s + name);
  }

  //---------------------------------------------

  file::file(hid_t id_) : h5_object(h5_object(id_)) {}

  //---------------------------------------------

  std::string file::name() const { // same function as for group
    char _n[1];
    ssize_t size = H5Fget_name(id, _n, 1); // first call, get the size only
    std::vector<char> buf(size + 1, 0x00);
    H5Fget_name(id, buf.data(), size);// now get the name
    std::string res = "";
    res.append(&(buf.front()));
    return res;
  }

  // ======================= MEMORY FILE  ============================

  static hid_t make_memory_file() {
    proplist fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHECK_OR_THROW((fapl >= 0), "creating fapl");

    auto err = H5Pset_fapl_core(fapl, (size_t)(64 * 1024), false);
    CHECK_OR_THROW((err >= 0), "setting core file driver in fapl.");

    hid_t f = H5Fcreate("MemoryBuffer", 0, H5P_DEFAULT, fapl);
    CHECK_OR_THROW((H5Iis_valid(f)), "created core file");

    return f;
  }

  // -------------------------

  memory_file::memory_file() : file(make_memory_file()) {}

  // -------------------------

  h5_object memory_file_from_buffer(std::vector<unsigned char> const &buf) {

    proplist fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHECK_OR_THROW((fapl >= 0), "creating fapl");

    auto err = H5Pset_fapl_core(fapl, (size_t)(64 * 1024), false);
    CHECK_OR_THROW((err >= 0), "setting core file driver in fapl.");

    err = H5Pset_file_image(fapl, (void *)buf.data(), buf.size());
    CHECK_OR_THROW((err >= 0), "set file image in fapl.");

    h5_object f = H5Fopen("MemoryBuffer", H5F_ACC_RDONLY, fapl);
    CHECK_OR_THROW((f.is_valid()), "opened received file image file");

    return f;
  }

  // -------------------------

  std::vector<unsigned char> memory_file::as_buffer() const {

    auto f   = hid_t(*this);
    auto err = H5Fflush(f, H5F_SCOPE_GLOBAL);
    CHECK_OR_THROW((err >= 0), "flushed core file.");

    ssize_t image_len = H5Fget_file_image(f, NULL, (size_t)0);
    CHECK_OR_THROW((image_len > 0), "got image file size");

    std::vector<unsigned char> buf(image_len, 0);

    ssize_t bytes_read = H5Fget_file_image(f, (void *)buf.data(), (size_t)image_len);
    CHECK_OR_THROW(bytes_read == image_len, "wrote file into image buffer");

    return buf;
  }

  // -----------------------------

} // namespace h5
