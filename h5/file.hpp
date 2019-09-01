#pragma once
#include "./h5object.hpp"

namespace h5 {

  /**
  *  @brief A little handler for the HDF5 file
  *
  *  The class is basically a pointer to the file.
  *  Hence copy is very cheap.
  */
  class file : public h5_object {

    public:
    /**
     * Open the file
     *
     * @param name  name of the file
     * @param mode  Opening mode
     *       - 'r' : Read Only (HDF5 flag H5F_ACC_RDONLY)
     *       - 'w' : Write Only (HDF5 flag H5F_ACC_TRUNC)
     *       - 'a' : Append (HDF5 flag  H5F_ACC_RDWR)
     *       - 'e' : Like 'w' but fails if the file already exists (HDF5 flag  H5F_ACC_EXCL)
     */
    file(const char *name, char mode);

    ///
    file(std::string const &name, char mode) : file(name.c_str(), mode) {}

    /// Name of the file
    std::string name() const;

    protected:
    // Internal : from an hdf5 id.
    file(hid_t id);
  };

  /**
  *  @brief A file in a memory buffer
  *  
  *
  */
  class memory_file : public file {

    public:
    /// Construct a writable file in memory with a buffer
    memory_file();

    /// Construct a read_only file on top on the buffer.
    memory_file(std::vector<unsigned char> const &buf);

    /// Get a copy of the buffer
    std::vector<unsigned char> as_buffer() const;
  };

} // namespace h5
