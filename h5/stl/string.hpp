#pragma once
#include "../group.hpp"
#include <string>

namespace h5 {

  TRIQS_SPECIALIZE_HDF5_SCHEME2(std::string, string);

  /**
  * \brief Write a string  into an hdf5 file
  * \param f The h5 file or group
  * \param name The name of the hdf5 array in the file/group where the stack will be stored
  * \param value The string
  */
  void h5_write(group g, std::string const &name, std::string const &value);

  inline void h5_write(group g, std::string const &name, const char *s) { h5_write(g, name, std::string{s}); }

  /**
  * \brief Read a string from an hdf5 file
  * \param f The h5 file or group
  * \param name The name of the hdf5 array in the file/group where the stack will be stored
  * \param value The string to fill
  */
  void h5_read(group g, std::string const &name, std::string &value);

  inline void h5_read(group g, std::string const &name, char *s) = delete;

  /**
  * \brief Read a string from an hdf5 file
  * \param f The h5 file or group
  * \param name The name of the hdf5 array in the file/group where the stack will be stored
  * \param value The string.
  */
  void h5_write_attribute(hid_t id, std::string const &name, std::string const &value);

  inline void h5_write_attribute(hid_t id, std::string const &name, const char *s) { h5_write_attribute(id, name, std::string{s}); }

  /**
  * \brief Read a string attribute from id.
  * \param id  The object to which the attribute is attached
  * \param name The name of the attribute
  * \param value The string to fill
  */
  void h5_read_attribute(hid_t id, std::string const &name, std::string &s);

  inline void h5_read_attribute(hid_t id, std::string const &name, char *s) = delete;

} // namespace h5
