#pragma once
#include "../group.hpp"
#include <string>

namespace h5 {

  H5_SPECIALIZE_FORMAT2(std::string, string);

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

  // char_buf contains an n dimensional array of strings as fixed size strings, flatten in a 1d array of char.
  // the last dimension is the max length of the strings + 1, because of the ending 0 in C !
  struct char_buf {
    std::vector<char> buffer;
    v_t lengths;

    // the string datatype
    datatype dtype() const;

    // the dataspace (without last dim, which is the string).
    dataspace dspace() const;
  };

  // technical functions
  void h5_write(group g, std::string const &name, char_buf const &cb);
  void h5_write_attribute(hid_t id, std::string const &name, char_buf const &cb);
  void h5_read(group g, std::string const &name, char_buf &_cb);
  void h5_read_attribute(hid_t id, std::string const &name, char_buf &_cb);

} // namespace h5
