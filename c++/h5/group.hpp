#pragma once
#include "./file.hpp"
#include "./scheme.hpp"

namespace h5 {

  /**
   *  A HDF5 group
   */
  class group : public h5_object {

    public:
    // FIXME Do we need this ?
    //group() = default; // for python converter only

    /// Takes the "/" group at the top of the file
    group(h5::file f);

    ///
    group(group const &) = default;

    private:
    /*
     * Takes ownership of the id [expert only]
     * id can be :
     *  - a file : in this case, make a group on /
     *  - a group : in this case, take the id of the group. DOES NOT take ownership of the ref
     */
    group(hid_t id_);

    // [expert only]. If not present, the obj is casted to hid_t and there is a ref. leak
    group(h5_object obj);

    public:
    /// Name of the group
    std::string name() const;

    ///
    bool has_key(std::string const &key) const;

    ///
    void unlink_key_if_exists(std::string const &key) const;

    /**
   * \brief Open a subgroup.
   * \param key  The name of the subgroup. If empty, return this group.
   *
   * Throws if it does not exist.
   */
    group open_group(std::string const &key) const;

    /// Open an existing DataSet. Throw if it does not exist.
    dataset open_dataset(std::string const &key) const;

    /**
   * \brief Create a subgroup.
   * \param key  The name of the subgroup. If empty, return this group.
   * \param delete_if_exists  Unlink the group if it exists
   */
    group create_group(std::string const &key, bool delete_if_exists = true) const;

    /**
   * \brief Create a dataset.
   * \param key The name of the subgroup
   *
   * NB : It unlinks the dataset if it exists.
   */
    dataset create_dataset(std::string const &key, datatype ty, dataspace sp, hid_t pl) const;

    /**
   * \brief Create a dataset.
   * \param key The name of the subgroup
   *
   * NB : It unlinks the dataset if it exists.
   */
    dataset create_dataset(std::string const &key, datatype ty, dataspace sp) const;

    /// Returns all names of subgroup of  G
    std::vector<std::string> get_all_subgroup_names() const;

    /// Returns all names of dataset of G
    std::vector<std::string> get_all_dataset_names() const;

    /// Returns all names of dataset of G
    std::vector<std::string> get_all_subgroup_dataset_names() const;
  };

  //------------- read iff a key exists ------------------

  template <typename T>
  inline int h5_try_read(group fg, std::string key, T &t) {
    if (fg.has_key(key)) {
      h5_read(fg, key, t);
      return 1;
    }
    return 0;
  }

} // namespace h5
