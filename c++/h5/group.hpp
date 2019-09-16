#pragma once
#include "./file.hpp"
#include "./format.hpp"

namespace h5 {

  /**
   *  HDF5 group
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

    /**
     * True iff key is a subgroup of this.
     *
     * @param key
     */
    bool has_key(std::string const &key) const;

    /**
     * Unlinks the subgroup key if it exists
     * No error is thrown if key does not exists
     * NB : unlink is almost a remove, but it does not really remove from the file (memory is not freed).
     * After unlinking a large object, a h5repack may be needed. Cf HDF5 documentation.
     *
     * @param key The name of the subgroup to be removed.
     * @param error_if_absent If True, throws an error if the key is missing.
     */
    void unlink(std::string const &key, bool error_if_absent = false) const;

    /**
     * Open a subgroup.
     * Throws std::runtime_error if it does not exist.
     *
     * @param key  The name of the subgroup. If empty, return this group
     */
    group open_group(std::string const &key) const;

    /**
     * Create a subgroup in this group
     * 
     * @param key  The name of the subgroup. If empty, return this group.
     * @param delete_if_exists  Unlink the group if it exists
     */
    group create_group(std::string const &key, bool delete_if_exists = true) const;

    /**
     * Open a existing DataSet in the group.
     * Throws std::runtime_error if it does not exist.
     *
     * @param key  The name of the subgroup. If empty, return this group
     */
    dataset open_dataset(std::string const &key) const;

    /**
     * Create a dataset in this group
     * 
     * @param key  The name of the dataset. 
     * @param ty Datatype
     * @param sp Dataspace
     */
    dataset create_dataset(std::string const &key, datatype ty, dataspace sp) const;

    /**
     * Create a dataset in this group
     * 
     * @param key  The name of the dataset. 
     * @param ty Datatype
     * @param sp Dataspace
     * @param pl Property list
     */
    dataset create_dataset(std::string const &key, datatype ty, dataspace sp, hid_t pl) const;

    /// Returns all names of subgroup of  G
    std::vector<std::string> get_all_subgroup_names() const;

    /// Returns all names of dataset of G
    std::vector<std::string> get_all_dataset_names() const;

    /// Returns all names of dataset of G
    std::vector<std::string> get_all_subgroup_dataset_names() const;
  };

} // namespace h5
