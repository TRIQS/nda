# We need the python_and_numpy target
find_package(Python) 

# Build the pybind11 module
pybind11_add_module(_h5py _h5py.cpp h5py_io.cpp)
target_link_libraries(_h5py PRIVATE h5_interface hdf5 python_and_numpy)
