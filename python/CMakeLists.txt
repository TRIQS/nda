add_subdirectory(pybind11)

# FIX THIS : EXTRACT JUST NUMPY FROM THE MODULE We also need the python_and_numpy target
find_package(Python) 

add_subdirectory(h5)
add_subdirectory(nda)

