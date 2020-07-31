# Copyright (c) 2020 Simons Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generated automatically using the command :
# c++2py wrap_basic.hpp -C nda_py -m wrap_basic -o wrap_basic --cxxflags="-std=c++17" --includes=./../../c++ --includes=./../../python --target_file_only
from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "wrap_basic", doc = r"", app_name = "wrap_basic")

# Imports

# Add here all includes
module.add_include("wrap_basic.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <nda_py/cpp2py_converters.hpp>

""")


# The class member_access
c = class_(
        py_type = "MemberAccess",  # name of the python class
        c_type = "member_access",   # name of the C++ class
        doc = r"""""",   # doc of the C++ class
        hdf5 = False,
)

c.add_member(c_name = "arr",
             c_type = "nda::array<long, 1>",
             read_only= False,
             doc = r"""""")

c.add_member(c_name = "arr_arr",
             c_type = "nda::array<nda::array<long, 1>, 1>",
             read_only= False,
             doc = r"""""")

module.add_class(c)

module.add_function ("nda::array<long, 1> make_arr (long n)", doc = r"""""")

module.add_function ("nda::array<long, 2> make_arr (long n1, long n2)", doc = r"""""")

module.add_function ("nda::array<nda::array<long, 1>, 1> make_arr_arr (long n1, long n2)", doc = r"""""")

module.add_function ("long size_arr (nda::array<long, 1> a)", doc = r"""""")

module.add_function ("long size_arr (nda::array<long, 2> a)", doc = r"""""")

module.add_function ("long size_arr_v (nda::array_view<long, 1> a)", doc = r"""""")

module.add_function ("long size_arr_v (nda::array_view<long, 2> a)", doc = r"""""")

module.add_function ("long size_arr_cv (nda::array_view<long, 1> a)", doc = r"""""")

module.add_function ("long size_arr_cv (nda::array_view<long, 2> a)", doc = r"""""")

module.add_function ("long size_arr_arr (nda::array<nda::array<long, 1>, 1> a)", doc = r"""""")

module.add_function ("long size_arr_arr_v (nda::array<nda::array_view<long, 1>, 1> a)", doc = r"""""")

module.add_function ("long size_arr_arr_cv (nda::array<nda::array_view<long, 1>, 1> a)", doc = r"""""")


module.add_function ("nda::array<long, 1> multby2 (nda::array<long, 1> a)", doc = r"""""")
module.add_function ("nda::array<long, 2> multby2 (nda::array<long, 2> a)", doc = r"""""")
module.add_function ("nda::array<double, 1> multby2_d (nda::array<double, 1> a)", doc = r"""""")


module.generate_code()
