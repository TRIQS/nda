################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2020, The Simons Foundation
#   author: N. Wentzell
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

r"""
DOC

"""

class Cpp2pyInfo:

    table_imports = {
    }

    table_converters = {
      'nda::basic_array' : 'nda_py/cpp2py_converters.hpp',
      'nda::basic_array_view' : 'nda_py/cpp2py_converters.hpp',
    }

__all__ = ['Cpp2pyInfo']
