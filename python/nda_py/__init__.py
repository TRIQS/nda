# Copyright (c) 2018--present, The Simons Foundation
# This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE in the root of this distribution for details.


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
