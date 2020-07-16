#!/usr/bin/env python

import unittest
import os
import sys

from wrap_basic import MemberAccess, make_arr, make_arr_arr, size_arr, size_arr_v, size_arr_cv, size_arr_arr, size_arr_arr_v, size_arr_arr_cv

import numpy as np

class test_wrap_basic(unittest.TestCase):

    def test_basic(self): 

        # ===== C2PY =====

        # Arr1D
        a = make_arr(4)
        assert np.all(a == np.array([0, 1, 2, 3]))

        # Arr2D
        b = make_arr(2, 2)
        assert np.all(b == np.array([[0, 1], [2, 3]]))

        # ArrArr 
        # NOTE: This always remains an np.array(np.array)
        #       and is never converted to np.array of higher rank
        c = make_arr_arr(2, 2)
        assert np.all(c[0] == b[0])
        assert np.all(c[1] == b[1])
        
        # ===== PY2C =====

        # Take Arr1D
        assert a.size == size_arr(a)
        assert a.size == size_arr_v(a)
        assert a.size == size_arr_cv(a)
        self.assertRaises(TypeError, size_arr_v, np.array([[1.0]]))

        # Take Arr2D
        assert b.size == size_arr(b)
        assert b.size == size_arr_v(b)
        assert b.size == size_arr_cv(b)

        # # Take ArrArr
        assert c.size == size_arr_arr(c)
        assert c.size == size_arr_arr_v(c)
        assert c.size == size_arr_arr_cv(c)
        assert 2 == size_arr_arr(np.array([make_arr(2), make_arr(2)]))
        # FIXME Not yet implemented
        # assert 2 == size_arr_arr([make_arr(2), make_arr(4)])


if __name__ == '__main__':
    unittest.main()
