#!/usr/bin/env python


import unittest
import os
import sys

from copy_move_stat import CopyMoveStat, MemberStat, make_obj, make_arr, take_obj, take_arr

class test_copy_move_stat(unittest.TestCase):

    def setUp(self):
        self.stat = CopyMoveStat(False)

    def counts(self):
        res = [self.stat.copy_count(), self.stat.move_count()]
        self.stat.reset()
        return res

    def test_basic(self):

        # Simple constr / delete
        a = CopyMoveStat()
        del a
        assert self.counts() == [0, 0]

        # Factory (c2py)
        a = make_obj()
        assert self.counts() == [0, 1]

        # Function call (c2py)
        take_obj(a)
        assert self.counts() == [1, 0]

    def test_arr1(self):

        # # Factory (c2py)
        b = make_arr(2)
        assert self.counts() == [0, 2]

        # Function call (py2c)
        take_arr(b)
        assert self.counts() == [2, 0]

    def test_arr2(self):

        # Factory (c2py)
        c = make_arr(2, 2)
        assert self.counts() == [0, 2*2]

        # Function call (py2c)
        take_arr(c)
        assert self.counts() == [2*2, 0]

    def test_member(self):

        d = MemberStat()
        
        # Member Access
        d.m
        assert self.counts() == [1, 0]

if __name__ == '__main__':
    unittest.main()
