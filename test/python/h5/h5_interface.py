import h5._h5py as h5
import numpy as np
import unittest

def assert_arrays_are_close(a, b, precision = 1.e-6):
    d = np.amax(np.abs(a - b))
    assert  d< precision, "Arrays are different. Difference is %s.\n %s \n\n --------- \n\n %s"%(d,a,b)

def assert_array_close_to_scalar(a, x, precision = 1.e-6):
    assert_arrays_are_close(a, np.identity(a.shape[0])*(x), precision)

class test_operators(unittest.TestCase):
    def setUp(self):
        pass

    def test_h5_subgroup(self):

        # Open a file and write a few things into it
        f = h5.File("test2.h5", 'w')
        g = h5.Group(f)
        g2 = g.create_group("GG")

        a = np.array(((1,2,3), (4,5,6)), np.int)
        h5.h5_write(g2, 'a', a)
        h5.h5_write(g, 'a1', a)
        g.write_attribute("ATTR1", "value1")

        del f

        # Read again
        ff = h5.File("test2.h5", 'r')
        g = h5.Group(ff)
        g.open_group("GG")
        assert_arrays_are_close(h5.h5_read(g2, 'a'), a)
        print g.keys()

        self.assertEqual(g.has_subgroup('GG'), True)
        self.assertEqual(g.has_subgroup('GG4'), False)

        self.assertEqual(g.has_dataset('a1'), True)
        self.assertEqual(g.has_dataset('a'), False)

        self.assertEqual(g.read_attribute('ATTR1'), "value1")
        self.assertEqual(g.read_attribute('ATTRNotPresent'), "")

        self.assertEqual(ff.name, "test2.h5")
        self.assertEqual(g.file.name, "test2.h5")

    def test_h5_io(self):

        # Open a file and write a few things into it
        f = h5.File("test.h5", 'w')
        g = h5.Group(f)

        h5.h5_write(g, 'i', 14)
        h5.h5_write(g, 'd', 3.2)
        h5.h5_write(g, 's', "a string")
        h5.h5_write(g, 'c', 1.2 + 3j)

        c = 1
        for types in [np.int, np.float, np.complex]:
            a = np.array(((1,2,3), (4,5,6)), types)
            h5.h5_write(g, 'a%s'%c, a)
            c += 1 

        del f

        # Read again
        f = h5.File("test.h5", 'r')
        g = h5.Group(f)

        self.assertEqual(h5.h5_read(g, 'i'), 14)
        self.assertEqual(h5.h5_read(g, 'd'), 3.2)
        self.assertEqual(h5.h5_read(g, 'c'), 1.2 + 3j)
        c = h5.h5_read(g, 'c')
        self.assertEqual(type(c), type(1j))
        self.assertEqual(h5.h5_read(g, 's'), "a string")

        c = 1
        for types in [np.int, np.float, np.complex]:
            a = np.array(((1,2,3), (4,5,6)), types)
            r = h5.h5_read(g, 'a%s'%c)
            assert_arrays_are_close(r, a)
            c += 1 

if __name__ == '__main__':
    unittest.main()


