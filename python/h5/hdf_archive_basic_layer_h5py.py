
################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2011 by M. Ferrero, O. Parcollet
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

import numpy,string
from hdf_archive import *
import _h5py as h5
 
class HDFArchiveGroupBasicLayer :
    _class_version = 1

    def __init__(self, parent, subpath ):
        """  """
        self.options = parent.options
        self._group = parent._group.open_group(subpath) if subpath else parent._group
        self.ignored_keys = [] 
        self.cached_keys = self._group.keys()

    def _init_root(self, LocalFileName, open_flag) :
        try :
            fich = h5.File(LocalFileName, open_flag)
        except :
            print "Cannot open the HDF file %s"%LocalFileName
            raise
        # checking the version
        if open_flag not in ['r','r+','a'] :
            self._version = self._class_version
        else :
            try :
                self._version = int(fich.attrs['HDFArchive_Version'])
            except :
                self._version = 1
            if self._version > self._class_version :
                raise IOError, "File %s is too recent for this version of HDFArchive module"%Filename
        self._group =  h5.Group(fich)

    def is_group(self,p) :
        """Is p a subgroup ?"""
        assert len(p)>0 and p[0]!='/'
        return p in self.cached_keys and self._group.has_subgroup(p)

    def is_data(self,p) :
        """Is p a leaf ?"""
        assert len(p)>0 and p[0]!='/'
        return p in self.cached_keys and self._group.has_dataset(p)

    def write_attr (self, key, val) :
        self._group.write_attribute(key, val)

    def _read (self, key):
        return h5.h5_read(self._group, key)

    def _write(self, key, val) :
        h5.h5_write(self._group, key, val)

    def _flush(self):
        if bool(self._group): self._group.file.flush()

    def _close(self):
        if bool(self._group): self._group.file.close()

    def create_group (self,key):
        self._group.create_group(key)
        self.cached_keys.append(key)

    def keys(self) :
        return self.cached_keys

    def _clean_key(self,key, report_error=False) :
        if report_error and key not in self.cached_keys :
             raise KeyError, "Key %s is not in archive !!"%key
        if key in self.cached_keys :
          # FIXME
          #del self._group[key]
          self.cached_keys.remove(key)
        else: raise KeyError, "Key %s is not in archive !!"%key

