# distutils: language = c++
# distutils: sources = ibench/benchmarks/native/inv.cpp

import numpy as np
import scipy

import ibench
from ibench.benchmarks.bench import Bench

class Inv(Bench):

    def _ops(self, n):
        # scipy is getrf getri
        return 2.*n*n*n*1e-9
        # numpy calls gesv
        # lu + triangular solve
        # TRF + TRS
        # 2/3 n^3 + 2 n^3 = 8/3 n^3
        # return 8./3.*N*N*N*1e-9

    def _make_args(self, n):
        self._A = np.asfortranarray(np.random.rand(n,n), dtype=self._dtype)

    def _compute(self):
        scipy.linalg.inv(self._A, overwrite_a=True, check_finite=False)

# Expose the C++ class
cdef extern from 'native/inv.h':
     cdef cppclass C_inv:
        C_inv(int) except +
        c_compute() except +

# Wrap the c++ class in an extension class
cdef class Foo:
    cdef C_inv* c_inv
    def __cinit__(self, int n):
        self.c_inv = new C_inv(n)
    def __dealloc__(self):
        del self.c_inv

# The extension class cannot inherit from python class, so we need 1 more layer
class Native_inv(Inv):
    def _make_args(self, n):
        pass

    def _compute(self):
        self.c_compute()
