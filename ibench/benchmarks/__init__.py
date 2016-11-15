from ibench.benchmarks.cholesky import Cholesky
from ibench.benchmarks.det import Det
from ibench.benchmarks.dot import Dot
from ibench.benchmarks.fft import FFT
from ibench.benchmarks.inv import Inv,Native_inv
from ibench.benchmarks.lu import LU
from ibench.benchmarks.qr import QR
from ibench.benchmarks.svd import SVD


benchmarks = {
    'cholesky': Cholesky,
    'det': Det,
    'dot': Dot,
    'fft': FFT,
    'inv': Inv,
    'lu': LU,
    'native_inv': Native_inv,
    'qr': QR,
    'svd': SVD
}

benchmark_groups = {
    'linalg': ['cholesky', 'det', 'dot', 'inv', 'lu', 'qr', 'svd'],
    'native_linalg': ['native_inv'],
    'all': list(benchmarks.keys())
}
