#include <algorithm>
#include <iostream>
#include <cstring>

using namespace std;

#include "inv.h"

C_inv::C_inv(int size) {
  N = size;
  M = size;
  LDA = size;
  int mat_size = M*N, mn_min = min(M, N);

  assert(M == N);

  cerr <<  "Computing matrix inverse for random (" << M << ", " << N << ") matrix using native code" << endl << endl;

  /* input matrix */
  x_mat = make_random_mat(mat_size);

  /* list of pivots */
  ipiv = (int *) mkl_malloc(mn_min * sizeof(int), 64);
  assert(ipiv);

  /* matrix for result */
  r_mat = make_random_mat(mat_size);
}

void C_inv::compute() {
  /* compute pivoted lu decomposition */
  int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, M, N, r_mat, LDA, ipiv);
  assert(info == 0);

  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, r_mat, LDA, ipiv);
  assert(info == 0);

}

C_inv::~C_inv() {
  mkl_free(r_mat);
  mkl_free(ipiv);
  mkl_free(x_mat);
}
