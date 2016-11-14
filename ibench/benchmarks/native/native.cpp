#include <string>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include "mkl_lapacke.h"
#include "mkl.h"
#include "mkl_vsl.h"
#include "assert.h"
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "mkl_dfti.h"

#include <iostream>
using namespace std;

#define SEED 77777

#define min(a, b) (((a)>(b)) ? (b) : (a))

enum {BILLION=1000000000L};

int trials;

static void _print_mkl_version() {
  int len = 198;
  char buf[198];

  mkl_get_version_string(buf, len);
  cerr << "MKL Version: " << buf << endl;
}

/* Auxiliary routine: printing a matrix */
static void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda ) {
	MKL_INT i, j;
	printf( "\n %s\n", desc );
	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " %3.8f", a[i*lda+j] );
		printf( "\n" );
	}
}

static void timer_start(struct timespec* start) {
   clock_gettime(CLOCK_MONOTONIC, start);
}

static double timer_end(struct timespec* start, struct timespec* end) {
  double diff;
  clock_gettime(CLOCK_MONOTONIC, end);
  diff = BILLION * (end->tv_sec - start->tv_sec) + end->tv_nsec - start->tv_nsec;
  return diff/1.0e9;
}

class Bench {

public:
  Bench() {
    int err = vslNewStream(&stream, VSL_BRNG_MT19937, SEED);
    assert(err == VSL_STATUS_OK);
  }
  ~Bench() {
    int err = vslDeleteStream(&stream);
    assert(err == VSL_STATUS_OK);
  }

  double* make_random_mat(int size) {
    double* mat = make_mat(size);
    int err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, size, mat, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
    return mat;
  }

  double* make_mat(int mat_size) {
    double *mat = (double *) mkl_malloc(mat_size * sizeof(double), 64);
    assert(mat);
    return mat;
  }

  void run() {
    double* elapsed = new double(trials);

    // warmup
    this->compute();

    for (int i = 0; i < trials; i++) {
      struct timespec start, end;
      timer_start(&start);
      this->compute();
      elapsed[i] = timer_end(&start, &end);
    }

    cout << "[ " << elapsed[0];
    for (int i = 1; i < trials; i++) {
      cout << ", " << elapsed[i];
    }
    cout << "]" << endl;
    delete [] elapsed;
  }
      
  double d_zero = 0.0, d_one = 1.0;
  VSLStreamStatePtr stream;

  virtual void compute()=0;
};

class lu : public Bench {
public:
  lu(int n);
  ~lu();
  void compute();
  int *ipiv;
  double *x_mat, *r_mat, *l_mat, *u_mat, *p_mat;
  int N,M,LDA;
  int ld_a;
};


lu::lu(int size) {
  N = size;
  M = size;
  LDA = size;
  ld_a = LDA;
  int mat_size = M*N, mn_min = min(M, N);
  int l_size = M*mn_min, 
      r_size = mat_size, 
      u_size = mn_min*N,
      p_size = M*M;

  cerr << "Solving lu problem for random (" << M << ", " << N << ") matrix" << endl;

  /* input matrix */
  x_mat = make_random_mat(mat_size);

  /* list of pivots */
  ipiv = (int *) mkl_malloc(mn_min * sizeof(int), 64);
  assert(ipiv);

  /* matrix for result */
  r_mat = make_random_mat(r_size);

  /* lower-triangular matrix */
  l_mat = make_random_mat(l_size);

  /* upper triangular matrix */
  u_mat = make_random_mat(u_size);

  /* permutation matrix */
  p_mat = make_random_mat(p_size);

  mkl_domatcopy('R', 'T', M, N, 1.0, x_mat, N, r_mat, M);
  ld_a = M + N - ld_a;
}

void lu::compute() {
  int mat_size = M*N, mn_min = min(M, N);
  int l_size = M*mn_min, 
      r_size = mat_size, 
      u_size = mn_min*N,
      p_size = M*M;

  /* compute pivoted lu decomposition */
  int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, r_mat, ld_a, ipiv);
  assert(info == 0);

  int ld_l = M;
  int ld_u = mn_min;
  int ld_p = M;
  memset(l_mat, 0, l_size * sizeof(double));
  memset(u_mat, 0, u_size * sizeof(double));

  /* extract L and U matrix elements from r_mat */
  #pragma ivdep
  for(int i = 0; i < M; i++) 
  {
  #pragma ivdep
    for(int j = 0; j < N; j++) 
    {
      if (j < mn_min) 
      {
	if(i == j) 
        {
	  l_mat[j * ld_l + i] = 1.0;
	} 
	else if (i > j) 
        {
	  l_mat[j * ld_l + i] = r_mat[j * ld_a + i];
	} 
      }
      if (i < mn_min) {
	if(i <= j) 
	  u_mat[j * ld_u + i] = r_mat[j * ld_a + i];
      }
    }
  }

  /* make a diagonal matrix (m,m) */
  memset(p_mat, 0, p_size * sizeof(double)); 
  for(int i = 0; i < M; i++) p_mat[i*(M + 1)] = 1.0;    

  info = LAPACKE_dlaswp(LAPACK_COL_MAJOR, M, p_mat, M, 1, mn_min, ipiv, -1);
  assert(info == 0);
}

lu::~lu() {
  mkl_free(l_mat);
  mkl_free(u_mat);
  mkl_free(r_mat);
  mkl_free(p_mat);

  mkl_free(ipiv);
  mkl_free(x_mat);
}

class inv : public Bench {
public:
  inv(int n);
  ~inv();
  void compute();
  double *x_mat, *r_mat;
  int *ipiv;
  int N,M,LDA;
};

inv::inv(int size) {
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

void inv::compute() {
  /* compute pivoted lu decomposition */
  int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, M, N, r_mat, LDA, ipiv);
  assert(info == 0);

  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, r_mat, LDA, ipiv);
  assert(info == 0);

}

inv::~inv() {
  mkl_free(r_mat);
  mkl_free(ipiv);
  mkl_free(x_mat);
}

class det : public Bench {
public:
  det(int n);
  ~det();
  void compute();
  double *x_mat, *r_mat;
  int *ipiv;
  int N,M,LDA;
  int mn_min;
  double result;
};

det::det(int size) {
  N = size;
  M = size;
  LDA = size;
  int mat_size = M*N;
  mn_min = min(M, N);
  assert(M == N);

  cerr <<  "Computing determinant for random (" << M << ", " << N << ") matrix using native code" << endl << endl;

  /* input matrix */
  x_mat = make_random_mat(mat_size);

  /* list of pivots */
  ipiv = (int *) mkl_malloc(mn_min * sizeof(int), 64);
  assert(ipiv);

  /* matrix for result */
  r_mat = make_random_mat(mat_size);
}

void det::compute() {
  /* compute pivoted lu decomposition */
  int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, M, N, r_mat, LDA, ipiv);
  assert(info == 0);

  double t = 1.0;
  int i,j;
  for(i=0, j=0; i < mn_min; i++, j+= LDA+1) {
    t *= (ipiv[i]==i) ? r_mat[j] : -r_mat[j];
  }
  result = t;
}

det::~det() {
  mkl_free(r_mat);
  mkl_free(ipiv);
  mkl_free(x_mat);
}

class dot : public Bench {
public:
  dot(int n);
  ~dot();
  void compute();
  double *a_mat, *b_mat, *r_mat;
  int N,M,K;
};

dot::dot(int size) {
  N = 20000;
  M = N;
  K = size;

  assert(M == N);

  cerr <<  "Computing matrix dot product for random (M: " << M << ", K: " << K << ", N: " << N << ") matrix using native code" << endl << endl;

  /* input matrix */
  a_mat = make_random_mat(M*K);
  b_mat = make_random_mat(K*N);

  /* matrix for result */
  r_mat = make_random_mat(M*N);
}

void dot::compute() {
  double alpha = 1.0; 
  double beta = 0.0;

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, alpha, a_mat, K, b_mat, N, beta, r_mat, N);
}

dot::~dot() {
  mkl_free(a_mat);
  mkl_free(b_mat);
  mkl_free(r_mat);
}

class qr : public Bench {
public:
  qr(int n);
  ~qr();
  void compute();
  int *ipiv;
  double *x_mat, *r_mat, *l_mat, *u_mat, *p_mat;
  int N,M,LDA;
  int ld_a;
};

qr::qr(int size) {
  N = size;
  M = size;
  LDA = size;
  ld_a = LDA;
  int mat_size = M*N, mn_min = min(M, N);
  int l_size = M*mn_min, 
      r_size = mat_size, 
      u_size = mn_min*N,
      p_size = M*M;

  cerr << "Solving lu problem for random (" << M << ", " << N << ") matrix" << endl;

  /* input matrix */
  x_mat = make_random_mat(mat_size);

  /* list of pivots */
  ipiv = (int *) mkl_malloc(mn_min * sizeof(int), 64);
  assert(ipiv);

  /* matrix for result */
  r_mat = make_random_mat(r_size);

  /* lower-triangular matrix */
  l_mat = make_random_mat(l_size);

  /* upper triangular matrix */
  u_mat = make_random_mat(u_size);

  /* permutation matrix */
  p_mat = make_random_mat(p_size);

  mkl_domatcopy('R', 'T', M, N, 1.0, x_mat, N, r_mat, M);
  ld_a = M + N - ld_a;
}

void qr::compute() {
  int mat_size = M*N, mn_min = min(M, N);
  int l_size = M*mn_min, 
      r_size = mat_size, 
      u_size = mn_min*N,
      p_size = M*M;

  /* compute pivoted lu decomposition */
  int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, r_mat, ld_a, ipiv);
  assert(info == 0);

  int ld_l = M;
  int ld_u = mn_min;
  int ld_p = M;
  memset(l_mat, 0, l_size * sizeof(double));
  memset(u_mat, 0, u_size * sizeof(double));

  /* extract L and U matrix elements from r_mat */
  #pragma ivdep
  for(int i = 0; i < M; i++) 
  {
  #pragma ivdep
    for(int j = 0; j < N; j++) 
    {
      if (j < mn_min) 
      {
	if(i == j) 
        {
	  l_mat[j * ld_l + i] = 1.0;
	} 
	else if (i > j) 
        {
	  l_mat[j * ld_l + i] = r_mat[j * ld_a + i];
	} 
      }
      if (i < mn_min) {
	if(i <= j) 
	  u_mat[j * ld_u + i] = r_mat[j * ld_a + i];
      }
    }
  }

  /* make a diagonal matrix (m,m) */
  memset(p_mat, 0, p_size * sizeof(double)); 
  for(int i = 0; i < M; i++) p_mat[i*(M + 1)] = 1.0;    

  info = LAPACKE_dlaswp(LAPACK_COL_MAJOR, M, p_mat, M, 1, mn_min, ipiv, -1);
  assert(info == 0);
}

qr::~qr() {
  mkl_free(l_mat);
  mkl_free(u_mat);
  mkl_free(r_mat);
  mkl_free(p_mat);

  mkl_free(ipiv);
  mkl_free(x_mat);
}

class cholesky : public Bench {
public:
  cholesky(int n);
  ~cholesky();
  void compute();
  double *a_mat;
  int N,LDA;
};

cholesky::cholesky(int size) {
  N = size;
  LDA = size;
  int mat_size = N*N;

  cerr <<  "Computing chokesky decomposition for random (" << N << ", " << N << ") matrix using native code" << endl << endl;

  double* r_mat = make_random_mat(mat_size);

  // Input Matrix
  a_mat = make_mat(mat_size);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a_mat[i*N+j] = r_mat[i*N+j] * r_mat[j*N+i];
      if (i == j) a_mat[i*N+j] += N;
    }
  }

  mkl_free(r_mat);
}

void cholesky::compute() {
  int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', N, a_mat, LDA);
  assert(info == 0);
}

cholesky::~cholesky() {
  mkl_free(a_mat);
}

class fft : public Bench {
public:
  fft(int n);
  ~fft();
  void compute();
  int N;
  double* x_mat;
  DFTI_DESCRIPTOR_HANDLE ffth;
};

fft::fft(int size) {
  N = size;
  int mat_size = N;

  cerr <<  "Computing fft for random (" << N << ") matrix using native code" << endl << endl;
  x_mat = make_mat(mat_size);

  // initialize x
  DftiCreateDescriptor ( &ffth, DFTI_DOUBLE, DFTI_REAL, 1, N );
  DftiCommitDescriptor ( ffth );
}

void fft::compute() {
  // Sync this variable with run.py
  int runs = 1000;

  for (int i = 0; i < runs; i++)
    DftiComputeForward ( ffth, x_mat );
}

fft::~fft() {
  DftiFreeDescriptor ( &ffth );
}



/* Main program */
int main(int argc, char *argv[]) {
  int n;
  string name;
  _print_mkl_version();

  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " bench size" << endl;
    return 1;
  }
  istringstream(argv[1]) >> name;
  istringstream(argv[2]) >> n;
  istringstream(argv[3]) >> trials;
  cerr << "Running: " << name << " size: " << n << " trials: " << trials << endl;
  class Bench* b;

  if (name == "lu") {
    b = new lu(n);
  } else if (name == "dot") {
    b = new dot(n);
  } else if (name == "inv") {
    b = new inv(n);
  } else if (name == "cholesky") {
    b = new cholesky(n);
  } else if (name == "fft") {
    b = new fft(n);
  } else if (name == "qr") {
    b = new qr(n);
  } else if (name == "det") {
    b = new det(n);
  }
  b->run();
  delete b;

  return 0;
}

