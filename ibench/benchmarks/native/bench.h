#include "assert.h"
#include "stdlib.h"
#include "time.h"

#ifdef USE_MKL
#include "mkl.h"
#else
#include "lapacke.h"

static void* mkl_malloc(int size, int align) {
   return malloc(size);
}

static void mkl_free(void*p) {
   free(p);
}
#endif


#define SEED 77777

class Bench {

public:
#ifdef USE_MKL
  Bench() {
    int err = vslNewStream(&stream, VSL_BRNG_MT19937, SEED);
    assert(err == VSL_STATUS_OK);
  }
  ~Bench() {
    int err = vslDeleteStream(&stream);
    assert(err == VSL_STATUS_OK);
  }
#endif

  double* make_random_mat(int size) {
    double* mat = make_mat(size);
#ifdef USE_MKL
    int err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, size, mat, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
#endif
    return mat;
  }

  double* make_mat(int mat_size) {
    double *mat = (double *) mkl_malloc(mat_size * sizeof(double), 64);
    assert(mat);
    return mat;
  }

  void run() {
      this->compute();
  }
      
  static double const d_zero = 0.0, d_one = 1.0;
#if defined USE_MKL
  VSLStreamStatePtr stream;
#endif

  virtual void compute()=0;
};
