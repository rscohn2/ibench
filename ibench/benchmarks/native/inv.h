#include "bench.h"

class C_inv : public Bench {
 public:
  C_inv(int n);
  ~C_inv();
  void compute();
 private:
  double *x_mat, *r_mat;
  int *ipiv;
  int N,M,LDA;
};
