
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


static const double s_dt = 0.00082212448155679772495;
static const double s_nu = 1.0;
static const double s_dx = 1.0;
static const int s0 = SIZE;
static const int s1 = SIZE;
static const int s2 = SIZE;
//For dimension lifting on last dimension
static const int s2split = KSPLIT;
//For dimension lifting into tiles
static const int s0tiles=TSIZE;
static const int s1tiles=TSIZE;
static const int s2tiles=TSIZE;
static const int asize = s0*s1*s2;
static const int steps = 50;
//the step procedure uses 3 arrays
static const int total = asize*3;


//test data generation
void dumpsine(int n,double* a) {
  double step = 0.01;
  double PI = 3.14159265358979323846;
  double amplitude = 10.0;
  double phase=0.0125;
  double t=0.0;
  for(int i=0;i<n;i++) {
    a[i]=amplitude*sin(PI*t+phase);
    //printf("%.20lf\n",a[i]);
    t+=step;
  }
}

// row major indexing
static inline int gamma(int i, int j, int k)
{
  return  i*s1*s2 + j*s2 + k;
}

static inline int mod(int x, int m) {
  int r = x%m;
  return r<0 ? r+m : r;
}

static void SC_NO_DL(double *u,
                  const double *restrict v,
                  const double *restrict u0,
                  const double *restrict u1,
                  const double *restrict u2,
                  const double c0, const double c1, const double c2,
                  const double c3, const double c4)
{
  for (int i=0; i<s0; i++) {
    for (int j=0; j<s1; j++) {
      for (int k=0; k<s2; k++) {
        u[gamma(i,j,k)] =
        u[gamma(i,j,k)] + c4 * (c3 * (c1 *
        v[gamma((mod(i-1,s0)),j,k)] +
        v[gamma((mod(i+1,s0)),j,k)] +
        v[gamma(i,(mod(j-1,s1)),k)] +
        v[gamma(i,(mod(j+1,s1)),k)] +
        v[gamma(i,j,(mod(k-1,s2)))] +
        v[gamma(i,j,(mod(k+1,s2)))]) -
        3 * c2 * u[gamma(i,j,k)] - c0 *
        ((v[gamma((mod(i+1,s0)),j,k)] -
        v[gamma((mod(i-1,s0)),j,k)]) *
        u0[gamma(i,j,k)] +
        (v[gamma(i,(mod(j+1,s1)),k)] -
        v[gamma(i,(mod(j-1,s1)),k)]) *
        u1[gamma(i,j,k)] +
        (v[gamma(i,j,(mod(k+1,s2)))] -
        v[gamma(i,j,(mod(k-1,s2)))]) *
         u2[gamma(i,j,k)]));
      }
    }
  }
}

// Introduce a thread dimension; we go from shape <s0,s1,s2> to
// <THREADS,s0/THREADS,s1,s2>. Before we had wrap around effects
// at i==0 and i==s0-1. These will now occur elsewhere.
// If the indices were <i,j,k> and are now <p,i,j,k> we must do the mapping
// f(<p,i,j,k>)= gamma(i*s0/THREADS,j,k). Call this map f gammaDL.
// mod calculations that affect i must propagate into gammaDL
// generating two new functions gammaDLMN, gammaDLNMP

static inline int gammaDL(int p, int i, int j, int k)
{
  return  (p*s0/THREADS+i)*s1*s2 + j*s2 + k;
}

// here MN means mod negative
static inline int gammaDLMN(int p, int i, int j, int k)
{
  return  mod(p*s0/THREADS+i-1,s0)*s1*s2 + j*s2 + k;
}

// here MP means mod positive
static inline int gammaDLMP(int p, int i, int j, int k)
{
  return  mod(p*s0/THREADS+i+1,s0)*s1*s2 + j*s2 + k;
}

static void MC_DL_ON_THREADS(double *u,
                        const double *restrict v,
                        const double *restrict u0,
                        const double *restrict u1,
                        const double *restrict u2,
                        const double c0, const double c1,
                        const double c2, const double c3,
                        const double c4)
{

#pragma omp parallel for schedule(static) num_threads(THREADS)
  for (int p=0;p<THREADS;p++) {
    for (int i=0; i<s0/THREADS; i++) {
      for (int j=0; j<s1; j++) {
        for (int k=0; k<s2; k++) {
          u[gammaDL(p,i,j,k)]=
          u[gammaDL(p,i,j,k)] + c4 * (c3 * (c1 *
          v[gammaDLMN(p,i,j,k)] +
          v[gammaDLMP(p,i,j,k)] +
          v[gammaDL(p,i,mod(j-1,s1),k)] +
          v[gammaDL(p,i,mod(j+1,s1),k)] +
          v[gammaDL(p,i,j,mod(k-1,s2))] +
          v[gammaDL(p,i,j,mod(k+1,s2))]) -
          3 * c2 * u[gammaDL(p,i,j,k)] - c0 *
          ((v[gammaDLMP(p,i,j,k)] -
          v[gammaDLMN(p,i,j,k)]) *
          u0[gammaDL(p,i,j,k)] +
          (v[gammaDL(p,i,mod(j+1,s1),k)] -
          v[gammaDL(p,i,mod(j-1,s1),k)]) *
          u1[gammaDL(p,i,j,k)] +
          (v[gammaDL(p,i,j,mod(k+1,s2))] -
          v[gammaDL(p,i,j,mod(k-1,s2))]) *
          u2[gammaDL(p,i,j,k)]));
        }
      }
    }
  }
}

// We now split the index k in <i,j,k> to get <i,j,c,k>
// We go from shape <s0,s1,s2> to <s0,s1,s2split,s2/s2split>
// And parallelize on the new dimension. The same issue
// arises as before with mod having to propagate in

// i*s1*s2split*s2/s2split + j*s2split*(s2/s2split) + c*(s2/s2split)+ k
// i*s1*s2 + j*s2 + c*(s2/s2split) + k
static inline int gammaDLK(int i, int j, int c, int k)
{
  return  i*s1*s2 + j*s2 + c*(s2/s2split) + k;
}

static inline int gammaDLKMN(int i, int j, int c, int k)
{
  return  i*s1*s2 + j*s2 + mod(c*s2/s2split + k-1,s0);
}

static inline int gammaDLKMP(int i, int j, int c, int k)
{
  return  i*s1*s2 + j*s2 + mod(c*s2/s2split + k+1,s0);
}


static void MC_DL_ON_DIMK(double *u,
                        const double *restrict v,
                        const double *restrict u0,
                        const double *restrict u1,
                        const double *restrict u2,
                        const double c0, const double c1,
                        const double c2, const double c3,
                        const double c4)
{
// assuming cache line of 64 bytes, try to spread by at least that
#pragma omp parallel for collapse(2) schedule(static,8) num_threads(THREADS)
  for (int i=0; i<s0; i++) {
    for (int j=0; j<s1; j++) {
      for (int c=0; c<s2split; c++) {
        for (int k=0; k<s2/s2split; k++) {
          u[gammaDLK(i,j,c,k)] =      
            u[gammaDLK(i,j,c,k)] + c4 * (c3 * (c1 *
            v[gammaDLK(mod(i-1,s0),j,c,k)] +
            v[gammaDLK(mod(i+1,s0),j,c,k)] +
            v[gammaDLK(i,mod(j-1,s1),c,k)] +
            v[gammaDLK(i,mod(j+1,s1),c,k)] +
            v[gammaDLKMN(i,j,c,k)] +
            v[gammaDLKMP(i,j,c,k)]) -
            3 * c2 * u[gammaDLK(i,j,c,k)] - c0 *
            ((v[gammaDLK((mod(i+1,s0)),j,c,k)] -
            v[gammaDLK((mod(i-1,s0)),j,c,k)]) *
             u0[gammaDLK(i,j,c,k)] +
            (v[gammaDLK(i,mod(j+1,s1),c,k)] -
            v[gammaDLK(i,mod(j-1,s1),c,k)]) *
             u1[gammaDLK(i,j,c,k)] +
            (v[gammaDLKMP(i,j,c,k)] -
             v[gammaDLKMN(i,j,c,k)]) *
            u2[gammaDLK(i,j,c,k)]));
        }                
      }
    }
  }
}


// Here we go from the shape <s0,s1,s2> to the shape
// <s0tiles,s0/s0tiles,s1tiles,s1/s0tiles,s2tiles,s0/s2tiles>
// however we do not need to alter gamma

static void MC_DL_TILED(double *u,
                  const double *restrict v,
                  const double *restrict u0,
                  const double *restrict u1,
                  const double *restrict u2,
                  const double c0, const double c1, const double c2,
                  const double c3, const double c4)
{

#pragma omp parallel for collapse(3) schedule(static) num_threads(THREADS)
  for (int ti=0; ti<s0; ti+=s0/s0tiles) {
    for (int tj=0; tj<s1; tj+=s1/s1tiles) {
      for (int tk=0; tk<s2; tk+=s2/s2tiles) {
        for (int i=ti; i<ti+s0/s0tiles; i++) {
          for (int j=tj; j<tj+s1/s1tiles; j++) {
            for (int k=tk; k<tk+s2/s2tiles; k++) {
              u[gamma(i,j,k)] =
                u[gamma(i,j,k)] + c4 * (c3 * (c1 *
                v[gamma((mod(i-1,s0)),j,k)] +
                v[gamma((mod(i+1,s0)),j,k)] +
                v[gamma(i,(mod(j-1,s1)),k)] +
                v[gamma(i,(mod(j+1,s1)),k)] +
                v[gamma(i,j,(mod(k-1,s2)))] +
                v[gamma(i,j,(mod(k+1,s2)))]) -
                3 * c2 * u[gamma(i,j,k)] - c0 *
                ((v[gamma((mod(i+1,s0)),j,k)] -
                v[gamma((mod(i-1,s0)),j,k)]) *
                u0[gamma(i,j,k)] +
                (v[gamma(i,(mod(j+1,s1)),k)] -
                v[gamma(i,(mod(j-1,s1)),k)]) *
                u1[gamma(i,j,k)] +
                (v[gamma(i,j,(mod(k+1,s2)))] -
                v[gamma(i,j,(mod(k-1,s2)))]) *
                u2[gamma(i,j,k)]));
            }
          }
        }
      }
    }
  }
}
typedef void (*kernel)(double*,
                       const double*,
                       const double*,
                       const double*,
                       const double*,
                       const double, const double, const double,
                       const double, const double);
static void step (double *u,
                  double *v,
                  double nu, double dx, double dt, kernel snippet)
{
  double c0 = 0.5/dx;
  double c1 = 1/dx/dx;
  double c2 = 2/dx/dx;
  double c3 = nu;
  double c4 = dt/2;

  memcpy(v,u,3*asize*sizeof(double));
  double *u0 = &u[0];
  double *u1 = &u[asize];
  double *u2 = &u[2*asize];

  double *v0 = &v[0];
  double *v1 = &v[asize];
  double *v2 = &v[2*asize];

  snippet(v0,u0,u0,u1,u2,c0,c1,c2,c3,c4);
  snippet(v1,u1,u0,u1,u2,c0,c1,c2,c3,c4);
  snippet(v2,u2,u0,u1,u2,c0,c1,c2,c3,c4);
  snippet(u0,v0,v0,v1,v2,c0,c1,c2,c3,c4);
  snippet(u1,v1,v0,v1,v2,c0,c1,c2,c3,c4);
  snippet(u2,v2,v0,v1,v2,c0,c1,c2,c3,c4);
}


int main() {
  int status = EXIT_SUCCESS;
  if(s2 % s2split) {
    printf("s2split does not divide s2\n");
    return EXIT_FAILURE;
  }
  if(s0 % s0tiles) {
    printf("s0tiles does not divide s0\n");
    return EXIT_FAILURE;
  }
  if(s1 % s1tiles) {
    printf("s1tiles does not divide s1\n");
    return EXIT_FAILURE;
  }
  if(s2 % s2tiles) {
    printf("s2tiles does not divide s2\n");
    return EXIT_FAILURE;
  }


  //original data kept here
  double *start = malloc(total * sizeof(double));
  double *u = malloc(total * sizeof(double));
  double *v = malloc(total * sizeof(double));
  
  
  dumpsine(total,start);

  double begin;
  double end;
  double tspent;
  
  memcpy(u,start,total*sizeof(double));
  printf("Singlecore-no-dimension-lifting:");
  begin = omp_get_wtime();
  for(int i=0;i<steps;i++)
    step(u,v,s_nu,s_dx,s_dt,SC_NO_DL);
  end = omp_get_wtime();
  tspent = end - begin;
  printf("%lf\n",tspent);
  for (int i =0;i<total;i++) {
    if (isnan(u[i])) {
      printf("NAN detected\n");
      status = EXIT_FAILURE;
      goto cleanup;
    }
  }

  memcpy(u,start,total*sizeof(double));
  printf("Multicore-dimension-lifted-on-threads:");
  begin = omp_get_wtime();
  for(int i=0;i<steps;i++) {
    step(u,v,s_nu,s_dx,s_dt,MC_DL_ON_THREADS);
  }
  end = omp_get_wtime();
  tspent = end - begin;
  printf("%lf\n",tspent);
  for (int i =0;i<total;i++) {
    if (isnan(u[i])) {
      printf("NAN detected\n");
      status = EXIT_FAILURE;
      goto cleanup;

    }
  }
  
  memcpy(u,start,total*sizeof(double));
  printf("Multicore-dimension-lifted-on-second-last-dimension:");
  begin = omp_get_wtime();
  for(int i=0;i<steps;i++)
    step(u,v,s_nu,s_dx,s_dt,MC_DL_ON_DIMK);
  end = omp_get_wtime();
  tspent = end - begin;
  printf("%lf\n",tspent);
  for (int i =0;i<total;i++) {
    if (isnan(u[i])) {
      printf("NAN detected\n");
      status = EXIT_FAILURE;
      goto cleanup;

    }
  }
  
  memcpy(u,start,total*sizeof(double));
  printf("Multicore-dimension-lifted-tiled:");
  begin = omp_get_wtime();
  for(int i=0;i<steps;i++)
    step(u,v,s_nu,s_dx,s_dt,MC_DL_TILED);
  end = omp_get_wtime();
  tspent = end - begin;
  printf("%lf\n",tspent);
  for (int i =0;i<total;i++) {
    if (isnan(u[i])) {
      printf("NAN detected\n");
      status = EXIT_FAILURE;
      goto cleanup;

    }
  }
  
 cleanup:
  free(start);
  free(u);
  free(v);
  
  return status;

}
