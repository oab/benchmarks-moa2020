#include <assert.h>
#include <time.h>
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
static const int s1padded = SIZE+PAD;
static const int s2padded = SIZE+PAD;
static const int array_size = s0*s1*s2;
static const int array_size_padded = s0*s1*s2padded;

static const int steps = 50;
//the step procedure uses 3 arrays

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
static inline int gamma1(int i, int j, int k)
{
  return  i*s1*s2 + j*s2 + k;
}

static inline int gamma2(int i, int j, int k)
{
  return  i*s1*s2padded + j*s2padded + k;
}

static inline int gamma3(int i, int j, int k)
{
  return  i*s1padded*s2 + j*s2 + k;
}



static inline int mod(int x, int m) {
  int r = x%m;
  return r<0 ? r+m : r;
}

static void SC_NO_PADDING(double *u,
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
        u[gamma1(i,j,k)] =
        u[gamma1(i,j,k)] + c4 * (c3 * (c1 *
        v[gamma1((mod(i-1,s0)),j,k)] +
        v[gamma1((mod(i+1,s0)),j,k)] +
        v[gamma1(i,(mod(j-1,s1)),k)] +
        v[gamma1(i,(mod(j+1,s1)),k)] +
        v[gamma1(i,j,(mod(k-1,s2)))] +
        v[gamma1(i,j,(mod(k+1,s2)))]) -
        3 * c2 * u[gamma1(i,j,k)] - c0 *
        ((v[gamma1((mod(i+1,s0)),j,k)] -
        v[gamma1((mod(i-1,s0)),j,k)]) *
        u0[gamma1(i,j,k)] +
        (v[gamma1(i,(mod(j+1,s1)),k)] -
        v[gamma1(i,(mod(j-1,s1)),k)]) *
        u1[gamma1(i,j,k)] +
        (v[gamma1(i,j,(mod(k+1,s2)))] -
        v[gamma1(i,j,(mod(k-1,s2)))]) *
        u2[gamma1(i,j,k)]));
      }
    }
  }
}

static void SC_PADDING_K (double *u,
                        const double *restrict v,
                        const double *restrict u0,
                        const double *restrict u1,
                        const double *restrict u2,
                        const double c0, const double c1, const double c2,
                        const double c3, const double c4)
{  
  for (int i=0; i<s0; i++) {
    for (int j=0; j<s1; j++) {
      for (int k=1; k<s2+1; k++) {
        u[gamma2(i,j,k)] =
        u[gamma2(i,j,k)] + c4 * (c3 * (c1 *
        v[gamma2((mod(i-1,s0)),j,k)] +
        v[gamma2((mod(i+1,s0)),j,k)] +
        v[gamma2(i,(mod(j-1,s1)),k)] +
        v[gamma2(i,(mod(j+1,s1)),k)] +
        v[gamma2(i,j,k-1)] +
        v[gamma2(i,j,k+1)]) -
        3 * c2 * u[gamma2(i,j,k)] - c0 *
        ((v[gamma2((mod(i+1,s0)),j,k)] -
        v[gamma2((mod(i-1,s0)),j,k)]) *
        u0[gamma2(i,j,k)] +
        (v[gamma2(i,(mod(j+1,s1)),k)] -
        v[gamma2(i,(mod(j-1,s1)),k)]) *
        u1[gamma2(i,j,k)] +
        (v[gamma2(i,j,k+1)] -
        v[gamma2(i,j,k-1)]) *
        u2[gamma2(i,j,k)]));
      }
      // update the padding values
      u[gamma2(i,j,0)] = u[gamma2(i,j,s2)];
      u[gamma2(i,j,s2+1)] = u[gamma2(i,j,0)];
    }
  }
}

static void SC_PADDING_J (double *u,
                        const double *restrict v,
                        const double *restrict u0,
                        const double *restrict u1,
                        const double *restrict u2,
                        const double c0, const double c1, const double c2,
                        const double c3, const double c4)
{  
  for (int i=0; i<s0; i++) {
    for (int j=1; j<s1+1; j++) {
      for (int k=0; k<s2; k++) {
        u[gamma3(i,j,k)] =
        u[gamma3(i,j,k)] + c4 * (c3 * (c1 *
        v[gamma3((mod(i-1,s0)),j,k)] +
        v[gamma3((mod(i+1,s0)),j,k)] +
        v[gamma3(i,j-1,k)] +
        v[gamma3(i,j+1,k)] +
        v[gamma3(i,j,k-1)] +
        v[gamma3(i,j,k+1)]) -
        3 * c2 * u[gamma3(i,j,k)] - c0 *
        ((v[gamma3((mod(i+1,s0)),j,k)] -
        v[gamma3((mod(i-1,s0)),j,k)]) *
        u0[gamma3(i,j,k)] +
        (v[gamma3(i,j+1,k)] -
        v[gamma3(i,j-1,k)]) *
        u1[gamma3(i,j,k)] +
        (v[gamma3(i,j,k+1)] -
        v[gamma3(i,j,k-1)]) *
         u2[gamma3(i,j,k)]));
      }
    }
    // update the padding values
    memcpy(u+gamma3(i,0,0),u+gamma3(i,s1,0),sizeof(double)*s2);
    memcpy(u+gamma3(i,s1+1,0),u+gamma3(i,0,0),sizeof(double)*s2);

  }
}




typedef void (*kernel)(double*,
                       const double*,
                       const double*,
                       const double*,
                       const double*,
                       const double, const double, const double,
                       const double, const double);
static void step(const int size, double *u,
                 double *v,
                 double nu, double dx, double dt, kernel snippet)
{
  double c0 = 0.5/dx;
  double c1 = 1/dx/dx;
  double c2 = 2/dx/dx;
  double c3 = nu;
  double c4 = dt/2;
 
  memcpy(v,u,size*3*sizeof(double));
  double *u0 = &u[0];
  double *u1 = &u[size];
  double *u2 = &u[2*size];

  double *v0 = &v[0];
  double *v1 = &v[size];
  double *v2 = &v[2*size];

  snippet(v0,u0,u0,u1,u2,c0,c1,c2,c3,c4);
  snippet(v1,u1,u0,u1,u2,c0,c1,c2,c3,c4);
  snippet(v2,u2,u0,u1,u2,c0,c1,c2,c3,c4);
  snippet(u0,v0,v0,v1,v2,c0,c1,c2,c3,c4);
  snippet(u1,v1,v0,v1,v2,c0,c1,c2,c3,c4);
  snippet(u2,v2,v0,v1,v2,c0,c1,c2,c3,c4);
}


void nopad()
{
  const int total = array_size*3;
  //original data kept here
  double *start = malloc(total * sizeof(double));
  double *u = malloc(total * sizeof(double));
  double *v = malloc(total * sizeof(double));

  dumpsine(total,start);

  clock_t begin;
  clock_t end;
  double tspent;
  
  memcpy(u,start,total*sizeof(double));
  printf("Singlecore-no-padding:");
  begin = clock();
  for(int i=0;i<steps;i++)
    step(array_size,u,v,s_nu,s_dx,s_dt,SC_NO_PADDING);
  end = clock();
  tspent = ((double)(end - begin))/CLOCKS_PER_SEC;
  printf("%f\n",tspent);
  for (int i =0;i<total;i++) {
    if (isnan(u[i])) {
      printf("NAN detected\n");
    }
  }
  
  free(u);
  free(v);
  free(start);

}
void padk() {
  const int total =  array_size_padded*3;
  
  //original data kept here
  double *start = malloc(total * sizeof(double));
  dumpsine(total,start);

  clock_t begin;
  clock_t end;
  double tspent;

  //padded version of u, padded along all dimensions
  double *upadded = malloc(total * sizeof(double));
  double *vpadded = malloc(total * sizeof(double));

  //prepare the padded array, this can be done in a more clever
  //manner but that is another topic and not important for benchmarking
  //should be done with memcpy and related functions
  for (int part = 0;part<3;part++) {
    int offsetpadded = part*array_size_padded;
    int offset = part*array_size;
    for (int i = 0; i<s0; i++) {
      for (int j = 0; j<s1; j++) {
        // first pad value
        upadded[offsetpadded+gamma2(i,j,0)]=start[offset+gamma1(i,j,s2-1)];
        // last pad value
        upadded[offsetpadded+gamma2(i,j,s2padded-1)]=start[offset+gamma1(i,j,0)];

        memcpy(upadded+offsetpadded+gamma2(i,j,1),start+offset+gamma1(i,j,s2-1),s2*sizeof(double));  
      }
    }
  }
  
  printf("Singlecore-padding-third-axis:");
  begin = clock();
  for(int i=0;i<steps;i++)
    step(array_size_padded,upadded,vpadded,s_nu,s_dx,s_dt,SC_PADDING_K);
  end = clock();
  tspent = ((double)(end - begin))/CLOCKS_PER_SEC;
  printf("%f\n",tspent);
  for (int i =0;i<total;i++) {
    if (isnan(upadded[i])) {
      printf("NAN detected\n");
    }
  }
  // Now there would need to be a reversal of the padding if the expected
  // output is the data as an array of the kind start is, but that is omitted
  // and we consider the computation complete
  
  free(start);
  free(upadded);
  free(vpadded);

}

void padj()
{
  const int total =  array_size_padded*3;

  //original data kept here
  double *start = malloc(total * sizeof(double));
  dumpsine(total,start);

  clock_t begin;
  clock_t end;
  double tspent;

  //padded version of u, padded along all dimensions
  double *upadded = malloc(total * sizeof(double));
  double *vpadded = malloc(total * sizeof(double));

  // we must do these replications
  // The padded values
  // gamma2(i,0,k)  == gamma(i,s1-1,k)
  // gamma2(i,s1,k) == gamma(i,0,k)
  // The regular values
  // gamma2(i,j,k) == gamma(i,j,k)
  
  for (int part = 0;part<3;part++) {
    int offsetpadded = part*array_size_padded;
    int offset = part*array_size;
    for(int i = 0; i<s0; i++) {
      for(int j = -1; j<s1+1; j++) {
        for(int k = 0; k<s2; k++) {
          int jj  = (j==-1)? s1-1 : (j==s1)? 0 : j; 
          upadded[offsetpadded+gamma3(i,j+1,k)] = start[offset+gamma1(i,jj,k)];
        }
      }
    }
  }
  
  printf("Singlecore-with-padding-second-axis:");
  begin = clock();
  for(int i=0;i<steps;i++)
    step(array_size_padded,upadded,vpadded,s_nu,s_dx,s_dt,SC_PADDING_J);
  end = clock();
  tspent = ((double)(end - begin))/CLOCKS_PER_SEC;
  printf("%f\n",tspent);
  for (int i =0;i<total;i++) {
    if (isnan(upadded[i])) {
      printf("NAN detected\n");
    }
  }
  // Now there would need to be a reversal of the padding if the expected
  // output is the data as an array of the kind start is, but that is omitted
  // and we consider the computation complete
  
  free(start);
  free(upadded);
  free(vpadded);
}

int main() {
  nopad();
  padk();
  padj();
  return EXIT_SUCCESS;
}

