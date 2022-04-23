#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  float j_index[N] = {0,1,2,3,4,5,6,7};
  __m256 jvec = _mm256_load_ps(j_index);
  for(int i=0; i<N; i++) {
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(jvec, ivec, _CMP_EQ_OQ);
    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);
    __m256 invrvec = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));
    __m256 zerovec = _mm256_set1_ps(0.0);
    invrvec = _mm256_blendv_ps(invrvec, zerovec, mask);
    __m256 invr3vec = _mm256_mul_ps(invrvec,_mm256_mul_ps(invrvec,invrvec));
    __m256 fxvec = _mm256_mul_ps(_mm256_mul_ps(rxvec, mvec),invr3vec);
    __m256 fx2vec = _mm256_permute2f128_ps(fxvec,fxvec,1);
    fx2vec = _mm256_add_ps(fx2vec,fxvec);
    fx2vec = _mm256_hadd_ps(fx2vec,fx2vec);
    fx2vec = _mm256_hadd_ps(fx2vec,fx2vec);
    __m256 fyvec = _mm256_mul_ps(_mm256_mul_ps(ryvec, mvec),invr3vec);
    __m256 fy2vec = _mm256_permute2f128_ps(fyvec,fyvec,1);
    fy2vec = _mm256_add_ps(fy2vec,fyvec);
    fy2vec = _mm256_hadd_ps(fy2vec,fy2vec);
    fy2vec = _mm256_hadd_ps(fy2vec,fy2vec);
    __m256 minusvec = _mm256_set1_ps(-1.0);
    fx2vec = _mm256_mul_ps(fx2vec,minusvec);
    fy2vec = _mm256_mul_ps(fy2vec,minusvec);
    _mm256_store_ps(fx, fx2vec);
    _mm256_store_ps(fy, fy2vec);
    /*
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    */
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
