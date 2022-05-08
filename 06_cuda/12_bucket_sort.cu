#include <cstdio>
#include <cstdlib>
//#include <vector>

__global__ void bucket_init(int *bucket, int* key, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=n) return;
  // 初期化
  bucket[i] = 0;
}

__global__ void bucket_count(int *bucket, int *key, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  // atomicにバケットでカウント
  atomicAdd(&bucket[key[i]],1);
}

__global__ void bucket_key(int *bucket, int *key, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  // sort済列のiの開始indexを計算
  int offset = 0;
  for(int j=0;j<i;j++){
    offset += bucket[j];
  }
  // 全スレッドでoffsetの計算が終了する前に，bucket[i]--が実行されないように同期
  __syncthreads();
  for(; bucket[i]>0; bucket[i]--){
    key[offset++] = i;
  }
}


int main() {
  const int M = 4;
  int n = 50;
  int range = 5;
  int *key;
  int *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  //for (int i=0; i<range; i++) {
  //  bucket[i] = 0;
  //}
  bucket_init<<<(n+M-1)/M,M>>>(bucket, key, n);
  cudaDeviceSynchronize();

  //for (int i=0;i<n;i++){
  //  printf("i=%d, bucket[i]=%d\n",i,bucket[i]);
  //for (int i=0; i<n; i++) {
  //  bucket[key[i]]++;
  //}
  bucket_count<<<(n+M-1)/M,M>>>(bucket, key, n);
  cudaDeviceSynchronize();

  //for (int i=0, j=0; i<range; i++) {
  //  for (; bucket[i]>0; bucket[i]--) {
  //    key[j++] = i;
  //  }
  //}
  bucket_key<<<(range+M-1)/M,M>>>(bucket, key, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(bucket);
  cudaFree(key);
}
