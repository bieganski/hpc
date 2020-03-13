#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define RADIUS        3000
#define NUM_ELEMENTS  1000000

#define NUM_THREADS_PER_BLOCK  32 
#define NUM_BLOCKS_PER_GRID    2

static void handleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))


__device__ int in_range(int idx) {
  if (idx >= 0 && idx <= NUM_ELEMENTS -1)
    return 1;
  else
    return 0;
}


__global__ void stencil_1d(int *in, int *out) {
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  // int all = blockDim.x * gridDim.x;
  // printf("me, all: %d, %d\n", idx, all);
  
  for(int j = 0; j <= RADIUS; j++) {
    int idx1 = i - j;
    if (!in_range(idx1))
      continue;
    out[i] += in[idx1];
    if (j == 0)
      continue;
    int idx2 = i + j;
    if (!in_range(idx2))
      continue;
    out[i] += in[idx2];
  }
}

void cpu_stencil_1d(int *in, int *out) {
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    for(int j = 0; j <= RADIUS; j++) {
      int idx1 = i - j;
      if (idx1 < 0)
        continue;
      out[i] += in[idx1];
      if (j == 0)
        continue;
      int idx2 = i + j;
        if (idx1 > NUM_ELEMENTS - 1)
          continue;
      out[i] += in[idx2];
    }
  }
}

int main() {
  //PUT YOUR CODE HERE - INPUT AND OUTPUT ARRAYS
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );

  int *dev_in, *dev_out;

  int BYTES_NUM = sizeof(int) * NUM_ELEMENTS;

  int *host_in = (int*)malloc(BYTES_NUM);
  int *host_out = (int*)malloc(BYTES_NUM);

  //PUT YOUR CODE HERE - DEVICE MEMORY ALLOCATION
  cudaMalloc((void**)&dev_in, BYTES_NUM);
  cudaMalloc((void**)&dev_out, BYTES_NUM);

  //PUT YOUR CODE HERE - KERNEL EXECUTION
  for(int i = 0; i < NUM_ELEMENTS; i++) {
    host_in[i] = i;
  }
  cudaMemcpy(dev_in, host_in, BYTES_NUM, cudaMemcpyHostToDevice);
  
  int num_blocks = NUM_ELEMENTS / NUM_THREADS_PER_BLOCK + 1;

  stencil_1d<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(dev_in, dev_out);

  // blockDim.x,y,z gives the number of threads in a block, in the particular direction
  // gridDim.x,y,z gives the number of blocks in a grid, in the particular direction

  
  cudaCheck(cudaPeekAtLastError());


  //PUT YOUR CODE HERE - COPY RESULT FROM DEVICE TO HOST
  cudaMemcpy(host_out, dev_out, BYTES_NUM, cudaMemcpyDeviceToHost);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop);
  printf("Total GPU execution time:  %3.1f ms\n", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //PUT YOUR CODE HERE - FREE DEVICE MEMORY  
  cudaFree(dev_in);
  cudaFree(dev_out);

  struct timespec cpu_start, cpu_stop;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

  cpu_stencil_1d(host_in, host_out);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
  double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf( "CPU execution time:  %3.1f ms\n", result);
  
  return 0;
}
