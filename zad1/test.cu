#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <iostream>
#include <math.h>

#include <thrust/generate.h>

__device__
int getGlobalIdx(){
    int numInRow = blockDim.x * gridDim.x;
    int numInBlock = blockDim.x * blockDim.y;
    return blockIdx.y * numInRow + blockIdx.x * numInBlock + threadIdx.y * blockDim.x + threadIdx.x;
    // return blockIdx.x * blockDim.x * blockDim.y
    // + threadIdx.y * blockDim.x + threadIdx.x;
}

__host__ static __inline__ float rand_01()
{
    return ((float)rand()/RAND_MAX);
}

__global__
void test() {
    int idx = getGlobalIdx();
    int x = threadIdx.x; // blockIdx.x * blockDim.x + threadIdx.x;
    int y = threadIdx.y; // blockIdx.y * blockDim.y + threadIdx.y;

    // if (x == 0 && y == 0) {
    //     printf("%d, %d\n", blockIdx.x, blockIdx.y);
    // }

    int bigX = blockIdx.x * blockDim.x + threadIdx.x;
    int bigY = blockIdx.y * blockDim.y + threadIdx.y;
    if (bigX == 0) {
        printf("%d\n", bigY);
    }

    return;
    // printf("%d: (%d, %d)\n", idx, x , y);
}

int main(){
    
    dim3 lolblock(32, 32);
    dim3 lolgrid(ceil(400.0 / 32.0), ceil(900.0 / 32.0));
    test<<<lolgrid, lolblock>>>();

    cudaDeviceSynchronize();

    return 0;
}
