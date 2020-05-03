#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <iostream>

#include "hasharray.h"

// using namespace std;

typedef HashArray HA;


static void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))
#define FULL_MASK 0xFFFFFFFF


// struct pred {
//     __host__ __device__
//     bool operator()(const int &x) {
//         return x < 3;
//     }
// };

#include <float.h>


__device__ 
void binprintf(uint32_t v)
{
    uint32_t mask = 1 << ((sizeof(uint32_t) << 3) - 1);
    while (mask) {
        printf("%u", (v & mask ? 1 : 0));
        mask >>= 1;
    }
    printf("\n");
}
__device__ __forceinline__ unsigned int __laneid() { unsigned int laneid; asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid)); return laneid; }

__global__ 
void wtf(uint32_t* ptr) { //KeyValueFloat* hashtable

    extern __shared__ float arr[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Tid: %d\n", __laneid());

    if (tid != 0)
        return;
    printf("%f\n", -FLT_MAX);

    printf("%f\n", -FLT_MAX + FLT_MAX);

    printf("%f\n", -FLT_MAX + 1000);

    return;


    int val = tid;
    if (tid == 1)
        return;

    int mask = 0x00000003; // FULL_MASK; // __activemask(); // __ballot_sync(FULL_MASK, 1);
    
    for (int offset = 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(mask, val, offset)); // only warp with idx == 0 keeps proper value
        if (tid == 0) {
            printf("_%d_\n", val);
        }
    }
        


    // int leader = __ffs(mask) - 1;
    // if (tid == 0) {
    //     printf("leader = %d\n", leader);
    //     printf("mask:\n");
    //     binprintf(mask);
    // }
    // int val = __laneid() == leader ? 5 : 1;

    // int res = __shfl_sync(mask, val, leader);

    if (tid == 0)
        printf("val = %d\n", val);



    // if(tid == 0){
    //     ;
    // }
    // else {
    //     return;
    // }
        
    // __syncthreads();
    
    // uint32_t res_key = HA::addFloat(hashtable, 1, 1.01, 2 << 5);

    // printf("%d: wstawilem pod %d, patrze: %f\n", tid, res_key, hashtable[res_key].value);
}


// __global__ void kernel() {
//   __shared__ int semaphore;
//   semaphore=0;
//   __syncthreads();
//   while (true) {
//     int prev=atomicCAS(&semaphore,0,1);
//     if (prev==0) {
//       //critical section
//       semaphore=0;
//       break;
//     }
//   }
// }

__device__ uint32_t CONTRACT_BINS[] = {
    0,
    121,
    385,
    UINT32_MAX
};

int main(void)
{
    KeyValueFloat* hashtable;

    HANDLE_ERROR(cudaHostAlloc((void**)&hashtable, sizeof(KeyValueFloat) * (2 << 5), cudaHostAllocDefault));
    cudaDeviceSynchronize();

    // uint32_t* ptr = (uint32_t*) malloc(4 * 6);

    // cudaMemcpyFromSymbol(ptr, CONTRACT_BINS, 4 * 4, 0, cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();
    // HA::init(hashtable, 2 << 5);

    // printf("LOL: %d\n", ptr[2]);
    // printf("LOLSIZE: %d\n", sizeof(CONTRACT_BINS));

    wtf<<<1, 32, 48 * 1024>>>((uint32_t*) hashtable);

    cudaDeviceSynchronize();

    printf("WYNIK: %d", * ((int*) hashtable));
    return 0;
}
