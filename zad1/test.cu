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

// struct pred {
//     __host__ __device__
//     bool operator()(const int &x) {
//         return x < 3;
//     }
// };

__global__ void kernel(float* ptr) { //KeyValueFloat* hashtable

    extern __shared__ float arr[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    arr[490000 * 1024 / sizeof(float)] = 0.2;

    *ptr = arr[490000 * 1024 / sizeof(float) - 15 + tid];

    if (tid > 30)
        return;

    __syncwarp((1 << 5) - 1);

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


static void HandleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))


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

    HANDLE_ERROR(cudaMalloc((void**) &hashtable, sizeof(KeyValueFloat) * (2 << 5)));
    cudaDeviceSynchronize();

    uint32_t* ptr = (uint32_t*) malloc(4 * 6);

    cudaMemcpyFromSymbol(ptr, CONTRACT_BINS, 4 * 4, 0, cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();
    // HA::init(hashtable, 2 << 5);

    printf("LOL: %d\n", ptr[2]);
    printf("LOLSIZE: %d\n", sizeof(CONTRACT_BINS));

    // kernel<<<1, 16, 490000 * 1024>>>((float*) hashtable);

    cudaDeviceSynchronize();
    return 0;
}














    // initialize all ten integers of a device_vector to 1
    // thrust::device_vector<int> D(10, 1);

    // // set the first seven elements of a vector to 9
    // thrust::fill(D.begin(), D.begin() + 7, 9);

    // thrust::sequence(D.begin(), D.end());

    // // print D
    // for(int i = 0; i < D.size(); i++)
    //     std::cout << "D[" << i << "] = " << D[i] << std::endl;

    // struct pred p;
    // auto it = thrust::partition(D.begin(), D.end(), p);


    // for(int i = 0; i < D.size(); i++)
    //     std::cout << "D[" << i << "] = " << D[i] << std::endl;

    // cout << endl << *it;