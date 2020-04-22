#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <thrust/partition.h>

#include "hasharray.h"
#include "utils.h"

using namespace std;

float MIN_GAIN;
char *FILE_PATH;
bool VERBOSE = 0;

uint32_t* V; // vertices
uint32_t* E; // edges
float* W; // weights
float* k; // sum of weights

uint32_t V_MAX_IDX;

struct partition_pred {
    __host__ __device__
    bool operator()(const int &x) {
        return x < 3;
    }
};


// [ BINS[i], BINS[i+1] ) (left side exclusive)
__device__ uint32_t BINS[] =  
    {
        0,
        5,
        9,
        17,
        33,
        84,
        319,
        UINT32_MAX
    };


#define THREADS_PER_BLOCK 128 // 4 warps


__global__ void son() {
    int tid = threadIdx.x + blockIdx.x + blockDim.x;
    printf("son: %d (%d, %d)\n", tid, blockIdx.x, threadIdx.x);
}

__global__ void parent() {
    int tid = threadIdx.x + blockIdx.x + blockDim.x;
    printf("parent: %d (%d, %d)\n", tid, blockIdx.x, threadIdx.x);
    son<<<2,2>>>();
}


__global__ void compute_move_single_edge(uint32_t* __restrict__ V, 
                        uint32_t V_NUM,
                        uint32_t* __restrict__ E,
                        uint32_t E_NUM,
                        uint32_t* __restrict__ W,
                        uint32_t* __restrict__ comm,
                        uint32_t* __restrict__ ac,
                        uint32_t node) {
    // pass


}

__host__ void compute_move_host(uint32_t* __restrict__ V, 
                        uint32_t V_NUM,
                        uint32_t* __restrict__ E,
                        uint32_t E_NUM,
                        uint32_t* __restrict__ W,
                        uint32_t* __restrict__ comm,
                        uint32_t* __restrict__ ac) {
    // pass
}

/**
 * UWAGA:
 * N - rozmiar tablicy k
 * N + 1 - rozmiar tablicy V
 * V - iterujemy od 1
 * E, W - rozmiar 2*N, iterujemy od 0
 * */
int main(int argc, char **argv) {
    if (parse_args(argc, argv)) {
        exit(1);
    }
    
    ret_t res = parse_inut_graph(get_input_content());

    V_MAX_IDX = std::get<0>(res);

    HANDLE_ERROR(cudaHostGetDevicePointer(&V, std::get<1>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&E, std::get<2>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&W, std::get<3>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&k, std::get<4>(res), 0));

    cout << "WMAX: " << V_MAX_IDX << "\n";
    for(int i = 0; i <= V_MAX_IDX; i++)
        cout << k[i] << "," << endl;

    struct partition_pred p;
    auto it = thrust::partition(V, V + V_MAX_IDX + 1, p);

    parent<<<2,2>>>();
    
    cudaDeviceSynchronize();

    return 0;
}

