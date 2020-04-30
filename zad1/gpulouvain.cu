#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/distance.h>
#include <thrust/functional.h>

#include "hasharray.h"
#include "utils.h"
#include "modopt.h"

using namespace std; // TODO wywalić

using HA = HashArray;

float MIN_GAIN;
char *FILE_PATH;
bool VERBOSE = 0;

uint32_t* V; // vertices
uint32_t* E; // edges
float* W; // weights
float* k; // sum of weights per node
float* ac; // sum of weights per community
uint32_t* comm; // communities
uint32_t* newComm; // new communities after reassign phase
float m; // total sum of weights

uint32_t V_MAX_IDX;



__global__
void compute_size_degree(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V,
                          const uint32_t* __restrict__ comm,
                          uint32_t* __restrict__ commSize,
                          uint32_t* __restrict__ commDegree) {
    int tid = 1 + getGlobalIdx();
    assert(tid <= V_MAX_IDX);

    atomicAdd(&commSize[comm[tid]], 1);
    atomicAdd(&commDegree[comm[tid]], V[tid + 1] - V[tid]);

    __syncthreads(); // TODO - wywalić

    if (tid == 1) {
        printf("WYPISUJĘ OBLICZONE COMMUNITY SIZES: \n");
        for (int i = 1; i <=V_MAX_IDX; i++) {
            printf("%d ", commSize[i]); 
        }
        printf("\n");


        printf("WYPISUJĘ OBLICZONE COMMUNITY DEGREES: \n");
        for (int i = 1; i <=V_MAX_IDX; i++) {
            printf("%d ", commDegree[i]); 
        }
        printf("\n");
    }
}


#define NODE_EXISTS(i, V, E) (V[i+1] - V[i] > 0)

__global__
void compute_compressed_comm(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V,
                          const uint32_t* __restrict__ E,
                          const uint32_t* __restrict__ comm,
                          uint32_t* __restrict__ commSize,
                          uint32_t* __restrict__ vertexStart,
                          uint32_t* __restrict__ tmpCounter,
                          uint32_t* __restrict__ res) {
    int tid = 1 + getGlobalIdx();
    assert(tid <= V_MAX_IDX);

    if (!NODE_EXISTS(tid, V, E))
        return;

    int my_comm = comm[tid];

    int idx = atomicAdd(&tmpCounter[my_comm], 1);

    res[vertexStart[my_comm] + idx] = tid;
}

#define RAW(vec) (thrust::raw_pointer_cast(&(vec)[0]))

__device__ uint32_t CONTRACT_BINS[] = {
    0,
    121,
    385,
    UINT32_MAX
};

__host__
void contract(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V, 
                          const uint32_t* __restrict__ E,
                          const float*    __restrict__ W,
                          const float*    __restrict__ k,
                          const uint32_t* __restrict__ comm) {

    // TODO przenieśc je wyżej, żeby alokować tylko raz
    thrust::device_vector<uint32_t> commSize(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> commDegree(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> newID(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> vertexStart(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> tmpCounter(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> compressedComm(V_MAX_IDX + 1, 0);
    thrust::device_vector<uint32_t> commSeq(V_MAX_IDX + 1, 0);

    thrust::sequence(commSeq.begin(), commSeq.end());

    auto pair = getBlockThreadSplit(V_MAX_IDX);
    compute_size_degree<<<pair.first, pair.second>>> (V_MAX_IDX, V, comm, RAW(commSize), RAW(commDegree));

    cudaDeviceSynchronize();

    thrust::transform(commSize.begin(), commSize.end(), newID.begin(), [] __device__ (const uint32_t& size) {return size > 0 ? 1 : 0;});
    thrust::inclusive_scan(newID.begin(), newID.end(), newID.begin());
    
    thrust::inclusive_scan(commSize.begin(), commSize.end(), &vertexStart[1]); // start output at 1 
    printf("newID: \n");
    // PRINT(newID.begin(), newID.end());
    thrust::copy(newID.begin(),newID.end(), std::ostream_iterator<uint32_t>(std::cout, " "));
    printf("\n");

    compute_compressed_comm <<<pair.first, pair.second>>> (V_MAX_IDX, V, E, comm, 
            RAW(commSize), RAW(vertexStart), RAW(tmpCounter), RAW(compressedComm));


    cudaDeviceSynchronize();

    printf("VERTEX START: \n");
    thrust::copy(vertexStart.begin(), vertexStart.end(), std::ostream_iterator<uint32_t>(std::cout, " "));


    printf("\nCOMPRESSED COMM: \n");
    thrust::copy(compressedComm.begin(), compressedComm.end(), std::ostream_iterator<uint32_t>(std::cout, " "));


    auto commDegreeLambda = RAW(commDegree); // you cannot use thrust's vector in device code

    auto partitionGenerator = [=](int rightIdx) {
        return [=] __device__ (const uint32_t& i) {
            return commDegreeLambda[i] <= CONTRACT_BINS[rightIdx];
        };
    };

    // we don't want empty communities
    auto it0 = thrust::partition(commSeq.begin(), commSeq.end(), partitionGenerator(0));

    for (int i = 1; ; i++) {
        auto it = thrust::partition(commSeq.begin(), commSeq.end(), partitionGenerator(i));
        if (it == commSeq.end())
            break;
        // handle communities with same degree boundary

        uint32_t numCommunities = thrust::distance(it0, it);
        printf("Num comm: %d\n", numCommunities);

        return;

        // uint32_t hashArrayEntriesPerNode = maxDegree; // next_2_pow(maxDegree); // TODO customize this, maybe check 2 * maxDegree?
        // uint32_t hashArrayEntriesPerBlock = SHARED_MEM_SIZE / sizeof(KeyValueInt); // should be 384


        // uint32_t threadNum = std::min(numCommunities, hashArrayEntriesPerBlock / (2 * hashArrayEntriesPerNode));

        // uint32_t blockNum = binNodesNum % threadNum ? binNodesNum / threadNum + 1 : binNodesNum / threadNum;

        // dim3 dimBlock(maxDegree, threadNum); // x per edges, y per nodes
    }
    return;
}

__global__
void compute_comm_neighbors() {

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
    m = std::get<5>(res);

    HANDLE_ERROR(cudaHostGetDevicePointer(&V, std::get<1>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&E, std::get<2>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&W, std::get<3>(res), 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&k, std::get<4>(res), 0));

    uint32_t* tmp;
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(uint32_t) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&newComm, tmp, 0));
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(float) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&ac, tmp, 0));

    // communities separately, because they must be initialized
    // auto _comm = thrust::device_vector<uint32_t>(V_MAX_IDX + 1);
    // thrust::sequence(_comm.begin(), _comm.end());
    // comm = thrust::raw_pointer_cast(&_comm[0]);


    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(uint32_t) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    for (int i = 0; i <= V_MAX_IDX; i++) {
        tmp[i] = i;
    }
    HANDLE_ERROR(cudaHostGetDevicePointer(&comm, tmp, 0));
    // TODO ogarnąć to
    // auto _ei_to_Ci = thrust::device_vector<float>(V_MAX_IDX + 1, 0);
    // float* ei_to_Ci = thrust::raw_pointer_cast(&_ei_to_Ci[0]);

    float* ei_to_Ci;
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(float), cudaHostAllocDefault));
    *tmp = (float) 0;
    HANDLE_ERROR(cudaHostGetDevicePointer(&ei_to_Ci, tmp, 0));
    

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte); // TODO customize
    // cudaFuncSetCacheConfig(reassign_nodes, cudaFuncCachePreferShared);

    cudaDeviceSynchronize();

    float gain = reassign_communities(V_MAX_IDX, V, E, W, k, ac, comm, newComm, ei_to_Ci, m, MIN_GAIN);

    // cudaFree(V);
    return 0;
}
