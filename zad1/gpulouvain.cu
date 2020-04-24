#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/distance.h>

#include "hasharray.h"
#include "utils.h"

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


#define WARP_SIZE 32
#define SHARED_MEM_SIZE (48 * 1024)



// [ BINS[i], BINS[i+1] ) (right side exclusive)
__device__ uint32_t BINS[] =  
    {
        0, // [0,1) is handled separately (lonely nodes; their modularity impact is 0)
        1,
        5,
        9,
        17,
        33,
        97, // up to 3 hashed edges per thread in warp
        385, // up to _ hashed edges per thread in warp, but still in shared memory
        UINT32_MAX // hash arrays in global memory
    };


#define FULL_MASK 0xffffffff // TODO wywalić


/**
 * This kernel function converts multiple nodes,
 * each node got it's own part of shared memory for node-common data.
 * Graph description array is shifted, i.a thread with idx.x = 3 accesses node V_shift + 3.
 */
__global__ void reassign_nodes(
                        const uint32_t  numNodes,
                        const uint32_t* __restrict__ binNodes,
                        const uint32_t* __restrict__ V,
                        const uint32_t* __restrict__ E,
                        const float*    __restrict__ W,
                        const float*    __restrict__ k,
                        const float*    __restrict__ ac,
                        const uint32_t* __restrict__ comm,
                        uint32_t*       __restrict__ newComm,
                        const uint32_t maxDegree,
                        const uint32_t nodesPerBlock,
                        const uint32_t hasharrayEntries) {

    extern __shared__ KeyValueFloat hashtables[];

    assert(next_2_pow(maxDegree) == maxDegree); // maxDegree is power of 2         TODO usunąć przy wiekszych grafach
    
    int i_ptr = threadIdx.x + (blockIdx.x * blockDim.x); // my node pointer
    int edgeNum = threadIdx.y; // my edge pointer
    int nodeNum = binNodes[i_ptr];

    if (numNodes -1 < i_ptr) {
        printf("node:%u  - nie istnieję, jestem narzutem na blok\n", i_ptr);
        return;
    }
    
    if (V[nodeNum + 1] - V[nodeNum] -1 < edgeNum) {
        // printf("ne: (%u, %u) jestem niepotrzebny\n", i_ptr, edgeNum);
        return;
    }

    uint32_t j = E[V[nodeNum] + edgeNum];
    // printf(">> %d: (%d, %d): ogarniam sąsiada: %u\n", nodeNum, i_ptr, edgeNum, j);


    // hasharray_size elements each, first one up to `hasharray_size -1`
    KeyValueFloat* hashWeight = (KeyValueFloat*) hashtables + i_ptr * (2 * hasharrayEntries);
    KeyValueInt*   hashComm   = (KeyValueInt*)   hashWeight + hasharrayEntries;


    // TODO tu jest za dużo roboty
    for (int i = 0; i < hasharrayEntries; i++) {
        hashWeight[i].key = hashArrayNull;
        hashWeight[i].value = (float) 0;
        hashComm[i].key = hashArrayNull;
        hashComm[i].value = hashArrayNull;
    }
    __syncthreads();

    // ok, data initialized, let the run start
    uint32_t mySlot = HA::insertInt(hashComm, j, j, hasharrayEntries);
    float sum = HA::addFloatSpecificPos(hashWeight, mySlot, W[V[nodeNum] + edgeNum]);

    __syncthreads();

    // if (edgeNum == 0) {
    //     printf(">>from node (i=%d,v=%d)\n", i_ptr, nodeNum);
    //     for (int i =0 ; i < hasharrayEntries; i++) {
    //         printf("%d <<< comm: %d    weight: %f\n", nodeNum, hashComm[i].value, hashWeight[i].value);
    //     }
    // }


    // TODO obliczanie modularity
}

__host__ float reassign_communities_bin(
                        const uint32_t* binNodes,
                        const uint32_t binNodesNum,
                        const uint32_t* __restrict__ V, 
                        const uint32_t* __restrict__ E,
                        const float*    __restrict__ W,
                        const float*    __restrict__ k,
                        const float*    __restrict__ ac,
                        uint32_t* __restrict__ comm,
                        uint32_t* __restrict__ newComm,
                        uint32_t maxDegree) {
    assert(next_2_pow(maxDegree) == maxDegree); // max degrees up to power of 2 // mozna wylaczyc potem
    assert(maxDegree <= WARP_SIZE);

    uint32_t hashArrayEntriesPerNode = next_2_pow(maxDegree); // TODO customize this, maybe check 2 * maxDegree?

    assert(sizeof(KeyValueFloat) == sizeof(KeyValueInt));
    uint32_t hashArrayEntriesPerBlock = SHARED_MEM_SIZE / sizeof(KeyValueInt); // should be 384

    // remark, that we need 2 hash arrays per node
    uint32_t threadNum = std::min(binNodesNum, hashArrayEntriesPerBlock / (2 * hashArrayEntriesPerNode));

    uint32_t blockNum = binNodesNum % threadNum ? binNodesNum / threadNum + 1 : binNodesNum / threadNum;

    dim3 dimBlock(threadNum, maxDegree); // x per vertices, y per edge

    printf("maxDeg: %u \nall nodes in bucket: %u \nthreadDim:(%u, %u) \nblockNum:%u\n", 
        maxDegree, binNodesNum, threadNum, maxDegree, blockNum);

    // printf("debug: \nhashArrayEntriesPerNode: %u \nhashArrayEntriesPerBlock: %u\n", hashArrayEntriesPerNode, hashArrayEntriesPerBlock);
    printf(">>>> KERNEL RUNNING!\n\n");

    reassign_nodes<<<blockNum, dimBlock, SHARED_MEM_SIZE>>> (binNodesNum, binNodes, 
            V, E, W, k, ac, comm, newComm, maxDegree, threadNum, hashArrayEntriesPerNode);

    return 21.37;
}

// returns gain obtained
__host__ float reassign_communities(
                        const uint32_t V_MAX_IDX,
                        const uint32_t* __restrict__ V, 
                        const uint32_t* __restrict__ E,
                        const float*    __restrict__ W,
                        const float*    __restrict__ k,
                        const float*    __restrict__ ac,
                        uint32_t* __restrict__ comm,
                        uint32_t* __restrict__ newComm) {
    // TODO more than one bin
    uint32_t maxDegree = 4;

    thrust::device_vector<uint32_t> G(V_MAX_IDX);

    thrust::sequence(G.begin(), G.end(), 1);

    auto partitionGenerator = [=](int rightIdx) {
        return [=] __device__ (const uint32_t& i) {
            return V[i + 1] - V[i] <= BINS[rightIdx];
        };
    };

    // [0,1) is handled separately (lonely nodes; their modularity impact is 0)
    auto it0 = thrust::partition(G.begin(), G.end(), partitionGenerator(1));

    if (thrust::distance(G.begin(), it0) != 0) {
        // TODO - wywalić to
        assert(false && "single nodes detected!");
    }

    decltype(it0) it = it0;
    float gain = 0.0;

    for (int i = 2; ; i++) {
        it = thrust::partition(it0, G.end(), partitionGenerator(i));
        assert(it == G.end()); // TODO wywalić

        uint32_t binNodesNum = thrust::distance(it0, it);
        if (binNodesNum == 0)
            break;

        uint32_t* binNodes = thrust::raw_pointer_cast(&it0[0]);

        printf(">>>BIN RUN: running %u nodes in bin, i (right)=%d\n", binNodesNum, i);
        reassign_communities_bin(binNodes, binNodesNum, V, E, W, k, ac, comm, newComm, maxDegree);

        it0 = it;
    }

    return gain;
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
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, V_MAX_IDX + 1, cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&newComm, tmp, 0));
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, V_MAX_IDX + 1, cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&comm, tmp, 0));
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, V_MAX_IDX + 1, cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&ac, tmp, 0));

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte); // TODO customize
    // cudaFuncSetCacheConfig(reassign_nodes, cudaFuncCachePreferShared);

    cudaDeviceSynchronize();

    float gain = reassign_communities(V_MAX_IDX, V, E, W, k, ac, comm, newComm);

    // cudaFree(V);
    return 0;
}

