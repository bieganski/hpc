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
#include <thrust/functional.h>

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

__global__
void computeAC(const float* __restrict__ k,
               float*       __restrict__ ac,
               uint32_t*    __restrict__ comm) {

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    atomicAdd(&ac[comm[tid + 1]], k[tid + 1]);
}

__host__
void zeroAC(float* ac) {
    std::memset(ac, 0, sizeof(float) * (V_MAX_IDX + 1));
}

/**
 * This kernel function converts multiple nodes,
 * each node got it's own part of shared memory for node-common data, i.a hashArrays.
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
                        const uint32_t hasharrayEntries,
                        const float m,
                        const float minGain) {

    extern __shared__ KeyValueFloat hashtables[];

    assert(next_2_pow(maxDegree) == maxDegree); // maxDegree is power of 2         TODO usunąć przy wiekszych grafach
    
    int i_ptr = threadIdx.y + (blockIdx.y * blockDim.y); // my node pointer
    int edgeNum = threadIdx.x; // my edge pointer
    int i = binNodes[i_ptr];

    printf("XXXXXXXXXXXXXXXX     x + y: %u + %u ||| ||  %d\n", i_ptr, edgeNum, getGlobalIdx());

    uint32_t j = E[V[i] + edgeNum];
    printf(">> %d: (%d, %d): ogarniam sąsiada: %u\n", i, i_ptr, edgeNum, j);


    if (numNodes -1 < i_ptr) {
        printf("node:%u  - nie istnieję, jestem narzutem na blok\n", i_ptr);
        return;
    }
    
    if (V[i + 1] - V[i] -1 < edgeNum) {
        printf("(i, j): (%u, %u) jestem niepotrzebny\n", i, j);
        return;
    }

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
    __syncwarp();

    // ok, data initialized, let the run start
    uint32_t mySlot = HA::insertInt(hashComm, comm[j], comm[j], hasharrayEntries);
    float sum = HA::addFloatSpecificPos(hashWeight, mySlot, W[V[i] + edgeNum]);

    __syncwarp();

    // if (edgeNum == 0) {
    //     printf(">>from node (i=%d,v=%d)\n", i_ptr, i);
    //     for (int i =0 ; i < hasharrayEntries; i++) {
    //         printf("%d <<< comm: %d    weight: %f\n", i, hashComm[i].value, hashWeight[i].value);
    //     }
    // }

    // TODO istnieje funkcja ceilf, tylko zwraca floata

    uint32_t mask = __ballot_sync(FULL_MASK, edgeNum < maxDegree / 2);

    if (i_ptr + edgeNum == 0) {
        binprintf(mask);
        printf("\n");
        binprintf(1 << 5);
    }
    

    // sum of weights from node i to Ci\{i}
    float ei_to_Ci = comm[j] == comm[i] ? hashWeight[mySlot].value : 0;
    for (int offset = maxDegree / 2; offset > 0; offset /= 2)
        ei_to_Ci = fmaxf(ei_to_Ci, __shfl_down_sync(mask, ei_to_Ci, offset)); // only warp with idx % maxDegree == 0 keeps proper value

    if (edgeNum == 0) {
        printf("---%d, ei_to_ci: %f\n", i, ei_to_Ci);
    }

    // lack of -(e_i -> C_i\{i} / m) addend in that sum, it will be computed later
    float deltaMod = k[i] * ( ac[i] - k[i] - ac[comm[j]] ) / (2 * m * m)  +  hashWeight[mySlot].value / m;
    uint32_t newCommIdx = comm[j];
    
    printf("***%d, deltaMod: %f, a sprawdzam %d\n", i, deltaMod, newCommIdx);

    

    for (int offset = maxDegree / 2; offset > 0; offset /= 2) {
        float deltaModRed = __shfl_down_sync(mask, deltaMod, offset);
        uint32_t newCommIdxRed = __shfl_down_sync(mask, newCommIdx, offset);

        if (deltaModRed > deltaMod) {
            deltaMod = deltaModRed;
            newCommIdx = newCommIdxRed;
        } else if (deltaModRed == deltaMod) {
            newCommIdx = (uint32_t) fminf((uint32_t) newCommIdx, (uint32_t) newCommIdxRed);
        }
    }

    if (edgeNum == 0) {
        // this code executes once per vertex
        float gain = deltaMod + ei_to_Ci / m;

        if (gain >= minGain && newCommIdx < comm[i]) {
            newComm[i] = newCommIdx;
            printf("%u: wywalam do communiy %u, bo gain %f\n", i, newCommIdx, gain);
        } else {
            newComm[i] = comm[i];
        }
    }
}

__host__ 
float computeMod(float ei_to_Ci_sum, float m, const float* ac) {
    auto tmp = thrust::device_vector<float>(V_MAX_IDX + 1);
    thrust::transform(ac, ac + V_MAX_IDX + 1, tmp.begin(), thrust::square<float>());
    float sum = thrust::reduce(tmp.begin(), tmp.end(), (int) 0, thrust::plus<float>());

    return ei_to_Ci_sum / m - ( sum / (4 * m * m));
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
                        const uint32_t maxDegree,
                        const float m,
                        const float minGain) {
    assert(next_2_pow(maxDegree) == maxDegree); // max degrees up to power of 2 // mozna wylaczyc potem
    assert(maxDegree <= WARP_SIZE);

    uint32_t hashArrayEntriesPerNode = next_2_pow(maxDegree); // TODO customize this, maybe check 2 * maxDegree?

    assert(sizeof(KeyValueFloat) == sizeof(KeyValueInt));
    uint32_t hashArrayEntriesPerBlock = SHARED_MEM_SIZE / sizeof(KeyValueInt); // should be 384

    // remark, that we need 2 hash arrays per node
    uint32_t threadNum = std::min(binNodesNum, hashArrayEntriesPerBlock / (2 * hashArrayEntriesPerNode));

    uint32_t blockNum = binNodesNum % threadNum ? binNodesNum / threadNum + 1 : binNodesNum / threadNum;

    dim3 dimBlock(maxDegree, threadNum); // x per edges, y per nodes

    printf("maxDeg: %u \nall nodes in bucket: %u \nthreadDim:(%u, %u) \nblockNum:%u\n", 
        maxDegree, binNodesNum, threadNum, maxDegree, blockNum);

    // printf("debug: \nhashArrayEntriesPerNode: %u \nhashArrayEntriesPerBlock: %u\n", hashArrayEntriesPerNode, hashArrayEntriesPerBlock);
    printf(">>>> KERNEL RUNNING!\n\n");

    reassign_nodes<<<blockNum, dimBlock, SHARED_MEM_SIZE>>> (binNodesNum, binNodes, 
            V, E, W, k, ac, comm, newComm, maxDegree, threadNum, hashArrayEntriesPerNode, m, minGain);

    return 21.37;
}

__global__ void computeEiToCi(float*    __restrict__ ei_to_Ci,
                         const uint32_t* __restrict__ V,
                         const uint32_t* __restrict__ E,
                         const float*    __restrict__ W,
                         const uint32_t* __restrict__ comm) {
    uint32_t me = 1 + getGlobalIdx();
    assert(me = 1 + threadIdx.x + (blockIdx.x * blockDim.x)); // TODO wywalić
    uint32_t my_com = comm[me];

    for(uint32_t i = 0; i < V[me + 1] - V[me]; i++) {
        uint32_t comj = comm[ E[V[me] + i] ];

        if (my_com == comj) {
            printf("-- ME:    %d, %d, %f\n", me, comj, W[me + i]);
            atomicAdd(ei_to_Ci, W[me + i]);
        }
    }
}

// returns gain obtained
__host__ float reassign_communities(
                        const uint32_t V_MAX_IDX,
                        const uint32_t* __restrict__ V, 
                        const uint32_t* __restrict__ E,
                        const float*    __restrict__ W,
                        const float*    __restrict__ k,
                        float*    __restrict__ ac,
                        uint32_t* __restrict__ comm,
                        uint32_t* __restrict__ newComm,
                        float*    __restrict__ ei_to_Ci,
                        const float m,
                        const float minGain) {
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

    // those with at least one edge, otherwise no modularity influence
    // TODO wazne - czy ta zmienna jest potrzebna ? redundancja ?
    uint32_t numNodes = thrust::distance(it0, G.end());

    printf("XXXXX    all nodes: %d", numNodes);  // TODO check
    

    decltype(it0) it = it0;
    float gain = 0.0;


    auto all_nodes_pair = getBlockThreadSplit(numNodes);

    computeEiToCi <<<all_nodes_pair.first, all_nodes_pair.second>>> (ei_to_Ci, V, E, W, comm);
    cudaDeviceSynchronize();
    printf("EI 2 CI: %f\n", *ei_to_Ci);

    zeroAC(ac);
    computeAC<<<1, 5>>> (k, ac, comm);  // TODO tu jestesmy

    cudaDeviceSynchronize();
    computeMod(*ei_to_Ci, m, ac); // TODO tutaj bug

    while(true) {
        for (int i = 2; ; i++) {
            it = thrust::partition(it0, G.end(), partitionGenerator(i));
            assert(it == G.end()); // TODO wywalić

            uint32_t binNodesNum = thrust::distance(it0, it);
            if (binNodesNum == 0)
                break;

            uint32_t* binNodes = thrust::raw_pointer_cast(&it0[0]);

            printf(">>>BIN RUN: running %u nodes in bin, i (right)=%d\n", binNodesNum, i);
            reassign_communities_bin(binNodes, binNodesNum, V, E, W, k, ac, comm, newComm, maxDegree, m, minGain);

            cudaDeviceSynchronize();

            auto pair = getBlockThreadSplit(binNodesNum);

            printf("before update:\n");
            PRINT(comm + 1, comm + numNodes + 1);
            printf("\n");
            // update newComm table
            updateSpecific<<<pair.first, pair.second>>> (binNodes, binNodesNum, newComm, comm);
            cudaDeviceSynchronize();

            zeroAC(ac);
            computeAC<<<pair.first, pair.second>>> (k, ac, comm);
            cudaDeviceSynchronize();

            it0 = it;
        }
        break; // TODO
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
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(uint32_t) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&newComm, tmp, 0));
    HANDLE_ERROR(cudaHostAlloc((void**)&tmp, sizeof(float) * (V_MAX_IDX + 1), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostGetDevicePointer(&ac, tmp, 0));

    zeroAC(ac); // TODO wywalić

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
