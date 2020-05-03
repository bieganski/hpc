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
#include <float.h>

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/distance.h>
#include <thrust/functional.h>

#include "hasharray.h"
#include "utils.h"

#include "modopt.h"


/**
 * TODO
 * 
 * TODO
 * 
 */


__host__
void contract(const uint32_t V_MAX_IDX,
                          const uint32_t* __restrict__ V, 
                          const uint32_t* __restrict__ E,
                          const float*    __restrict__ W,
                          const float*    __restrict__ k,
                          const uint32_t* __restrict__ comm);


using HA = HashArray;

// [ BINS[i], BINS[i+1] ) (right side exclusive)
__device__ uint32_t BINS[] =  
    {
        0, // [0,1) is handled separately (lonely nodes; their modularity impact is 0)
        1,
        4,
        8,
        16,
        32,
        96, // up to 3 hashed edges per thread in warp
        384, // up to _ hashed edges per thread in warp, but still in shared memory
        UINT32_MAX // hash arrays in global memory
    };

__global__
void computeAC(const float* __restrict__ k,
               float*       __restrict__ ac,
               const uint32_t*    __restrict__ comm) {

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    atomicAdd(&ac[comm[tid + 1]], k[tid + 1]);
}

__host__
void zeroAC(float* ac, uint32_t V_MAX_IDX) {
    std::memset(ac, 0, sizeof(float) * (V_MAX_IDX + 1));
}

/**
 * This kernel function converts multiple nodes,
 * each node got it's own part of shared memory for node-common data, i.a hashArrays.
 */
__global__ 
void reassign_nodes(
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
                        const float m) {

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
    assert(mask == __ballot_sync(__activemask(), edgeNum < maxDegree / 2)); // TODO - to się kiedyś wysypie i dobrze, wtedy podmienić mask

    if (i_ptr + edgeNum == 0) {
        binprintf(mask);
        printf("\n");
        binprintf(1 << 5);
    }
    

    printf("WAGA: (comm %d, comm %d) = %f\n", comm[i], comm[j], hashWeight[mySlot].value);

    // sum of weights from node i to Ci\{i} from paper doesn't use loop values
    float loop = i == j ? W[V[i] + edgeNum] : 0;
    float ei_to_Ci = comm[j] == comm[i] ? hashWeight[mySlot].value : 0;

    float todo = i == j ? W[V[i] + edgeNum] : 0;

    __syncwarp();

    if (j == 6) {
        printf("%d: AAA: %f\n", edgeNum, ei_to_Ci);
    }

    for (int offset = maxDegree / 2; offset > 0; offset /= 2) {
        ei_to_Ci = fmaxf(ei_to_Ci, __shfl_down_sync(mask, ei_to_Ci, offset)); // only warp with idx % maxDegree == 0 keeps proper value
        loop = fmaxf(loop, __shfl_down_sync(mask, loop, offset));
        // if (j == 6) {
        //     printf("%d: AAAXD: %f\n", edgeNum, ei_to_Ci);
        // }
        todo += __shfl_down_sync(mask, todo, offset);
    }
   

    // TODO nie usuwaj, wazny assert
    if (edgeNum == 0) {
        assert(todo == loop);
        ei_to_Ci -= loop;
        printf("---%d, ei_to_ci: %f\n", i, ei_to_Ci);
    }

    // TODO zrobic dobrze
    // lack of -(e_i -> C_i\{i} / m) addend in that sum, it will be computed later
    float deltaMod = comm[j] >= comm[i] ? 
        -(1 << 5) : 
        k[i] * ( ac[comm[i]] - k[i] - ac[comm[j]] ) / (2 * m * m)  +  hashWeight[mySlot].value / m;

    uint32_t newCommIdx = comm[j];

    if (i == 6) {
        printf("e_i_cj: %f, m: %f, k_i: %f, a_c_i_minus_i: %f, a_c_j: %f\n", hashWeight[mySlot].value, m, k[i], ac[i] - k[i], ac[comm[j]]);
    }
    

    
    if (deltaMod > 0)
        printf("***%d, deltaMod: %f, a sprawdzam %d\n", i, deltaMod, newCommIdx);

    

    for (int offset = maxDegree / 2; offset > 0; offset /= 2) {
        float deltaModRed = __shfl_down_sync(mask, deltaMod, offset);
        uint32_t newCommIdxRed = __shfl_down_sync(mask, newCommIdx, offset);

        if (newCommIdxRed == 0)
            continue; // TODO - brzydki hack na undefined behavior

        if (deltaModRed > deltaMod) {
            deltaMod = deltaModRed;
            newCommIdx = newCommIdxRed;
        } else if (deltaModRed == deltaMod) {
            newCommIdx = (uint32_t) fminf((uint32_t) newCommIdx, (uint32_t) newCommIdxRed);
        }
    }

    if (edgeNum == 0) {
        // this code executes once per vertex
        if (i == 6) {
            printf("AAABBB: %f\n", ei_to_Ci);
        }
        float gain = deltaMod - ei_to_Ci / m;
        printf(" ~~~~~~~~~~~~~~~~ *MAX_GAIN for %d: %f (to node %d)\n", i, gain, newCommIdx);
        if (gain > 0 && newCommIdx < comm[i]) {
            newComm[i] = newCommIdx;
            printf("~~~~~~~~~~~~~~~~ *%u: wywalam do communiy %u, bo gain %f\n", i, newCommIdx, gain);
        } else {
            newComm[i] = comm[i];
        }
    }
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
                        const float m) {
    assert(next_2_pow(maxDegree) == maxDegree); // max degrees up to power of 2 // mozna wylaczyc potem
    assert(maxDegree <= WARP_SIZE);

    uint32_t hashArrayEntriesPerComm = next_2_pow(maxDegree); // TODO customize this, maybe check 2 * maxDegree?

    assert(sizeof(KeyValueFloat) == sizeof(KeyValueInt));
    uint32_t hashArrayEntriesPerBlock = SHARED_MEM_SIZE / sizeof(KeyValueInt); // should be 6144

    // remark, that we need 2 hash arrays per node
    uint32_t threadNum = std::min(binNodesNum, hashArrayEntriesPerBlock / (2 * hashArrayEntriesPerComm));

    uint32_t blockNum = ceil(binNodesNum / threadNum);

    dim3 dimBlock(maxDegree, threadNum); // x per edges, y per nodes

    printf("maxDeg: %u \nall nodes in bucket: %u \nthreadDim:(%u, %u) \nblockNum:%u\n", 
        maxDegree, binNodesNum, threadNum, maxDegree, blockNum);

    // printf("debug: \nhashArrayEntriesPerNode: %u \nhashArrayEntriesPerBlock: %u\n", hashArrayEntriesPerComm, hashArrayEntriesPerBlock);
    printf(">>>> KERNEL RUNNING!\n\n");

    uint32_t shm = (2 * hashArrayEntriesPerComm) * sizeof(KeyValueInt) * threadNum;

    if (shm > SHARED_MEM_SIZE)
        assert(false);

    reassign_nodes<<<blockNum, dimBlock, shm>>> (binNodesNum, binNodes, 
            V, E, W, k, ac, comm, newComm, maxDegree, threadNum, hashArrayEntriesPerComm, m);

    return 21.37;
}

__global__ 
void computeEiToCiSum(float*    __restrict__ ei_to_Ci,
                         const uint32_t* __restrict__ V,
                         const uint32_t* __restrict__ E,
                         const float*    __restrict__ W,
                         const uint32_t* __restrict__ comm) {
    uint32_t me = 1 + getGlobalIdx();
    assert(me = 1 + threadIdx.x + (blockIdx.x * blockDim.x)); // TODO wywalić
    uint32_t my_com = comm[me];

    for(uint32_t i = 0; i < V[me + 1] - V[me]; i++) {
        uint32_t comj = comm[ E[V[me] + i] ];
        
        if (E[V[me] + i] == me) {
            // loop edge, do nothing
            continue;
        }

        if (my_com == comj) {
            atomicAdd(ei_to_Ci, W[me + i]);
        }
    }
}

__host__ 
float __computeMod(float ei_to_Ci_sum, float m, const float* ac, uint32_t V_MAX_IDX) {
    auto tmp = thrust::device_vector<float>(V_MAX_IDX + 1);
    thrust::transform(ac, ac + V_MAX_IDX + 1, tmp.begin(), thrust::square<float>());
    float sum = thrust::reduce(tmp.begin(), tmp.end(), (int) 0, thrust::plus<float>());

    return ei_to_Ci_sum / m - ( sum / (4 * m * m));
}


__host__
float computeModAndAC(uint32_t V_MAX_IDX,
                const uint32_t* __restrict__ V,
                const uint32_t* __restrict__ E,
                const float*    __restrict__ W,
                const float*    __restrict__ k,
                const uint32_t* __restrict__ comm,
                float* __restrict__ ac,
                float m) {
    
    float* ei_to_Ci;
    HANDLE_ERROR(cudaHostAlloc((void**)&ei_to_Ci, sizeof(float), cudaHostAllocDefault));
    *ei_to_Ci = 0;
    HANDLE_ERROR(cudaHostGetDevicePointer(&ei_to_Ci, ei_to_Ci, 0));
    
    auto all_nodes_pair = getBlockThreadSplit(V_MAX_IDX);

    computeEiToCiSum <<<all_nodes_pair.first, all_nodes_pair.second>>> (ei_to_Ci, V, E, W, comm);
    cudaDeviceSynchronize();

    zeroAC(ac, V_MAX_IDX);
    computeAC<<<all_nodes_pair.first, all_nodes_pair.second>>> (k, ac, comm);
    cudaDeviceSynchronize();

    printf("TODO: ei2CI: %f\n", *ei_to_Ci);
    return __computeMod(*ei_to_Ci, m, ac, V_MAX_IDX);
}


__host__ 
float reassign_communities(
                        const uint32_t V_MAX_IDX,
                        const uint32_t* __restrict__ V, 
                        const uint32_t* __restrict__ E,
                        const float*    __restrict__ W,
                        const float*    __restrict__ k,
                        float*    __restrict__ ac,
                        uint32_t* __restrict__ comm,
                        uint32_t* __restrict__ newComm,
                        const float m,
                        const float minGain) {

    // TODO free
    uint32_t* binsHost = (uint32_t*) malloc(sizeof(BINS));
    cudaMemcpyFromSymbol(binsHost, BINS, sizeof(BINS), 0, cudaMemcpyDeviceToHost);


    thrust::device_vector<uint32_t> G(V_MAX_IDX);
    thrust::sequence(G.begin(), G.end(), 1);

    auto partitionGenerator = [=](int rightIdx) {
        return [=] __device__ (const uint32_t& i) {
            return V[i + 1] - V[i] <= BINS[rightIdx];
        };
    };

    // [0,1) is handled separately (lonely nodes; their modularity impact is 0)
    auto it0 = thrust::partition(G.begin(), G.end(), partitionGenerator(1));
    auto it = it0;

    if (thrust::distance(G.begin(), it0) != 0) {
        // TODO - wywalić to
        assert(false && "single nodes detected!");
    }
    assert(V_MAX_IDX == thrust::distance(G.begin(), G.end()));

    

    float mod0, mod1;    
    mod0 = computeModAndAC(V_MAX_IDX, V, E, W, k, comm, ac, m);
    printf("MOD0 = %f\n", mod0);

    // thrust::copy(ac, (ac + V_MAX_IDX + 1), std::ostream_iterator<float>(std::cout, " "));

    while(true) {

        // for each bin sequentially computes new communities
        for (int i = 2; ; i++) {
            it = thrust::partition(it0, G.end(), partitionGenerator(i));

            // printf("\nBIN nodes with maxDeg <= %d\n", binsHost[i]);
            // thrust::copy(it0, it, std::ostream_iterator<uint32_t>(std::cout, " "));

            uint32_t binNodesNum = thrust::distance(it0, it);
            if (binNodesNum == 0)
                break;

            uint32_t* binNodes = RAW(it0);

            uint32_t maxDegree = binsHost[i];

            printf(">>>BIN RUN: running %u nodes in bin, i (right)=%d\n", binNodesNum, i);
            reassign_communities_bin(binNodes, binNodesNum, V, E, W, k, ac, comm, newComm, maxDegree, m);

            cudaDeviceSynchronize();

            auto pair = getBlockThreadSplit(binNodesNum);

            // printf("before update:\n");
            // PRINT(comm + 1, comm + numNodes + 1);
            // printf("\n");

            // update newComm table
            updateSpecific<<<pair.first, pair.second>>> (binNodes, binNodesNum, newComm, comm);
            cudaDeviceSynchronize();

            zeroAC(ac, V_MAX_IDX);
            computeAC<<<pair.first, pair.second>>> (k, ac, comm);
            cudaDeviceSynchronize();

            it0 = it;
        }

        // OK, we computed new communities for all bins, let's check whether
        // modularity gain is satisfying

        printf("*****************                 ASSIGNMENT      ****************\n");

        print_comm_assignment(V_MAX_IDX, comm);

        mod1 = computeModAndAC(V_MAX_IDX, V, E, W, k, comm, ac, m);

        printf("END: mod gain: %f\n", mod1 - mod0);

        printf("------------------AC: \n");
        // thrust::copy(ac, ac + V_MAX_IDX, std::ostream_iterator<float>(std::cout, " "));

        return mod1;



        if (mod1 - mod0 < 0) {
            assert(false);
        } else if (mod1 - mod0 < minGain) {

            contract(V_MAX_IDX, V, E, W, k, comm);
            printf("\n*****************                 CONTRACT                 ****************\n");
            print_comm_assignment(V_MAX_IDX, comm);
            cudaDeviceSynchronize();
            break; // TODO, poprawić contract
        } else {
            printf("going to next modularity iteration (mod gain sufficient): mod 0, 1: %f, %f\n", mod0, mod1);
            mod0 = mod1;
        }
    }

    return mod1;
}
