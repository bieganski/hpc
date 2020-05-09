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
                          uint32_t* __restrict__ V, 
                          uint32_t* __restrict__ E,
                          float*    __restrict__ W,
                          float*    __restrict__ k,
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

__device__ 
__forceinline__ 
unsigned int __lane_id() { 
    unsigned int laneid; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid)); 
    return laneid; 
}



// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// __device__ 
// int atomicAggInc(int *ptr) {
//     int mask = __match_any_sync(__activemask(), (unsigned long long) ptr);
//     int leader = __ffs(mask) - 1;    // select a leader
//     int res;
//     if (__lane_id() == leader)                  // leader does the update
//         res = atomicAdd(ptr, __popc(mask));
//     res = __shfl_sync(mask, res, leader);    // get leader’s old value
//     return res + __popc(mask & ((1 << __lane_id()) - 1)); //compute old value
// }

// __device__ 
// __forceinline__ 
// uint64_t packed_ei_to_ci(uint32_t ei_to_ci, uint32_t ) { 
//     unsigned int laneid; 
//     asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid)); 
//     return laneid; 
// }


// https://fgiesen.wordpress.com/2013/01/21/order-preserving-bijections/
__device__
__forceinline__
uint32_t int_to_uint(int32_t val) {
    return val ^ 0x80000000;
}

__device__
__forceinline__
int32_t uint_to_int(uint32_t val) {
    return val ^ 0x80000000;
}

__device__
__forceinline__
int32_t float_to_int(float f32_val) {
    int32_t tmp = __float_as_int(f32_val);
    return tmp ^ ((tmp >> 31) & 0x7fffffff);
}

__device__
__forceinline__
float int_to_float(int32_t i32_val) {
    int32_t tmp = i32_val ^ ((i32_val >> 31) & 0x7fffffff);
    return __int_as_float(tmp);
}

__device__
__forceinline__
float uint_to_float(uint32_t ui32_val) {
    int32_t tmp1 = ui32_val ^ 0x80000000;
    int32_t tmp2 = tmp1 ^ (tmp1 >> 31);
    return __uint_as_float(tmp2);
}

__device__
__forceinline__
uint32_t float_to_uint(float f32_val) {
    uint32_t tmp = __float_as_uint(f32_val);
    return tmp ^ (((int32_t)tmp >> 31) | 0x80000000);
}

#define VAR_MEM_PER_VERTEX_BYTES_DEFINE 16

/**
 * Version of 'reassign_nodes' that handles nodes with degree > 32.
 * It deduces whether total size of hasharrays fits into shared memory,
 * if no, then it uses global memory.
 */
__global__ 
void reassign_huge_nodes(
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
                        const void*    globalHasharray) {

    extern __shared__ char shared_mem[]; // shared memory is one-byte char type, for easy offset applying

    int i_ptr = threadIdx.y + (blockIdx.y * blockDim.y); // my vertex pointer
    int edge_ptr = threadIdx.x; // my edge pointer

    assert(edge_ptr < maxDegree);

    // before any early return, let's utilize all threads for zeroing memory.

    KeyValueFloat* hashWeight;
    KeyValueInt*   hashComm;

    // TODO: customize these
    // ei_to_Ci, loop, deltaModRes
    uint32_t VAR_MEM_PER_VERTEX_BYTES = sizeof(float) + sizeof(float) + sizeof(uint64_t);

    assert(VAR_MEM_PER_VERTEX_BYTES_DEFINE == VAR_MEM_PER_VERTEX_BYTES);

    uint32_t COMMON_VARS_SIZE_BYTES = VAR_MEM_PER_VERTEX_BYTES * numNodes;
    // TODO customize this
    bool shared = COMMON_VARS_SIZE_BYTES + (2 * hasharrayEntries * sizeof(KeyValueInt)) * numNodes <= SHARED_MEM_SIZE; 

    assert(shared);

    // very careful pointer handling here, because of lots of type mismatches and casting
    if (shared) {
        uint32_t off = COMMON_VARS_SIZE_BYTES + i_ptr * (2 * hasharrayEntries) * sizeof(KeyValueFloat);
        hashWeight = (KeyValueFloat*) &shared_mem[off];
        assert( shared_mem + COMMON_VARS_SIZE_BYTES <= (char*) hashWeight);
        hashComm   = (KeyValueInt*)   hashWeight + hasharrayEntries;
        assert(globalHasharray == nullptr);
    } else {
        hashWeight = (KeyValueFloat*) globalHasharray + i_ptr * (2 * hasharrayEntries);
        hashComm   = (KeyValueInt*)   hashWeight + hasharrayEntries;
    }
    

    // TODO za dużo roboty
    for (int i = edge_ptr; i < hasharrayEntries; i += maxDegree) {
        hashWeight[i] = {.key = hashArrayNull, .value = (float) 0}; // 0 for easy atomicAdd
        hashComm[i]   = {.key = hashArrayNull, .value = hashArrayNull};
    }

    if (numNodes -1 < i_ptr) {
        printf("node:%u  - nie istnieję, jestem narzutem na blok\n", i_ptr);
        return;
    }
    int i = binNodes[i_ptr]; // my vertex

    if (V[i + 1] - V[i] -1 < edge_ptr) {
        // printf("(i, j_ptr): (%u, %u) jestem niepotrzebny\n", i, edge_ptr);
        return;
    }
    uint32_t j = E[V[i] + edge_ptr]; //my neighbor
    
    // printf("XXXXXXXXXXXXXXXX     x + y: %u + %u ||| ||  %d\n", i_ptr, edge_ptr, getGlobalIdx());
    // printf(">> %d: (%d, %d): ogarniam sąsiada: %u\n", i, i_ptr, edge_ptr, j);


    // variables common for each vertex, accumulating ei_to_Ci value 
    // computed in parallel
    uint32_t ei_to_ci_off_bytes = i_ptr * VAR_MEM_PER_VERTEX_BYTES;
    int32_t* glob_ei_to_Ci = (int32_t*) &shared_mem[ei_to_ci_off_bytes];
    
    uint32_t loop_off_bytes = ei_to_ci_off_bytes + sizeof(int32_t);
    assert(ei_to_ci_off_bytes < loop_off_bytes);
    assert(loop_off_bytes < COMMON_VARS_SIZE_BYTES);
    int32_t* glob_loop = (int32_t*) &shared_mem[loop_off_bytes];

    // printf("????????????    v: %d    ei_off: %d    loop_off: %d\n", i_ptr, ei_to_ci_off_bytes, loop_off_bytes);


    __syncthreads();

    // ok, data initialized, let the run start
    uint32_t mySlot = HA::insertInt(hashComm, comm[j], comm[j], hasharrayEntries);
    float sum = HA::addFloatSpecificPos(hashWeight, mySlot, W[V[i] + edge_ptr]);

    if (i == 6) {
        printf("AAA>>>: dodałem %f (%d) do hash[%d], that belongs to comm %d\n", W[V[i] + edge_ptr], j, mySlot, comm[j]);
    }


    float loop = i == j ? W[V[i] + edge_ptr] : 0;
    float ei_to_Ci = comm[j] == comm[i] ? hashWeight[mySlot].value : 0;

    atomicMax(glob_loop, float_to_int(loop));
    atomicMax(glob_ei_to_Ci, float_to_int(ei_to_Ci));

    __syncthreads();

    if (edge_ptr == 0) {
        loop = int_to_float(*glob_loop);
        ei_to_Ci = int_to_float(*glob_ei_to_Ci);
        printf("!!!!!!!!!!  %d:  global loop: %f,   global ei_to_ci: %f\n", i, loop, ei_to_Ci);
    }

    if (edge_ptr == 0) {
        ei_to_Ci -= loop;
    }

    float deltaModRaw = comm[j] >= comm[i] ? 
        -(1 << 5) : 
        k[i] * ( ac[comm[i]] - k[i] - ac[comm[j]] ) / (2 * m * m)  +  hashWeight[mySlot].value / m;

    uint32_t newCommIdx = comm[j];

    if (i == 11) {
        if (j == 6 || j == 10) {
            printf("GREP: %d, delta: %f\n", j, deltaModRaw);
        }
    }

    if (i == 6) {
        printf("AAABBB: hashWeight[%d].value: %f\n", mySlot, hashWeight[mySlot].value);
    }
    
    // if (deltaModRaw > 0)
    //     printf("***%d, deltaMod: %f, a sprawdzam %d\n", i, deltaModRaw, newCommIdx);

    // Now, we must perform reduction on deltaMod values, to find best newCommIdx.
    // It's kinda hacky, cause in case of obtaining two identical deltaMod values,
    // we choose community with lower idx. It is accomplished by finding maximal value
    // of pair (int(deltaMod), -newCommIdx). We implement it using concatenation
    // of two unsigned values (obtained by order-preserving bijection from floats).
    uint32_t deltaMod_off_bytes = loop_off_bytes + sizeof(float);
    uint64_t* glob_deltaMod = (uint64_t*) &shared_mem[deltaMod_off_bytes];
    uint32_t newCommIdxRepr = -1 - newCommIdx; // bits flipped

    assert(-1 - newCommIdxRepr == newCommIdx);

    // LOL
    // printf("deltaMod: %f, toInt: %d, to uint: %u\n", deltaModRaw, float_to_int(deltaModRaw), int_to_uint(float_to_int(deltaModRaw)));
    // printf("deltaMod: %f, toInt: %d, to uint: %u, TO RAW::::%f\n", deltaModRaw, float_to_int(deltaModRaw), float_to_uint(deltaModRaw), uint_to_float(float_to_uint(deltaModRaw)));
    // printf("comm idx repr: %u\n", newCommIdxRepr);

    // LOL
    // uint64_t deltaMod = (((uint64_t) int_to_uint(float_to_int(deltaModRaw))) << 31) + newCommIdxRepr;
    uint64_t deltaMod = (((uint64_t) float_to_uint(deltaModRaw)) << 31) | newCommIdxRepr;

    // printf("moj deltamod: %llu oraz test: \n", deltaMod);
    assert(sizeof(uint64_t) == sizeof(unsigned long long int));

    atomicMax((unsigned long long int*) glob_deltaMod, (unsigned long long int)deltaMod);

    __syncthreads();

    if (edge_ptr == 0) {
        if (i == 6) {
            printf("AAABBB: ei - loop: %f, loop: %f\n", ei_to_Ci, loop);
        }
        // printf("totalmax: %llu\n", *glob_deltaMod);
        // float deltaModBest = int_to_float(uint_to_int((uint32_t)(*glob_deltaMod >> 32)));

        // TODO, commented one and line below aren't equivalent, it breaks for negative floats.
        // float deltaModBest = uint_to_float((uint32_t)(*glob_deltaMod >> 31));
        float deltaModBest = int_to_float(uint_to_int((uint32_t)(*glob_deltaMod >> 31)));
        uint32_t commIdxBest = (uint32_t) -1 - ( (uint32_t) (( (uint64_t) 0xffffffff) & *glob_deltaMod) );
        printf("%d NAJW:: best mod: %f,  best comm: %d\n",  i, deltaModBest, commIdxBest); 

        float gain = deltaModBest - ei_to_Ci / m;
        printf(" ~~~~~~~~~~~~~~~~ *MAX_GAIN for %d: %f (to node %d)\n", i, gain, commIdxBest);
        if (gain > 0 && commIdxBest < comm[i]) {
            newComm[i] = commIdxBest;
            printf("~~~~~~~~~~~~~~~~ *%u: wywalam do communiy %u, bo gain %f\n", i, newCommIdx, gain);
        } else {
            newComm[i] = comm[i];
        }
    }
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

    assert(next_2_pow(maxDegree) == maxDegree); // maxDegree is power of 2
    
    int i_ptr = threadIdx.y + (blockIdx.y * blockDim.y); // my node pointer
    int edgeNum = threadIdx.x; // my edge pointer
    int i = binNodes[i_ptr];

    // printf("XXXXXXXXXXXXXXXX     x + y: %u + %u ||| ||  %d\n", i_ptr, edgeNum, getGlobalIdx());

    uint32_t j = E[V[i] + edgeNum];
    // printf(">> %d: (%d, %d): ogarniam sąsiada: %u\n", i, i_ptr, edgeNum, j);


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

    if (i == 6) {
        printf("AAA>>>: dodałem %f (%d) do hash[%d], that belongs to comm %d\n", W[V[i] + edgeNum], j, mySlot, comm[j]);
    }

    __syncwarp();

    uint32_t mask = __ballot_sync(FULL_MASK, edgeNum < maxDegree / 2);
    assert(mask == __ballot_sync(__activemask(), edgeNum < maxDegree / 2)); // TODO - to się kiedyś wysypie i dobrze, wtedy podmienić mask

    if (i_ptr + edgeNum == 0) {
        binprintf(mask);
        printf("\n");
        binprintf(1 << 5);
    }
    

    // printf("WAGA: (comm %d, comm %d) = %f\n", comm[i], comm[j], hashWeight[mySlot].value);

    // sum of weights from node i to Ci\{i} from paper doesn't use loop values
    float loop = i == j ? W[V[i] + edgeNum] : 0;
    float ei_to_Ci = comm[j] == comm[i] ? hashWeight[mySlot].value : 0;

    float todo = i == j ? W[V[i] + edgeNum] : 0;

    __syncwarp();

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
        printf("AAABBB: hashWeight[%d].value: %f\n", mySlot, hashWeight[mySlot].value);
    }
    
    if (deltaMod > 0)
        printf("***%d, deltaMod: %f dla %d\n", i, deltaMod, newCommIdx);

    if (i == 11) {
        if (j == 6 || j == 10) {
            printf("GREP: %d, delta: %f\n", j, deltaMod);
        }
    }
    

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
            printf("AAABBB: ei - loop: %f, loop: %f\n", ei_to_Ci, loop);
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

__host__ 
float reassign_communities_bin(
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
    // TODO to tez jest zle
    uint32_t threadNum = std::min(binNodesNum, hashArrayEntriesPerBlock / (2 * hashArrayEntriesPerComm));

    uint32_t blockNum = ceil(binNodesNum / threadNum);

    dim3 dimBlock(maxDegree, threadNum); // x per edges, y per nodes

    printf("maxDeg: %u \nall nodes in bucket: %u \nthreadDim:(%u, %u) \nblockNum:%u\n", 
        maxDegree, binNodesNum, threadNum, maxDegree, blockNum);

    // printf("debug: \nhashArrayEntriesPerNode: %u \nhashArrayEntriesPerBlock: %u\n", hashArrayEntriesPerComm, hashArrayEntriesPerBlock);
    printf(">>>> KERNEL RUNNING!\n\n");


    // TODO to jest źle
    uint32_t shm = threadNum * VAR_MEM_PER_VERTEX_BYTES_DEFINE + (2 * hashArrayEntriesPerComm) * sizeof(KeyValueInt) * threadNum;

    if (shm > SHARED_MEM_SIZE)
        assert(false);


    // reassign_huge_nodes<<<blockNum, dimBlock, shm>>> (binNodesNum, binNodes, 
    //         V, E, W, k, ac, comm, newComm, maxDegree, threadNum, hashArrayEntriesPerComm, m, nullptr);
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

        if (my_com == comj) {
            atomicAdd(ei_to_Ci, W[V[me] + i]);
        }
    }
}

__host__ 
float __computeMod(float ei_to_Ci_sum, float m, const float* ac, uint32_t V_MAX_IDX) {
    auto tmp = thrust::device_vector<float>(V_MAX_IDX + 1);
    thrust::transform(ac, ac + V_MAX_IDX + 1, tmp.begin(), thrust::square<float>());
    float sum = thrust::reduce(tmp.begin(), tmp.end(), (float) 0, thrust::plus<float>());
    printf("sum: %f\n", sum);

    return ei_to_Ci_sum / (2 * m) - ( sum / (4 * m * m));
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
                        uint32_t* __restrict__ V, 
                        uint32_t* __restrict__ E,
                        float*    __restrict__ W,
                        float*    __restrict__ k,
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
    auto _it0 = thrust::partition(G.begin(), G.end(), partitionGenerator(1));
    auto it = _it0, it0 = _it0;

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

        it0 = _it0;

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

        printf("MOD1 %f\n", mod1);

        return mod1;

        if (mod1 - mod0 < 0) {
            printf("MOD DEBUG: %f\n", mod1 - mod0);
            // exit(1);
            assert(false);
        } else if (mod1 == mod0) {
            return mod1; // TODO
        }
        else if (mod1 - mod0 < minGain) {
            contract(V_MAX_IDX, V, E, W, k, comm);
            printf("\n*****************                 CONTRACT                 ****************\n");
            print_comm_assignment(V_MAX_IDX, comm);
            cudaDeviceSynchronize();
            mod0 = mod1;

            printf("COMMS END: \n");
            for(int i = 1; i <= V_MAX_IDX; i++) {
                printf("%d ", comm[i]);
            }
            printf("\n");

            float mod_debug = computeModAndAC(V_MAX_IDX, V, E, W, k, comm, ac, m);

            printf("MOD CONTRCT: %f\n", mod_debug);
            return mod1; // TODO debug  remove, up to first constract
            // break; // TODO, poprawić contract
        } else {
            printf("going to next modularity iteration (mod gain sufficient): mod 0, 1: %f, %f\n", mod0, mod1);
            mod0 = mod1;
        }
    }

    return mod1;
}
