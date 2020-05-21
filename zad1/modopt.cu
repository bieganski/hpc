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


__host__
void contract(const uint32_t V_MAX_IDX,
                          uint32_t* __restrict__ V, 
                          uint32_t* __restrict__ E,
                          float*    __restrict__ W,
                          float*    __restrict__ k,
                          const uint32_t* __restrict__ comm,
                          thrust::device_vector<uint32_t>& globalCommAssignment);


using HA = HashArray;

// [ BINS[i], BINS[i+1] ) (right side exclusive)
__device__ uint32_t BINS[] =  
    {
        0, // [0,1) is handled separately (lonely nodes; their modularity impact is 0)
        4,
        8,
        16,
        32,
        96,
        320,
        1024,
        UINT32_MAX // hash arrays in global memory
    };

__global__
void computeAC(const uint32_t V_MAX_IDX,
                const float* __restrict__ k,
               float*       __restrict__ ac,
               const uint32_t*    __restrict__ comm) {

    int tid = 1 + threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid > V_MAX_IDX)
        return;
    atomicAdd(&ac[comm[tid]], k[tid]);
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

    int i_ptr = threadIdx.y + blockIdx.x * blockDim.y; // my vertex pointer
    int edge_ptr = threadIdx.x; // my edge pointer

    assert(edge_ptr < maxDegree);

    KeyValueFloat* hashWeight;
    KeyValueInt*   hashComm;

    // TODO: customize these
    // VARIABLES:  ei_to_Ci, loop, deltaModRes
    uint32_t VAR_MEM_PER_VERTEX_BYTES = sizeof(float) + sizeof(float) + sizeof(uint64_t);

    assert(VAR_MEM_PER_VERTEX_BYTES_DEFINE == VAR_MEM_PER_VERTEX_BYTES);

    uint32_t COMMON_VARS_SIZE_BYTES = VAR_MEM_PER_VERTEX_BYTES * nodesPerBlock;

    // TODO customize this
    bool shared = globalHasharray == nullptr;

    // very careful pointer handling here, because of lots of type mismatches and casting
    if (shared) {
        uint32_t off = COMMON_VARS_SIZE_BYTES + (i_ptr % nodesPerBlock) * (2 * hasharrayEntries) * sizeof(KeyValueFloat);
        hashWeight = (KeyValueFloat*) (&shared_mem[off]);
        assert( shared_mem + COMMON_VARS_SIZE_BYTES <= (char*) hashWeight);
        hashComm   = (KeyValueInt*)   hashWeight + hasharrayEntries;
        assert(globalHasharray == nullptr);
    } else {
        hashWeight = ( (KeyValueFloat*) globalHasharray ) + i_ptr * (2 * hasharrayEntries);
        hashComm   = (KeyValueInt*)   hashWeight + hasharrayEntries;
    }

    // before any early return, let's utilize all threads for zeroing memory.

    uint64_t* __tmp = (uint64_t*) shared_mem;
    for (int i =0; i < COMMON_VARS_SIZE_BYTES / 8; i++) {
        __tmp[i] = 0;
    }

    for (int i = edge_ptr; i < hasharrayEntries; i += maxDegree) {
        hashWeight[i] = {.key = hashArrayNull, .value = (float) 0}; // 0 for easy atomicAdd
        hashComm[i]   = {.key = hashArrayNull, .value = hashArrayNull};
    }

    if (numNodes -1 < i_ptr) {
        return;
    }
    int i = binNodes[i_ptr]; // my vertex

    if (V[i + 1] - V[i] -1 < edge_ptr) {
        return;
    }
    uint32_t j = E[V[i] + edge_ptr]; // my neighbor

    // variables common for each vertex, accumulating ei_to_Ci value 
    // computed in parallel
    uint32_t ei_to_ci_off_bytes = (i_ptr % nodesPerBlock) * VAR_MEM_PER_VERTEX_BYTES;
    int32_t* glob_ei_to_Ci = (int32_t*) &shared_mem[ei_to_ci_off_bytes];
    
    uint32_t loop_off_bytes = ei_to_ci_off_bytes + sizeof(int32_t);
    assert(ei_to_ci_off_bytes < loop_off_bytes);
    assert(loop_off_bytes < COMMON_VARS_SIZE_BYTES);
    int32_t* glob_loop = (int32_t*) &shared_mem[loop_off_bytes];

    __syncthreads();

    // ok, data initialized, let the run start
    uint32_t mySlot = HA::insertInt(hashComm, comm[j], comm[j], hasharrayEntries);
    float sum = HA::addFloatSpecificPos(hashWeight, mySlot, W[V[i] + edge_ptr]);

    float loop = i == j ? W[V[i] + edge_ptr] : 0;
    float ei_to_Ci = comm[j] == comm[i] ? hashWeight[mySlot].value : 0;

    atomicMax(glob_loop, float_to_int(loop));
    atomicMax(glob_ei_to_Ci, float_to_int(ei_to_Ci));

    __syncthreads();

    if (edge_ptr == 0) {
        loop = int_to_float(*glob_loop);
        ei_to_Ci = int_to_float(*glob_ei_to_Ci);
        ei_to_Ci -= loop;
    }

    float deltaModRaw = comm[j] >= comm[i] ? 
        -(1 << 5) : 
        k[i] * ( ac[comm[i]] - k[i] - ac[comm[j]] ) / (2 * m * m)  +  hashWeight[mySlot].value / m;

    uint32_t newCommIdx = comm[j];

    // Now, we must perform reduction on deltaMod values, to find best newCommIdx.
    // It's kinda hacky, cause in case of obtaining two identical deltaMod values,
    // we choose community with lower idx. It is accomplished by finding maximal value
    // of pair (int(deltaMod), -newCommIdx). We implement it using concatenation
    // of two unsigned values (obtained by order-preserving bijection from floats).
    uint32_t deltaMod_off_bytes = loop_off_bytes + sizeof(float);
    uint64_t* glob_deltaMod = (uint64_t*) &shared_mem[deltaMod_off_bytes];
    assert(newCommIdx != UINT32_MAX);
    assert(newCommIdx != 0);
    uint32_t newCommIdxRepr = -1 - newCommIdx; // bits flipped

    assert(-1 - newCommIdxRepr == newCommIdx);

    uint64_t deltaMod = (((uint64_t) float_to_uint(deltaModRaw)) << 31) | newCommIdxRepr;

    assert((uint32_t)deltaMod != 0);

    assert(sizeof(uint64_t) == sizeof(unsigned long long int));

    atomicMax((unsigned long long int*) glob_deltaMod, (unsigned long long int)deltaMod);

    __syncthreads();

    if (edge_ptr == 0) {
        // TODO, commented one and line below aren't equivalent, it breaks for negative floats.
        // float deltaModBest = uint_to_float((uint32_t)(*glob_deltaMod >> 31));
        float deltaModBest = int_to_float(uint_to_int((uint32_t)(*glob_deltaMod >> 31)));
        uint64_t test = *glob_deltaMod;
        uint32_t commIdxBest = (uint32_t) -1 - (uint32_t) (test & UINT32_MAX);

        float gain = deltaModBest - ei_to_Ci / m;

        if (gain > 0 && commIdxBest < comm[i]) {
            assert(commIdxBest > 0);
            newComm[i] = commIdxBest;
        } else {
            newComm[i] = comm[i];
        }
    }
}

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
    
    int i_ptr =  threadIdx.y + blockIdx.x * blockDim.y; // my node pointer
    int edgeNum = threadIdx.x; // my edge pointer

    if (numNodes -1 < i_ptr) {
        return;
    }

    int i = binNodes[i_ptr];

    if (V[i + 1] - V[i] -1 < edgeNum) {
        return;
    }

    uint32_t j = E[V[i] + edgeNum];

    // each hashtable contains of `hasharrayEntries` elements
    KeyValueFloat* hashWeight = (KeyValueFloat*) hashtables + (i_ptr % nodesPerBlock) * (2 * hasharrayEntries);
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

    uint32_t mask = __ballot_sync(FULL_MASK, edgeNum < maxDegree / 2);
    
    // sum of weights from node i to Ci\{i} from paper doesn't use loop values
    float loop = i == j ? W[V[i] + edgeNum] : 0;
    float ei_to_Ci = comm[j] == comm[i] ? hashWeight[mySlot].value : 0;

    float todo = i == j ? W[V[i] + edgeNum] : 0;

    __syncwarp();

    for (int offset = maxDegree / 2; offset > 0; offset /= 2) {
        ei_to_Ci = fmaxf(ei_to_Ci, __shfl_down_sync(mask, ei_to_Ci, offset)); // only warp with idx % maxDegree == 0 keeps proper value
        loop = fmaxf(loop, __shfl_down_sync(mask, loop, offset));

        // todo += __shfl_down_sync(mask, todo, offset);
    }
   

    // TODO important sanity check (asserts no multi-loops)
    // if (edgeNum == 0) {
    //     assert(todo == loop);
    //     ei_to_Ci -= loop;
    // }

    // lack of -(e_i -> C_i\{i} / m) addend in that sum, it will be computed later
    float deltaMod = comm[j] >= comm[i] ? 
        -(1 << 5) : 
        k[i] * ( ac[comm[i]] - k[i] - ac[comm[j]] ) / (2 * m * m)  +  hashWeight[mySlot].value / m;

    uint32_t newCommIdx = comm[j];

    for (int offset = maxDegree / 2; offset > 0; offset /= 2) {
        float deltaModRed = __shfl_down_sync(mask, deltaMod, offset);
        uint32_t newCommIdxRed = __shfl_down_sync(mask, newCommIdx, offset);

        if (newCommIdxRed == 0)
            continue; // TODO - brzydki hack na undefined behavior __shfl_down_sync

        if (deltaModRed > deltaMod) {
            deltaMod = deltaModRed;
            newCommIdx = newCommIdxRed;
        } else if (deltaModRed == deltaMod) {
            newCommIdx = (uint32_t) fminf((uint32_t) newCommIdx, (uint32_t) newCommIdxRed);
        }
    }

    if (edgeNum == 0) {
        float gain = deltaMod - ei_to_Ci / m;

        if (gain > 0 && newCommIdx < comm[i]) {
            assert(newCommIdx != 0);
            newComm[i] = newCommIdx;
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

    // TODO customize this, maybe check 2 * maxDegree?
    uint32_t hashArrayEntriesPerComm;

    assert(sizeof(KeyValueFloat) == sizeof(KeyValueInt));

    uint32_t threadsX = maxDegree;
    uint32_t maxThreadsY = 1024 / threadsX;
    uint32_t threadsY = min(maxThreadsY, binNodesNum);
    uint32_t blockNum = ceil( (float)binNodesNum / threadsY );

    // assert(blockNum * threadsY >= binNodesNum); // TODO

    dim3 dim(maxDegree, threadsY);
    
    if (maxDegree == 1024 || maxDegree == UINT32_MAX) {
        hashArrayEntriesPerComm = next_2_pow(4096);
        
        float* globalHashArrays;
        HANDLE_ERROR(cudaHostAlloc((void**)&globalHashArrays, sizeof(KeyValueFloat) * binNodesNum * (2 * hashArrayEntriesPerComm), cudaHostAllocDefault));
        std::memset(globalHashArrays, '\0', sizeof(KeyValueFloat) * binNodesNum * (2 * hashArrayEntriesPerComm));
        HANDLE_ERROR(cudaHostGetDevicePointer(&globalHashArrays, globalHashArrays, 0));

        uint32_t shmBytes = threadsY * VAR_MEM_PER_VERTEX_BYTES_DEFINE;
        assert(shmBytes <= SHARED_MEM_SIZE);

        reassign_huge_nodes<<<blockNum, dim, shmBytes>>> (binNodesNum, binNodes, 
            V, E, W, k, ac, comm, newComm, maxDegree, threadsY, hashArrayEntriesPerComm, m, globalHashArrays);

        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaFreeHost(globalHashArrays));
    } else {
        hashArrayEntriesPerComm = next_2_pow(maxDegree); // TODO koniecznie sprawdzic +1

        if (maxDegree <= WARP_SIZE) {
            uint32_t shmBytes = (2 * hashArrayEntriesPerComm) * sizeof(KeyValueInt) * threadsY;
            assert(shmBytes <= SHARED_MEM_SIZE);

            reassign_nodes<<<blockNum, dim, shmBytes>>>      (binNodesNum, binNodes, 
                V, E, W, k, ac, comm, newComm, maxDegree, threadsY, hashArrayEntriesPerComm, m);
        } else {
            uint32_t shmBytes = threadsY * VAR_MEM_PER_VERTEX_BYTES_DEFINE + (2 * hashArrayEntriesPerComm) * sizeof(KeyValueInt) * threadsY;
            assert(shmBytes <= SHARED_MEM_SIZE);

            reassign_huge_nodes<<<blockNum, dim, shmBytes>>> (binNodesNum, binNodes, 
                V, E, W, k, ac, comm, newComm, maxDegree, threadsY, hashArrayEntriesPerComm, m, nullptr);
        }
    }
    return 21.37;
}

__global__ 
void computeEiToCiSum(uint32_t V_MAX_IDX,
                        float*    __restrict__ ei_to_Ci,
                         const uint32_t* __restrict__ V,
                         const uint32_t* __restrict__ E,
                         const float*    __restrict__ W,
                         const uint32_t* __restrict__ comm) {
    uint32_t me = 1 + getGlobalIdx();

    if (me > V_MAX_IDX)
        return;

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
    float sum = thrust::reduce(tmp.begin(), tmp.end(), (double) 0, thrust::plus<double>());

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

    computeEiToCiSum <<<all_nodes_pair.first, all_nodes_pair.second>>> (V_MAX_IDX, ei_to_Ci, V, E, W, comm);
    cudaDeviceSynchronize();

    zeroAC(ac, V_MAX_IDX);
    computeAC<<<all_nodes_pair.first, all_nodes_pair.second>>> (V_MAX_IDX, k, ac, comm);
    cudaDeviceSynchronize();

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
                        const float minGain,
                        thrust::device_vector<uint32_t>& globCommAssignment) {

    uint32_t* binsHost = (uint32_t*) malloc(sizeof(BINS));
    cudaMemcpyFromSymbol(binsHost, BINS, sizeof(BINS), 0, cudaMemcpyDeviceToHost);


    thrust::device_vector<uint32_t> G(V_MAX_IDX);
    thrust::sequence(G.begin(), G.end(), 1);

    // when running with --verbose option, we must keep proper community mapping
    // (community indeices are reassigned during contract phase) 
    thrust::sequence(globCommAssignment.begin(), globCommAssignment.end());

    auto partitionGenerator = [=](int rightIdx) {
        return [=] __device__ (const uint32_t& i) {
            return V[i + 1] - V[i] <= BINS[rightIdx];
        };
    };

    float mod0, mod1, maxMod;    
    mod0 = computeModAndAC(V_MAX_IDX, V, E, W, k, comm, ac, m);
    maxMod = mod0;

    bool changedSth = true;

    while(true) {

        // [0,1) is handled separately (lonely nodes; their modularity impact is 0)
        auto it0 = thrust::partition(G.begin(), G.end(), partitionGenerator(0));

        // for each bin sequentially computes new communities
        for (int i = 1; ; i++) {
            auto it = thrust::partition(it0, G.end(), partitionGenerator(i));
            uint32_t maxDegree = binsHost[i];
            
            uint32_t binNodesNum = thrust::distance(it0, it);
            if (binNodesNum == 0)
                break;

            uint32_t* binNodes = RAW(it0);

            reassign_communities_bin(binNodes, binNodesNum, V, E, W, k, ac, comm, newComm, maxDegree, m);

            cudaDeviceSynchronize();

            auto pair = getBlockThreadSplit(binNodesNum);

            // update newComm table
            updateSpecific<<<pair.first, pair.second>>> (binNodes, binNodesNum, newComm, comm, V);
            cudaDeviceSynchronize();

            // recompute AC values
            zeroAC(ac, V_MAX_IDX);
            computeAC<<<pair.first, pair.second>>> (V_MAX_IDX, k, ac, comm);
            cudaDeviceSynchronize();

            it0 = it;
        }

        // OK, we computed new communities for all bins, let's check whether
        // modularity gain is satisfying

        mod1 = computeModAndAC(V_MAX_IDX, V, E, W, k, comm, ac, m);

        maxMod = max(maxMod, mod1);

        if (abs(mod1 - mod0) <= 0.001) {
            if (!changedSth) {
                return maxMod;
            } else {
                contract(V_MAX_IDX, V, E, W, k, comm, globCommAssignment);
                changedSth = false;
            }
        } else if (mod1 - mod0 < minGain) {
            contract(V_MAX_IDX, V, E, W, k, comm, globCommAssignment);
            changedSth = false;
            cudaDeviceSynchronize();
            mod0 = mod1;
        } else {
            changedSth = true;
            mod0 = mod1;
        }
    }

    return mod1;
}
